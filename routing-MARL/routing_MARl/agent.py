from Q_network import QNetwork
from experience_buffer import JointExperienceReplayBuffer, PrioritizedReplayBuffer
from logs.logger import logger  
from read_config import get_config

import torch
import torch.nn.functional as F 
import torch.optim as optim
import os
import random
import numpy as np
import time

class CompactObs:
    """Observation stored in the buffer (S3): [edge index, one-hot of the next roads, density] without the zeros,
    plus the car's last K road indices when the memory branch is on (the last K entries of the observation)."""
    __slots__ = ('edge', 'ones', 'density', 'hist', 'extra')
    K = 0  # memory window length, set by Agent from config memory.window
    X = 0  # extra float features after the density (1 = the delay D(e)/T_max), set by Agent from config delay.enabled

    def __init__(self, obs):
        v = torch.as_tensor(obs, dtype=torch.float32).reshape(-1)
        self.hist = None
        self.extra = None
        if CompactObs.K:
            self.hist = v[-CompactObs.K:].round().numpy().astype(np.int16)
            v = v[:-CompactObs.K]
        if CompactObs.X:
            self.extra = v[-CompactObs.X:].numpy().astype(np.float32)
            v = v[:-CompactObs.X]
        mid = v[1:-1]
        ones = torch.nonzero(mid).flatten()
        if not bool(torch.all(mid[ones] == 1.0)):
            raise ValueError("compact_obs: the one-hot part of the observation holds values other than 0/1")
        self.edge, self.density = float(v[0]), float(v[-1])
        self.ones = ones.numpy().astype(np.int32)


def compact_vehicles(vehicles):
    """Member tuples with obs (index 1) and next_obs (index 3) replaced by CompactObs."""
    out = []
    for v in vehicles:
        v = list(v)
        v[1] = CompactObs(v[1])
        v[3] = CompactObs(v[3])
        out.append(tuple(v))
    return out


class Agent:  
    def __init__(self, obs_dim, action_dim, group_id, train_offline=False, config=None):

        # train offline from anothẻ folder
        if train_offline:
            self.CONFIG = config
        # train online with sumo interaction
        else:
            self.CONFIG = get_config()

        # Set up device
        # add cuda
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        logger.info(f"Agent {group_id} initialized using device: {self.device}")

        self.group_id = group_id  
        # switch to device processing
        # add cuda
        self.q_network = QNetwork(obs_dim, action_dim).to(self.device)
        # route baseline: shrink the last layer so the residual starts near 0
        head_scale = float((self.CONFIG.get('baseline') or {}).get('head_init_scale', 1.0))
        if head_scale != 1.0:
            with torch.no_grad():
                self.q_network.output_layer.weight.mul_(head_scale)
                self.q_network.output_layer.bias.mul_(head_scale)
        self.target_network = QNetwork(obs_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())  # Initial sync
        # Q = B + residual: B = fixed route-baseline vector of this group's shelter (set_baseline), None = off
        self.B = None
        self.changed_t = None   # decisions where argmax(B + res) != argmax(B) (training, GPU counter)
        self.changed_n = 0

        # gradient parameters
        self.max_grad_norm = float(self.CONFIG.get('max_grad_norm', 1.0))
        self.weight_decay = float(self.CONFIG.get('weight_decay', 1e-4))

        # optimizer: 'adam' (default, L2 decay inside Adam) or 'adamw' (decoupled decay, none on biases/LayerNorm)
        if str(self.CONFIG.get('optimizer', 'adam')).lower() == 'adamw':
            decay = [p for p in self.q_network.parameters() if p.dim() >= 2]
            no_decay = [p for p in self.q_network.parameters() if p.dim() < 2]
            self.optimizer = optim.AdamW([{'params': decay, 'weight_decay': self.weight_decay},
                                          {'params': no_decay, 'weight_decay': 0.0}],
                                         lr=self.CONFIG['learning_rate'])
        else:
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.CONFIG['learning_rate'],  
                                        weight_decay=self.weight_decay)  
        self.loss_reduction = str(self.CONFIG.get('loss_reduction', 'sum')).lower()
        self.reward_scale = float(self.CONFIG.get('reward_scale', 1.0))
        # per-episode learning-rate schedule: 'none' (default) or 'warmup_cosine'
        self.lr_scheduler = None
        if str(self.CONFIG.get('lr_schedule', 'none')).lower() == 'warmup_cosine':
            import math
            warm = int(self.CONFIG.get('lr_warmup_episodes', 2))
            total = int(self.CONFIG.get('lr_total_episodes', 100))
            floor = float(self.CONFIG.get('lr_min_factor', 0.1))
            def factor(ep):
                if ep < warm:
                    return (ep + 1) / warm
                t = min(1.0, (ep - warm) / max(1, total - warm))
                return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t))
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, factor)
        # per-episode training stats (kept on the GPU, read once per episode): unweighted loss, mean weight
        self.reset_train_stats()
        # how many vehicles decide together per stored group transition (N = 1, 2, 3, 4+)
        self.decision_counts = [0, 0, 0, 0]

        self.gamma = self.CONFIG['gamma']  # Discount factor
        # Use joint replay buffer
        # self.replay_buffer = JointExperienceReplayBuffer(capacity=self.CONFIG['buffer_capacity'])

        # replay.type: 'mixture' (default, old 60/20/20 buffer) | 'per' (Schaul et al., 2016)
        replay_cfg = self.CONFIG.get('replay') or {}
        self.replay_type = str(replay_cfg.get('type', 'mixture')).lower()
        # S3: store observations compactly in the buffer and rebuild them on the GPU in update()
        self.compact_obs = bool(replay_cfg.get('compact_obs', False))
        self.obs_dim = obs_dim
        mem = self.CONFIG.get('memory') or {}
        self.mem_k = int(mem.get('window', 8)) if mem.get('enabled') else 0   # last mem_k obs entries = road window
        CompactObs.K = self.mem_k
        self.extra_dim = 1 if (self.CONFIG.get('delay') or {}).get('enabled') else 0
        CompactObs.X = self.extra_dim
        # reward_mode: 'legacy' (default, get_reward_5 per decision) | 'evac_time' (per-vehicle rewards)
        self.reward_mode = str(self.CONFIG.get('reward_mode', 'legacy')).lower()
        gps = self.CONFIG.get('gamma_per_second')
        self.gamma_per_second = float(gps) if (gps is not None and self.reward_mode == 'evac_time') else None
        if self.replay_type == 'per':
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=self.CONFIG['buffer_capacity'],
                alpha=replay_cfg.get('alpha', 0.6),
                beta_start=replay_cfg.get('beta_start', 0.4),
                beta_end=replay_cfg.get('beta_end', 1.0),
                eps=float(replay_cfg.get('eps', 1e-6)),
                uniform_ratio=replay_cfg.get('uniform_ratio', 0.0),
                done_ratio=replay_cfg.get('done_ratio', 0.0))
        else:
            self.replay_buffer = JointExperienceReplayBuffer(
                capacity=self.CONFIG['buffer_capacity'],
                alpha=0.6,
                # long term memory
                priority_ratio=0.60,
                # forget gate, so just random
                uniform_ratio=0.20,
                # recent memory
                recent_ratio=0.20,
                recent_window=150000,
            )


        self.current_size_buffer = 0
        
        # Temporary storage for current timestep
        self.current_timestep_transitions = []
        self.current_timestep = 0 

        self.batch_size = self.CONFIG['batch_size']

        # Epsilon-greedy parameters  
        self.epsilon_start = self.CONFIG['epsilon_greedy']['initial']
        self.epsilon_final = self.CONFIG['epsilon_greedy']['final']
        self.epsilon_decay = self.CONFIG['epsilon_greedy']['decay']
        self.total_steps = 0  

        #  UCB parameters
        self.action_counts = np.zeros(action_dim, dtype=np.int64)  # Initialize with zeros
        self.c = self.CONFIG['UCB']['c']  # Exploration parameter for UCB

        # Softmax parameters
        self.tau = self.CONFIG['softmax']['tau']  # Temperature parameter for softmax action selection
        # B2: tau falls linearly from softmax.tau to softmax.tau_end over softmax.tau_decay_episodes (off: constant)
        self.tau_start = float(self.tau)
        self.tau_end = self.CONFIG['softmax'].get('tau_end')
        self.tau_decay_episodes = int(self.CONFIG['softmax'].get('tau_decay_episodes') or 1)
        # B0: target_mask 'feasible' = Double-DQN argmax over the next state's feasible roads only (off: all roads)
        self.target_mask = str(self.CONFIG.get('target_mask') or '').lower() == 'feasible'
        # B1: n-step returns (members carry the discounted n-step reward sum and their step count)
        self.n_step = int(self.CONFIG.get('n_step', 1) or 1)
    
        # Target update frequency params  
        # self.target_update_freq = self.CONFIG['update_frequency']  
        self.update_step_count = 0  
        # log gradients every k-th update (config `grad_log_every`, default 1 = every update)
        self.grad_log_every = max(1, int(self.CONFIG.get('grad_log_every', 1)))
        self.beta_soft = self.CONFIG.get('beta_soft', 0.01)  # Soft update parameter
        # S5: update every k-th second with new decisions (update_every, default 1); the target net then follows
        # at the same speed per second with beta' = 1 - (1 - beta)^k. Set batch_size to k x the old value in the yaml.
        self.update_every = max(1, int(self.CONFIG.get('update_every', 1)))
        if self.update_every > 1:
            self.beta_soft = 1.0 - (1.0 - self.beta_soft) ** self.update_every
        self.update_ticks = 0
        # value-learning tests: Huber delta (None = smooth_l1, delta 1, as before)
        # and per-episode TD-error stats by transition type (td_debug)
        hd = self.CONFIG.get('huber_delta')
        self.huber_delta = float(hd) if hd is not None else None
        self.td_debug = bool(self.CONFIG.get('td_debug', False))
        self.td_acc = None
        # VDN + per-car TD term: loss = rho(delta_sum) + lambda * sum_i rho(delta_i);
        # priority 'max' = max(|delta_sum|, max_i |delta_i|). lambda 0 + priority 'group' = the VDN loss as before.
        vl = self.CONFIG.get('vdn_loss') or {}
        self.per_car_lambda = float(vl.get('per_car_lambda', 0.0) or 0.0)
        self.priority_mode = str(vl.get('priority', 'group')).lower()
        if self.priority_mode not in ('group', 'max'):
            raise ValueError(f"vdn_loss.priority must be 'group' or 'max', not {self.priority_mode!r}")
        self.vdn_stats_on = self.per_car_lambda > 0 or self.priority_mode == 'max'
        # replay.split_cars: store each car's decision as its own transition of
        # size 1 = standard independent DQN; the existing group term of the loss is then rho(delta_k) and the 'group'
        # priority |delta_k|. The per-car term must be off, or the same error would count twice. Default false = unchanged.
        self.split_cars = bool((self.CONFIG.get('replay') or {}).get('split_cars', False))
        if self.split_cars and (self.per_car_lambda > 0 or self.priority_mode != 'group'):
            raise ValueError("replay.split_cars needs vdn_loss.per_car_lambda: 0 and vdn_loss.priority: group")
        logger.info(f"Agent {group_id}: encoder {getattr(self.q_network, 'encoder_type', 'transformer') if getattr(self.q_network, 'use_encoder', True) else 'OFF'}, "
                    f"one car per sample {'ON' if self.split_cars else 'off'}")
        self.reset_vdn_stats()

        # configs for save and load model
        self.saved_folder = os.path.join(os.path.dirname(__file__), '..', 'new_models') 
        # Add training mode flag  
        self.training = True  

    def epsilon_by_step(self):  
        return self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                np.math.exp(-1. * self.total_steps / self.epsilon_decay) 

    
    def softmax_select_action(self, observation, feasible_actions, encoded_disaster_obs):
        """
        Softmax selection per vehicle for a single timestep.
        
        observation:  [1, obs_dim]
        encoded_disaster_obs:  [1, dis_dim]
        feasible_actions: list of bools of length action_dim
        """
        if not self.training:
            return self.select_action_for_inference(observation, feasible_actions, encoded_disaster_obs)

        if not feasible_actions:
            logger.warning("No feasible actions available. Do nothing.")
            return None

        with torch.no_grad():
            # Move inputs to device
            # add cuda
            observation = observation.to(self.device)
            if encoded_disaster_obs is not None:
                encoded_disaster_obs = encoded_disaster_obs.to(self.device)

            # Forward pass through Q-network
            disaster_shape = encoded_disaster_obs.shape if encoded_disaster_obs is not None else None
            logger.debug("shape: observation {}, encoded_disaster_obs {}".format(observation.shape, disaster_shape))
            
            # Shape obs(B, obs dim) and encoded_disaster_obs(B, dis dim) 
            # where B=1 because for single vehicle 

            # remove last dim to make encoded_disaster_obs = (B, dis dim) 
            if encoded_disaster_obs is not None:
                encoded_disaster_obs = encoded_disaster_obs.squeeze(-1)
            q_values = self.with_b(self.q_network(observation, encoded_disaster_obs))  # [1, action_dim]
            q_values = q_values.squeeze(0)                          # [action_dim]

            # Softmax over feasible actions
            feasible_indices = [i for i, f in enumerate(feasible_actions) if f]
            # mask infeasible actions, keep feasible only
            q_for_softmax = q_values[feasible_indices]
            self.count_changed(q_for_softmax, feasible_indices)

            # Numerical stability
            q_norm = q_for_softmax - torch.max(q_for_softmax)
            exp_q = torch.exp(q_norm / self.tau)
            prob = exp_q / exp_q.sum()

            # Sample action
            idx_within_feasible = torch.multinomial(prob, 1).item()
            action = feasible_indices[idx_within_feasible]

            self.total_steps += 1

            return action

    def batch_softmax_select_actions(self, observations, feasible_list, disaster_list):
        """S4: softmax_select_action for several vehicles of this group with one forward pass.
        Each vehicle is its own sequence of length 1 ([n, 1, obs_dim]), as in the per-vehicle call;
        a [n, obs_dim] input would make the vehicles attend to each other. Returns one action per vehicle."""
        if not self.training:
            return [self.select_action_for_inference(o, f, d) for o, f, d in zip(observations, feasible_list, disaster_list)]
        with torch.no_grad():
            obs = torch.cat([o.reshape(1, -1) for o in observations], dim=0).to(self.device).unsqueeze(1)    # [n, 1, obs_dim]
            dis = None
            if disaster_list[0] is not None:
                dis = torch.cat([d.squeeze(-1).reshape(1, -1) for d in disaster_list], dim=0).to(self.device).unsqueeze(1)
            q_all = self.with_b(self.q_network(obs, dis))                                              # [n, action_dim]
            samples, feasible_sets = [], []
            for q_values, feasible_actions in zip(q_all, feasible_list):
                feasible_indices = [i for i, f in enumerate(feasible_actions) if f]
                q_for_softmax = q_values[feasible_indices]
                self.count_changed(q_for_softmax, feasible_indices)
                q_norm = q_for_softmax - torch.max(q_for_softmax)
                exp_q = torch.exp(q_norm / self.tau)
                prob = exp_q / exp_q.sum()
                samples.append(torch.multinomial(prob, 1))
                feasible_sets.append(feasible_indices)
            picks = torch.cat(samples).tolist()                                                        # one sync
            self.total_steps += len(picks)
            return [fi[k] for fi, k in zip(feasible_sets, picks)]

    # def softmax_select_action(self, observation, feasible_actions):
    #     """
    #     Select action using Softmax action selection strategy.
    #     Works with TransformerQNetwork output (1D Q-values).
    #     """
    #     if not self.training:
    #         return self.select_action_for_inference(observation, feasible_actions)

    #     with torch.no_grad():
    #         # Compute Q-values: output shape [action_dim]
    #         q_values = self.q_network(observation)

    #         feasible_mask = torch.tensor(feasible_actions, dtype=torch.bool)

    #         # Mask infeasible actions
    #         q_values_masked = q_values.clone()
    #         q_values_masked[~feasible_mask] = float('-inf')

    #         # Get feasible indices
    #         feasible_indices = torch.nonzero(feasible_mask, as_tuple=False).squeeze(-1).tolist()
    #         if len(feasible_indices) == 0:
    #             logger.warning("No feasible actions available during softmax selection. Skipping.")
    #             return None

    #         # Extract feasible Q-values
    #         q_values_for_softmax = q_values[feasible_indices]

    #         # Subtract max(Q) for numerical stability
    #         q_norm = q_values_for_softmax - torch.max(q_values_for_softmax)

    #         # Softmax with temperature
    #         exp_q = torch.exp(q_norm / self.tau)
    #         prob = exp_q / exp_q.sum()

    #         # Sample action
    #         action_index_within_feasible = torch.multinomial(prob, num_samples=1).item()
    #         action_index = feasible_indices[action_index_within_feasible]

    #         self.total_steps += 1
    #         return action_index

    
    def UCB_select_action(self, observation, feasible_actions):
        """
        Select action using Upper Confidence Bound (UCB) strategy
        """
        # If not in training mode, use inference selection
        if not self.training:
            logger.success("Set inference mode for UCB")
            return self.select_action_for_inference(observation, feasible_actions)

        with torch.no_grad():
            # Compute Q-values for the observation
            q_values = self.q_network(observation.view(1, -1))
            
            # Create feasible mask
            feasible_mask = torch.tensor(feasible_actions, dtype=torch.bool).unsqueeze(0)
            
            # Mask infeasible actions
            # normalize q_values as there are negative values
            # q_values_normalized = (q_values - q_values.min()) / (q_values.max() - q_values.min() + 1e-8)
            q_values_masked = q_values.clone()
            q_values_masked[~feasible_mask] = float('-inf')
            
            # Prepare UCB scores
            ucb_scores = q_values_masked.clone()
            
            # Get feasible action indices
            feasible_indices = [i for i, f in enumerate(feasible_actions) if f]
            
            # Compute UCB scores for feasible actions
            for action in feasible_indices:
                # Skip masked actions
                if q_values_masked[0, action] == float('-inf'):
                    continue

                # logger.error(f"Values of q_values_masked {q_values_masked[0, action]}")
                
                # Initial exploration for never-selected actions
                if self.action_counts[action] == 0:
                    ucb_scores[0, action] = float('inf')
                    continue

                # Standard UCB formula
                # Q-value + exploration bonus
                exploration_bonus = self.c * torch.sqrt(
                    torch.log(torch.tensor(self.total_steps + 1, dtype=torch.float)) / 
                    (self.action_counts[action] + 1e-8)
                )

                # Add exploration bonus to existing Q-value 
                ucb_scores[0, action] += exploration_bonus 
            
            # Select action with highest UCB score
            action_index = torch.argmax(ucb_scores).item()
            
            # Update action counts and total steps
            self.action_counts[action_index] += 1
            self.total_steps += 1
            
            # Logging for debugging
            logger.info(f"UCB Action Selection")
            logger.info(f"Q-values: {q_values_masked}")
            logger.info(f"UCB Scores: {ucb_scores}")
            logger.info(f"Selected Action: {action_index}")
            # output all possible actions counts for each agent. At certain time, some actions are infeasible
            # so we only select feasible actions
            logger.info(f"Action Counts: {self.action_counts}") 
            
            return action_index
    
    def select_action(self, observation, feasible_actions): 
        """
        select action based on epsilon-greedy policy
        """ 
        # if current model is inference 
        if not self.training:  
            logger.debug("Set infernece mode")
            action_index = self.select_action_for_inference(observation, feasible_actions)
            return action_index

        # Training mode: use epsilon-greedy as before   
        epsilon = self.epsilon_by_step()  
        self.total_steps += 1  
        
        with torch.no_grad():  
            # the input from size [N,1] => [B, N]
            # B is the batch size, B = 1 in here
            q_values = self.q_network(observation.view(1, -1))  
            logger.info(f"observation: {observation}")
            feasible_mask = torch.tensor(feasible_actions, dtype=torch.bool).unsqueeze(0)  
            logger.info(f"shapes: {q_values.shape}, {feasible_mask.shape}")  
            logger.info(f"q_values: {q_values}")  
            
            # Mask infeasible actions: set to -inf so they won't be picked  
            q_values_masked = q_values.clone()  
            q_values_masked[~feasible_mask] = float('-inf')  
            logger.info(f"q_values_masked: {q_values_masked}")  

            if random.random() < epsilon:  
                # Exploration: random feasible action  
                feasible_indices = [i for i, f in enumerate(feasible_actions) if f]  
                action_index = random.choice(feasible_indices)  
                logger.info(f"Exploration: chose action {action_index} with epsilon {epsilon:.4f}")  
            else:  
                # Exploitation: greedy action  
                action_index = torch.argmax(q_values_masked).item()  
                logger.info(f"Exploitation: chose action {action_index} with epsilon {epsilon:.4f}")  

            return action_index  
        
    def select_action_for_inference(self, observation, feasible_actions, encoded_disaster_obs):
        # In evaluation mode
        if not self.training:  
            with torch.no_grad():  

                # Move inputs to device
                # add cuda
                observation = observation.to(self.device)
                if encoded_disaster_obs is not None:
                    encoded_disaster_obs = encoded_disaster_obs.to(self.device)
                    # (1, zones, 1) -> (1, zones), as in the training branch of softmax_select_action
                    if encoded_disaster_obs.dim() == 3:
                        encoded_disaster_obs = encoded_disaster_obs.squeeze(-1)
                
                q_values = self.with_b(self.q_network(observation, encoded_disaster_obs))   # [1, action_dim]
                q_values = q_values.squeeze(0)                          # [action_dim]

                return self.sample_inference(q_values, feasible_actions)
    
    @torch.no_grad()
    def sample_inference(self, q_values, feasible_actions):
        """Inference choice from one car's Q row [action_dim]: softmax over the feasible roads at inference_tau (default 1),
        sampled (or greedy); shared by ego-only and the R2.1 oracle."""
        feasible_mask = torch.tensor(feasible_actions, dtype=torch.bool)
        q_values_masked = q_values.clone()  
        q_values_masked[~feasible_mask] = float('-inf')  

        ### SOFTMAX SELECTION DURING INFERENCE ###
        # Compute probabilities only among feasible actions
        feasible_q_values = q_values_masked[feasible_mask]
        tau = float(self.CONFIG.get('inference_tau', 1) or 1)   # r21: inference temperature (1 = today's code)
        probs = torch.softmax(feasible_q_values if tau == 1.0 else feasible_q_values / tau, dim=0)

        # inference_action: 'softmax' (default, sample at T = 1) | 'greedy' (best Q)
        if self.CONFIG.get('inference_action', 'softmax') == 'greedy':
            idx_within_feasible = torch.argmax(probs).item()
        else:
            idx_within_feasible = torch.multinomial(probs, 1).item()

        # Map back to global action index
        feasible_indices = torch.nonzero(feasible_mask, as_tuple=True)[0]
        action_index = feasible_indices[idx_within_feasible].item()
            
        #   GREEDY SELECTION DURING INFERENCE ###   
        # action_index = torch.argmax(q_values_masked).item()  
        
        # Sample action
        logger.debug(f"Inference mode: chose action {action_index}")  
        return action_index  

    @torch.no_grad()
    def q_pool(self, observations, disasters):
        """R2.1 oracle: Q rows [n, action_dim] of n cars of this group encoded together as ONE set [1, n, obs_dim], as in the
        training update (attention across the cars, group_embed = their mean). Each car's disaster features become one row."""
        obs = torch.cat([o.reshape(1, -1) for o in observations], dim=0).to(self.device).unsqueeze(0)
        dis = None
        if disasters[0] is not None:
            dis = torch.cat([d.reshape(1, -1) for d in disasters], dim=0).to(self.device).unsqueeze(0)
        return self.with_b(self.q_network(obs, dis))

    @torch.no_grad()
    def q_ego(self, observations, disasters):
        """Q rows [n, action_dim] of the same n cars, each car alone (n sets of one), as when acting ego-only (flip log)."""
        obs = torch.cat([o.reshape(1, -1) for o in observations], dim=0).to(self.device).unsqueeze(1)
        dis = None
        if disasters[0] is not None:
            dis = torch.cat([d.reshape(1, -1) for d in disasters], dim=0).to(self.device).unsqueeze(1)
        return self.with_b(self.q_network(obs, dis))

    @torch.no_grad()
    def select_action_offline(self, observation, feasible_actions):
        """
        Select an action for each vehicle within the agent group
        """
        logits = self.q_network(observation.view(1,- 1))
        feasible_mask = torch.tensor(feasible_actions, dtype=torch.bool).unsqueeze(0)  
        logits_masked = logits.clone()
        logits_masked[~feasible_mask] = float('-inf')

        probs = F.softmax(logits_masked, dim=1)
        actions = torch.argmax(logits_masked, dim=1)

        return actions.item()
    
    def store_transition_timestep(self, vehicle_id, obs, action, next_obs, done, disaster_obs):
        """
        Store individual transition for current timestep.
        Call end_timestep() when all vehicles have acted.
        """
        self.current_timestep_transitions.append((vehicle_id, obs, action, next_obs, done, disaster_obs))
    
    def end_timestep(self, global_reward):
        """
        Called when all vehicles in the group have acted in current timestep.
        Stores the joint transition with global reward.
        """
        if len(self.current_timestep_transitions) > 0:
            if self.split_cars:
                raise ValueError("replay.split_cars needs per-vehicle transitions (reward_mode legacy_end or evac_time)")
            self.decision_counts[min(len(self.current_timestep_transitions), 4) - 1] += 1
            vehicles = self.current_timestep_transitions.copy()
            joint_transition = {
                'timestep': self.current_timestep,
                'vehicles': compact_vehicles(vehicles) if self.compact_obs else vehicles,
                'global_reward': global_reward
            }
            self.replay_buffer.push(joint_transition)
            
            # Reset for next timestep
            self.current_timestep_transitions = []
            self.current_timestep += 1

    def reset_train_stats(self):
        self.stat_updates, self.stat_done_picks, self.stat_picks = 0, 0, 0
        self.stat_loss_raw, self.stat_weight = 0.0, 0.0
        self.td_acc = None
        self.reset_vdn_stats()

    def reset_vdn_stats(self):
        # vdn_loss: per-episode sums, kept on the GPU (no sync per update)
        self.stat_loss_group, self.stat_loss_car, self.stat_gnorm, self.stat_clipped = 0.0, 0.0, 0.0, 0.0
        self.stat_hidden, self.stat_multi = 0.0, 0.0

    def _rho(self, q, y):
        """Huber loss per element; the same rule for the group and the per-car term."""
        if self.huber_delta is None:
            return F.smooth_l1_loss(q, y, reduction='none')
        return F.huber_loss(q, y, reduction='none', delta=self.huber_delta)

    def push_group_transition(self, joint_transition):
        """evac_time mode: store a finished group transition (members carry their own reward, dt, done)."""
        n = len(joint_transition['vehicles'])
        if n > 0:
            self.decision_counts[min(n, 4) - 1] += 1
            if self.split_cars:
                # one transition per car: its own (obs, action, next_obs, done, disaster, reward v[6], dt, next_dis, feas, steps)
                parts = [dict(joint_transition, vehicles=[v], global_reward=float(v[6])) for v in joint_transition['vehicles']]
            else:
                parts = [joint_transition]
            for jt in parts:
                if self.compact_obs:
                    jt = dict(jt, vehicles=compact_vehicles(jt['vehicles']))
                self.replay_buffer.push(jt)


    def set_baseline(self, b):
        """Q = B + residual: b = fixed B of every road for this group's shelter."""
        self.B = torch.as_tensor(np.asarray(b, dtype=np.float32), device=self.device)

    def with_b(self, q):
        """Network output (the residual) plus B when the baseline is on; unchanged otherwise."""
        return q if self.B is None else q + self.B

    def count_changed(self, q_feasible, feasible_indices):
        """Training: count decisions where the best road by B + residual differs from the best road by B alone."""
        if self.B is None:
            return
        c = (torch.argmax(q_feasible) != torch.argmax(self.B[feasible_indices])).to(torch.float32)
        self.changed_t = c if self.changed_t is None else self.changed_t + c
        self.changed_n += 1

    def pop_changed_pct(self):
        """% of this episode's decisions changed by the residual; resets the counter."""
        pct = 100.0 * float(self.changed_t) / self.changed_n if self.changed_n else 0.0
        self.changed_t, self.changed_n = None, 0
        return pct

    def set_episode_tau(self, episode):
        """B2: softmax temperature for this (0-based) training episode."""
        if self.tau_end is not None:
            frac = min(1.0, max(0, episode) / self.tau_decay_episodes)
            self.tau = self.tau_start + (float(self.tau_end) - self.tau_start) * frac
        return self.tau

    def maybe_update(self, plotter=None):
        """S5: count a second with new decisions; run update() on every update_every-th one."""
        self.update_ticks += 1
        if self.update_ticks % self.update_every == 0:
            return self.update(plotter=plotter)
        return None

    def update(self, plotter=None):
        """
        VDN Training with optimized batch calculation.
        Transfers all data to GPU in one go and computes loss using vectorized operations.
        """
        if len(self.replay_buffer) < self.batch_size or not self.training:
            return None
        
        # if len(self.replay_buffer) - self.current_size_buffer == 100: 
        #     # update network params every 100 samples added to buffer
        #     self.current_size_buffer = len(self.replay_buffer)

        # Sample batch
        if self.replay_type == 'per':
            batch, sampled_indices, is_weights = self.replay_buffer.sample(self.batch_size)
        else:
            batch, sampled_indices, is_weights = self.replay_buffer.sample(self.batch_size), None, None
        kept = []  # position in the sampled batch of each kept (non-empty) transition
        td_cats = []  # td_debug: 0 ordinary, 1 with an arrival / done member, 2 with a fine (rule C, hazard, no route)
        
        # --- 1. Data Preparation (CPU Side) ---  
        all_obs = []  
        all_next_obs = []  
        all_actions = []  
        all_dones = []  
        all_disasters = []  
        batch_rewards = []  
        group_sizes = []  # New: track num vehicles per valid batch
        all_member_rewards = []  # evac_time: reward of each member
        all_dts = []             # evac_time: seconds until the member's next decision or its end
        all_next_disasters = []  # evac_time: disaster features of each member's next state
        all_next_feas = []       # B0: feasible next-road indices of each member's next state
        all_steps = []           # B1: decisions in each member's return (1 = one-step)

        valid_batch_count = 0  

        for b_pos, joint_transition in enumerate(batch):
            vehicles_data = joint_transition['vehicles']
            global_reward = joint_transition['global_reward']

            if len(vehicles_data) == 0:
                continue
            kept.append(b_pos)
            if joint_transition.get('per_vehicle'):
                for v in vehicles_data:
                    all_member_rewards.append(float(v[6]))
                    all_dts.append(float(v[7]))
                    all_next_feas.append(v[9] if len(v) > 9 else None)
                    all_steps.append(int(v[10]) if len(v) > 10 else 1)
                all_next_disasters.append(np.array([np.squeeze(v[8]) for v in vehicles_data]))

            if self.td_debug:
                big = 0.5 * float(self.CONFIG['simulation_steps'])
                fine = any(len(v) > 6 and float(v[6]) <= -big for v in vehicles_data)
                td_cats.append(2 if fine else (1 if any(float(v[4]) == 1.0 for v in vehicles_data) else 0))

            batch_rewards.append(global_reward)  
            num_vehicles = len(vehicles_data)  
            group_sizes.append(num_vehicles)  

            # Collect per-group, squeezing out singleton dims  
            group_obs = []  
            group_next_obs = []  
            group_actions = []  
            group_dones = []  
            group_disasters = []  

            for v in vehicles_data:  
                # Squeeze to remove extra dims (e.g., (1, O) -> (O,))  
                group_obs.append(v[1] if self.compact_obs else np.squeeze(v[1]))  
                group_next_obs.append(v[3] if self.compact_obs else np.squeeze(v[3]))  
                group_actions.append(v[2])  # Assuming action is scalar/int, no squeeze needed  
                group_dones.append(v[4])    # Assuming done is scalar/float  
                # For disaster: squeeze if it's an array, or handle None  
                disaster = v[5]  
                group_disasters.append(np.squeeze(disaster) if disaster is not None else None)  

            all_obs.append(group_obs if self.compact_obs else np.array(group_obs))  # [num_vehicles, obs_dim]  
            all_next_obs.append(group_next_obs if self.compact_obs else np.array(group_next_obs))  
            all_actions.extend(group_actions)  # Keep flat for gathering  
            all_dones.extend(group_dones)  
            # For disasters: if None, keep as None; else stack squeezed arrays  
            if group_disasters[0] is not None:  
                all_disasters.append(np.array(group_disasters))  # [num_vehicles, dis_dim]  
            else:  
                all_disasters.append(None)  

            valid_batch_count += 1  

        if not all_obs:  
            return 0.0  

        # Compute max_N across the batch  
        max_N = max(group_sizes)  

        # --- 2. Pad and Move to GPU ---  
        # Pad obs, next_obs to [B, max_N, obs_dim]  
        if not self.compact_obs:
            obs_padded = np.zeros((valid_batch_count, max_N, all_obs[0].shape[-1]), dtype=np.float32)  
            next_obs_padded = np.zeros((valid_batch_count, max_N, all_next_obs[0].shape[-1]), dtype=np.float32)  

        # For disaster: handle None case  
        if all(d is None for d in all_disasters):  
            disaster_padded = None  
        else:  
            dis_dim = all_disasters[0].shape[-1]  # Use first non-None for dim  
            disaster_padded = np.zeros((valid_batch_count, max_N, dis_dim), dtype=np.float32)  

        padding_mask = torch.ones((valid_batch_count, max_N), dtype=torch.bool)  # True = padded  

        for i in range(valid_batch_count):  
            num_v = group_sizes[i]  
            if not self.compact_obs:
                obs_padded[i, :num_v] = all_obs[i]  # Now 2D -> 2D assignment  
                next_obs_padded[i, :num_v] = all_next_obs[i]  
            if disaster_padded is not None and all_disasters[i] is not None:  
                disaster_padded[i, :num_v] = all_disasters[i]  
            padding_mask[i, :num_v] = False  # False = valid  

        # To tensors (unchanged)  
        if self.compact_obs:
            obs_padded = self.obs_from_compact(all_obs, valid_batch_count, max_N)
            next_obs_padded = self.obs_from_compact(all_next_obs, valid_batch_count, max_N)
        else:
            obs_padded = torch.tensor(obs_padded, dtype=torch.float32, device=self.device)  
            next_obs_padded = torch.tensor(next_obs_padded, dtype=torch.float32, device=self.device)  
        if disaster_padded is not None:  
            disaster_padded = torch.tensor(disaster_padded, dtype=torch.float32, device=self.device)  
        padding_mask = padding_mask.to(self.device)

        # evac_time: each member's own next-state disaster features (legacy reuses the current ones)
        per_vehicle = len(all_member_rewards) > 0 and len(all_member_rewards) == sum(group_sizes)
        next_disaster_padded = disaster_padded
        if per_vehicle and disaster_padded is not None:
            nd = np.zeros(tuple(disaster_padded.shape), dtype=np.float32)
            for i in range(valid_batch_count):
                nd[i, :group_sizes[i]] = all_next_disasters[i]
            next_disaster_padded = torch.tensor(nd, dtype=torch.float32, device=self.device)

        # ... (rest of the update function remains unchanged)

        # Actions, dones: flat [N_total]  
        actions_tensor = torch.tensor(all_actions, dtype=torch.int64, device=self.device)  
        dones_tensor = torch.tensor(all_dones, dtype=torch.float32, device=self.device)  

        # Rewards: [B]  
        rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32, device=self.device) * self.reward_scale  

        # batch_segment_ids: flat indices for index_add_ (0 to B-1 repeated for each vehicle's group)  
        batch_segment_ids = []  
        for i, size in enumerate(group_sizes):  
            batch_segment_ids.extend([i] * size)  
        batch_ids_tensor = torch.tensor(batch_segment_ids, dtype=torch.int64, device=self.device)  

        # --- 3. Forward Pass (Batch) ---  
        # Now outputs [B*max_N, A], but padded positions are ignored via gathering on real indices  
        q_values_all = self.with_b(self.q_network(obs_padded, disaster_padded, padding_mask=padding_mask))  
        
        # Select Q for actions: only real vehicles. The forward output is [B*max_N, A], transition by
        # transition with each transition's padded slots in between, so select rows with the mask.
        N_total = sum(group_sizes)  
        valid_rows = ~padding_mask.reshape(-1)
        q_values_all = q_values_all[valid_rows]  # [N_total, A], same order as actions_tensor
        assert q_values_all.shape[0] == N_total
        q_chosen = q_values_all.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)  

        # --- 4. VDN Aggregation (Sum Q per batch sample) ---
        
        # Initialize container for Total Q per batch sample: [B]
        q_total_current = torch.zeros(valid_batch_count, device=self.device, dtype=torch.float32)
        
        # Sum individual agent Qs into their corresponding batch index
        # [B].index_add_(dim=0, index=[N], source=[N]) -> Result: [B]
        q_total_current.index_add_(0, batch_ids_tensor, q_chosen)

        # --- 5. Target Calculation ---  
        with torch.no_grad():  
            next_q_main = self.with_b(self.q_network(next_obs_padded, next_disaster_padded, padding_mask=padding_mask))  
            next_q_main = next_q_main[valid_rows]  # real vehicles only
            if self.target_mask and per_vehicle:
                # B0: argmax only over the roads the vehicle could really take next (done members: no Q term anyway)
                rows, cols = [], []
                for i, f in enumerate(all_next_feas):
                    if f is not None and len(f):
                        rows.append(np.full(len(f), i, dtype=np.int64)); cols.append(np.asarray(f, dtype=np.int64))
                feas_mask = torch.zeros_like(next_q_main, dtype=torch.bool)
                if rows:
                    feas_mask[torch.as_tensor(np.concatenate(rows), device=self.device),
                              torch.as_tensor(np.concatenate(cols), device=self.device)] = True
                # a member still driving (done = 0) must have its list; done members keep all roads (no Q term)
                missing = [i for i, f in enumerate(all_next_feas)
                           if (f is None or not len(f)) and float(all_dones[i]) == 0.0]
                if missing:
                    raise ValueError(f"target_mask: {len(missing)} members with done = 0 have no feasible next roads")
                feas_mask |= ~feas_mask.any(dim=1, keepdim=True)
                next_q_main = next_q_main.masked_fill(~feas_mask, float('-inf'))
            next_actions = torch.argmax(next_q_main, dim=1)  

            next_q_target = self.with_b(self.target_network(next_obs_padded, next_disaster_padded, padding_mask=padding_mask))  
            next_q_target = next_q_target[valid_rows]  # real vehicles only
            next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)  

            next_q_values = next_q_values * (1 - dones_tensor)  

            if per_vehicle:
                # evac_time: y_i = r_i + gamma_s^dt_i (1 - d_i) Q_tgt(s'_i, a*_i); group target = sum_i y_i
                r_i = torch.tensor(all_member_rewards, dtype=torch.float32, device=self.device) * self.reward_scale
                if self.gamma_per_second is not None:
                    disc = self.gamma_per_second ** torch.tensor(all_dts, dtype=torch.float32, device=self.device)
                elif self.n_step > 1:
                    # B1: y_i = sum_k gamma^k r_ik + gamma^steps_i (1 - d_i) Q_tgt(s_i,steps, a*)
                    disc = self.gamma ** torch.tensor(all_steps, dtype=torch.float32, device=self.device)
                else:
                    disc = torch.full_like(r_i, self.gamma)
                y_i = r_i + disc * next_q_values
                target = torch.zeros(valid_batch_count, device=self.device, dtype=torch.float32)
                target.index_add_(0, batch_ids_tensor, y_i)
            else:
                q_total_next = torch.zeros(valid_batch_count, device=self.device, dtype=torch.float32)
                q_total_next.index_add_(0, batch_ids_tensor, next_q_values)

                target = rewards_tensor + self.gamma * q_total_next

        # --- 6. Loss & Optimization ---
        
        # Huber Loss per batch sample (delta=1.0)
        # ([B] - [B]) -> [B]
        # use huber loss, replace MSE loss
        loss = self._rho(q_total_current, target)
        delta_i = None
        if self.vdn_stats_on:
            if not per_vehicle:
                raise ValueError("vdn_loss needs per-vehicle transitions (reward_mode legacy_end or evac_time)")
            delta_i = (y_i - q_chosen).detach()  # [N_total], car by car in the order of batch_ids_tensor
            d_sum = (target - q_total_current).detach()
            # hidden error (rule of analysis/td_percar_plot.py): in groups of 2+ cars, |delta_i| > |delta_sum|
            sizes = torch.as_tensor(group_sizes, device=self.device)[batch_ids_tensor]
            multi = (sizes > 1).float()
            self.stat_multi = self.stat_multi + multi.sum()
            self.stat_hidden = self.stat_hidden + (multi * (delta_i.abs() > d_sum.abs()[batch_ids_tensor]).float()).sum()
            self.stat_loss_group = self.stat_loss_group + loss.detach().mean()
            if self.per_car_lambda > 0:
                car = torch.zeros(valid_batch_count, device=self.device, dtype=torch.float32)
                car.index_add_(0, batch_ids_tensor, self._rho(q_chosen, y_i))  # sum_i rho(delta_i) per transition
                self.stat_loss_car = self.stat_loss_car + car.detach().mean()
                loss = loss + self.per_car_lambda * car
        if self.td_debug:
            d = (q_total_current.detach() - target).float()
            c = torch.as_tensor(td_cats, dtype=torch.int64, device=self.device)
            parts = []
            for vals in (torch.ones_like(d), d, d.abs(), (d > 0).float()):
                parts.append(torch.zeros(3, device=self.device).index_add_(0, c, vals))
            acc = torch.stack(parts, dim=1)  # [3 types, (count, sum, sum |d|, count d > 0)]
            self.td_acc = acc if self.td_acc is None else self.td_acc + acc
        self.stat_loss_raw = self.stat_loss_raw + loss.detach().mean()  # unweighted, stays on the GPU
        self.stat_updates += 1
        self.stat_picks += len(kept)
        self.stat_done_picks += int(getattr(self.replay_buffer, 'last_n_done', 0))
        if is_weights is not None:
            # prioritized replay: importance-sampling weights of the kept transitions
            w = torch.tensor(np.asarray(is_weights)[kept], dtype=torch.float32, device=self.device)
            loss = loss * w
            self.stat_weight = self.stat_weight + w.mean()
        
        # Sum all batch losses to get single gradient scalar
        # [B] -> Scalar
        total_loss = loss.mean() if self.loss_reduction == 'mean' else loss.sum()

        self.optimizer.zero_grad()
        total_loss.backward()
        
        # track gradients for debugging (every grad_log_every-th update)
        if self.update_step_count % self.grad_log_every == 0:
            self.track_gradients(plotter)

        # Clip gradient norm to 1
        gn = torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=self.max_grad_norm)  # norm before clipping
        if self.vdn_stats_on:
            self.stat_gnorm = self.stat_gnorm + gn.detach()
            self.stat_clipped = self.stat_clipped + (gn.detach() > self.max_grad_norm).float()

        # Clip individual gradient values to [-0.5, 0.5]
        # torch.nn.utils.clip_grad_value_(self.q_network.parameters(), clip_value=0.5)
        self.optimizer.step()

        # --- NEW CODE START ---
        # Step the scheduler based on the loss of this batch
        # self.scheduler.step(total_loss.item())
        # --- NEW CODE END ---

        # Update target network
        self.update_step_count += 1
        # OPTION 1: Soft Update (Recommended for continuous control or modern DQN)
        # logic: "Move target slightly towards policy every single step"
        self.soft_update(self.q_network, self.target_network)

        # prioritized replay: new priority |delta_j| for each sampled transition
        if self.replay_type == 'per' and sampled_indices is not None:
            td = (q_total_current - target).detach().abs()
            if self.priority_mode == 'max':
                worst = torch.zeros_like(td).scatter_reduce_(0, batch_ids_tensor, delta_i.abs(), reduce='amax', include_self=True)
                td = torch.maximum(td, worst)
            self.replay_buffer.update_priorities([sampled_indices[k] for k in kept], td.cpu().numpy())

        # OPTION 2: Hard Update (Old school DQN)
        # logic: "Copy weights entirely every N steps"
        # if self.update_step_count % self.target_update_freq == 0:
        #     self.target_network.load_state_dict(self.q_network.state_dict())

        loss_value = total_loss.item()  # one sync, reused below
        if plotter:
            plotter.record_loss(self.group_id, loss_value)

        return loss_value

    def obs_from_compact(self, groups, B, max_N):
        """S3: dense padded observations [B, max_N, obs_dim] on the GPU from CompactObs lists (zeros in padded slots)."""
        rows, edges, dens, one_rows, one_cols = [], [], [], [], []
        for i, group in enumerate(groups):
            for j, c in enumerate(group):
                r = i * max_N + j
                rows.append(r); edges.append(c.edge); dens.append(c.density)
                one_rows.append(np.full(len(c.ones), r, dtype=np.int64)); one_cols.append(c.ones.astype(np.int64) + 1)
        obs = torch.zeros(B * max_N, self.obs_dim, dtype=torch.float32, device=self.device)
        rows_t = torch.as_tensor(np.asarray(rows, dtype=np.int64), device=self.device)
        base = self.obs_dim - self.mem_k - self.extra_dim
        obs[rows_t, 0] = torch.as_tensor(np.asarray(edges, dtype=np.float32), device=self.device)
        obs[rows_t, base - 1] = torch.as_tensor(np.asarray(dens, dtype=np.float32), device=self.device)
        if self.extra_dim:
            extra = np.stack([c.extra for group in groups for c in group]).astype(np.float32)
            obs[rows_t, base:base + self.extra_dim] = torch.as_tensor(extra, device=self.device)
        if self.mem_k:
            hist = np.stack([c.hist for group in groups for c in group]).astype(np.float32)
            obs[rows_t, base + self.extra_dim:] = torch.as_tensor(hist, device=self.device)
        if one_rows:
            obs[torch.as_tensor(np.concatenate(one_rows), device=self.device),
                torch.as_tensor(np.concatenate(one_cols), device=self.device)] = 1.0
        return obs.view(B, max_N, self.obs_dim)

    def track_gradients(self, plotter):
        """
        Logs gradient mean, std, and norm per layer to a text file for analysis.
        Format: step, layer_name, mean, std, norm
        The aim is to detect vanishing/exploding gradients during training.
        """
        grads = [(name, param.grad.data) for name, param in self.q_network.named_parameters() if param.grad is not None]
        if not grads:
            return
        # stats of all layers on the GPU, then one copy to the CPU (was 3 syncs per layer)
        stats = torch.stack([torch.stack([g.mean(), g.std(), g.norm()]) for _, g in grads]).cpu().tolist()
        for (name, _), (mean_val, std_val, norm_val) in zip(grads, stats):
            # Write to file (txt format)
            plotter.write_each_gradient(self.update_step_count, self.group_id, name, mean_val, std_val, norm_val)
    
    def soft_update(self, source_net, target_net):
        """
        Soft update target network parameters.
        theta_target = beta * theta_target + (1 - beta) * theta_source
        """
        with torch.no_grad():
            # 1. Update Parameters (Weights/Biases)
            # same arithmetic as beta * source + (1 - beta) * target, in a few fused kernels instead of a loop
            targets = list(target_net.parameters())
            scaled_source = torch._foreach_mul(list(source_net.parameters()), self.beta_soft)
            torch._foreach_mul_(targets, 1.0 - self.beta_soft)
            torch._foreach_add_(targets, scaled_source)
                
            # 2. Update Buffers (Batch Norm stats: running_mean, running_var)
            # Standard practice: Hard copy buffers (do not interpolate means/vars)
            # for target_buffer, source_buffer in zip(target_net.buffers(), source_net.buffers()):
            #     target_buffer.copy_(source_buffer)

    # # CPU calculation 
    # # def update(self, plotter=None):
    #     """
    #     VDN Training with proper joint transitions, one batch → one total loss.
    #     Each sample (timestep) in the batch is processed individually.
    #     """
    #     if len(self.replay_buffer) < self.batch_size or not self.training:
    #         return None

    #     # Sample joint transitions
    #     batch = self.replay_buffer.sample(self.batch_size)  # List of joint transitions
    #     total_loss = 0.0

    #     for joint_transition in batch:
    #         vehicles_data = joint_transition['vehicles']  # List of vehicles
    #         global_reward = joint_transition['global_reward']  # Scalar reward

    #         if len(vehicles_data) == 0:
    #             continue

    #         # Extract per-vehicle data
    #         obs_list = [v[1] for v in vehicles_data]        # Each obs: (1, obs_dim)
    #         next_obs_list = [v[3] for v in vehicles_data]   # Each next_obs: (1, obs_dim)
    #         actions = [v[2] for v in vehicles_data]         # Scalar per vehicle
    #         dones = [v[4] for v in vehicles_data]           # Scalar per vehicle
    #         disasters = [v[5] for v in vehicles_data]       # Each disaster: (1, dis_dim)
            
    #         # add cuda
    #         # Convert to tensors: squeeze first dimension (1, dim) -> (dim,)
    #         obs_tensor = torch.stack([torch.tensor(o, dtype=torch.float32).squeeze(0) for o in obs_list]).to(self.device)         # [N, obs_dim]
    #         next_obs_tensor = torch.stack([torch.tensor(o, dtype=torch.float32).squeeze(0) for o in next_obs_list]).to(self.device)  # [N, obs_dim]
    #         # disaster_tensor = torch.stack([torch.tensor(d, dtype=torch.float32).squeeze(0) for d in disasters if d is not None])     # [N, dis_dim]
    #         actions_tensor = torch.tensor(actions, dtype=torch.int64, device=self.device)                                              # [N]
    #         dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)                                                # [N]

    #         # if disaster is empty (meaning None)
    #         # we replace with None, then later it is replaced by -1 vector in Q_network model
    #         # add cuda
    #         if all(d is None for d in disasters):
    #             disaster_tensor = None
    #         else:
    #             disaster_tensor = torch.stack([torch.tensor(d, dtype=torch.float32).squeeze(0) for d in disasters]).to(self.device)     # [N, dis_dim]

    #         # --- Forward pass: current Q ---
    #         q_values_all = self.q_network(obs_tensor, disaster_tensor)  # [N, action_dim]
    #         q_chosen = q_values_all.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)            # [N]
    #         q_total_current = q_chosen.sum()                                                     # scalar

    #         # --- Target computation (Double DQN) ---
    #         with torch.no_grad():
    #             next_q_main = self.q_network(next_obs_tensor, disaster_tensor)  # [ N, action_dim]
    #             next_actions = torch.argmax(next_q_main, dim=1)                                         # [N]

    #             next_q_target = self.target_network(next_obs_tensor, disaster_tensor)  # [N, action_dim]
                
    #             next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)          # [N]

    #             # Mask done vehicles
    #             next_q_values = next_q_values * (1 - dones_tensor)                                      # [N]

    #             # Sum over vehicles → joint Q for this timestep
    #             q_total_next = next_q_values.sum()                                                     # scalar

    #             # Compute target
    #             target = global_reward + self.gamma * q_total_next                                      # scalar

    #         # --- Loss for this timestep ---
    #         loss = (q_total_current - target) ** 2

    #         # change to huber loss
    #         # loss = F.huber_loss(q_total_current.unsqueeze(0), target.unsqueeze(0), delta=1.0)
    #         total_loss += loss

    #     # Backpropagation
    #     self.optimizer.zero_grad()
    #     total_loss.backward()
    #     torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
    #     self.optimizer.step()

    #     # --- Update target network periodically ---
    #     self.update_step_count += 1
    #     if self.update_step_count % self.target_update_freq == 0:
    #         self.target_network.load_state_dict(self.q_network.state_dict())

    #     if plotter:
    #         plotter.record_loss(self.group_id, total_loss.item())

    #     logger.debug(f"Total Loss for batch: {total_loss.item()}")
    #     return total_loss.item()

    def save_model(self, path=None):  
        """Save the agent's model to disk"""  
        if path is None:  
            os.makedirs(self.saved_folder, exist_ok=True)  
            path = f"{self.saved_folder}/agent_{self.group_id}_{time.strftime('%Y%m%d_%H%M%S')}.pt"  
        
        try:
            # Save optimizer state before moving model to CPU
            optimizer_state = self.optimizer.state_dict()
            # Save model state dicts on CPU
            policy_state = self.q_network.state_dict()
            target_state = self.target_network.state_dict()
            # Save everything
            torch.save({
                'policy_net_state_dict': policy_state,
                'target_net_state_dict': target_state,
                'optimizer_state_dict': optimizer_state,
                'group_id': self.group_id,
                'total_steps': self.total_steps,
                **({'lr_scheduler_state': self.lr_scheduler.state_dict()} if self.lr_scheduler is not None else {}),
            }, path)
            logger.success(f"Model saved to {path}")
            return path
        except Exception as e:
            logger.error(f"Failed to save model at {path}: {e}")
   
        return path  
    
    # def load_model(self, path):  
    #     """Load a saved model from disk"""  
    #     if not os.path.exists(path):  
    #         logger.warning(f"Model file not found: {path}")  
    #         return False  
            
    #     checkpoint = torch.load(path)  
        
    #     # Load the saved state dictionaries  
    #     self.q_network.load_state_dict(checkpoint['policy_net_state_dict'])  
    #     self.target_network.load_state_dict(checkpoint['target_net_state_dict'])  
    #     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  
    #     # self.epsilon = checkpoint['epsilon']  
    #     self.total_steps = checkpoint['total_steps'] 
        
    #     logger.info(f"Model {checkpoint['group_id']} loaded from {path}")  
    #     return True  

    # def load_model(self, path, fine_tune_mode=False):
    #     """Load a saved model from disk, with optional fine-tuning mode."""
    #     if not os.path.exists(path):
    #         logger.warning(f"Model file not found: {path}")
    #         return False
            
    #     checkpoint = torch.load(path)
        
    #     # Load model weights
    #     self.q_network.load_state_dict(checkpoint['policy_net_state_dict'])
    #     self.target_network.load_state_dict(checkpoint['target_net_state_dict'])
        
    #     # Load optimizer state only if not fine-tuning
    #     if not fine_tune_mode:
    #         self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    #     self.total_steps = checkpoint.get('total_steps', 0)
    #     logger.info(f"Model {checkpoint.get('group_id', 'unknown')} loaded from {path}")

    #     # --- Fine-tuning setup ---
    #     if fine_tune_mode:
    #         self.setup_fine_tune()
          
    #     else:
    #     # otherwise, do inference mode
    #         self.set_eval_mode()

    #     return True

    def load_model(self, path, fine_tune_mode=False, resume_mode=False):
        """Load a saved model from disk, with optional fine-tuning mode.
        resume_mode: continue training as it was (optimizer, total_steps, lr schedule kept; nothing frozen; train mode)."""
        if not os.path.exists(path):
            logger.warning(f"Model file not found: {path}")
            return False
            
        checkpoint = torch.load(path, map_location=self.device)
        
        # 1. Load model weights 
        try:
            self.q_network.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_net_state_dict'])
        except RuntimeError as e:
            logger.error(f"Model architecture mismatch: {e}")
            return False
        
        # 2. Safely try to load optimizer in inference mode
        # in inference mode, we do not always need to load optimizer
        if not fine_tune_mode:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except ValueError as e:
                # This catches specific error and lets the code continue
                logger.warning(f"Warning: Could not load optimizer state (Size mismatch). Starting with fresh optimizer. \nDetails: {e}")
            except KeyError:
                logger.warning("Optimizer state not found in checkpoint.")
        
        self.total_steps = checkpoint.get('total_steps', 0)
        logger.info(f"Model {checkpoint.get('group_id', 'unknown')} loaded from {path}")

        if resume_mode:
            if self.lr_scheduler is not None and 'lr_scheduler_state' in checkpoint:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state'])
            self.set_train_mode()
            return True

        # --- Fine-tuning setup ---
        if fine_tune_mode:
            self.setup_fine_tune()
        else:
            # otherwise, do inference mode
            self.set_eval_mode()

        return True
    
    def setup_fine_tune(self):
        logger.info("Fine-tune mode activated: freezing encoder layers and setting training mode.")
        self.set_train_mode()

        # Freeze encoder layers in q_network
        for name, param in self.q_network.named_parameters():
            # name of freezing layers
            if any(k in name for k in ["input_proj", "transformer", "lstm_encoder"]):
                param.requires_grad = False
            else:
                param.requires_grad = True

        # Define new fine-tuning optimizer (fresh)
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.q_network.parameters()),
            lr=self.CONFIG['learning_rate'],  
            weight_decay=1e-4
        )

    
    def load_offline_model(self, path, agent_name):
        """Load a saved model from disk"""  
        if not os.path.exists(path):  
            logger.debug(f"Model file not found: {path}")  
            return False  

        # load the model
        checkpoint = torch.load(path)  
        # load the weights of model
        self.q_network.load_state_dict(checkpoint, map_location=torch.device('cpu'))
        # load the saved state of the model

        logger.info(f"Model {agent_name} loaded from {path}")
        return True
        
    def set_eval_mode(self):  
        """Set the agent to evaluation mode (no exploration)"""  
        self.training = False  # Critical flag to disable exploration  
        self.q_network.eval()  
        self.target_network.eval()

    def set_train_mode(self):  
        """Set the agent back to training mode"""  
        self.training = True  
        self.q_network.train()  
        self.target_network.train()
