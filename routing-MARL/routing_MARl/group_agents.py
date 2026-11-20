import traci
import torch
import os
import json
import glob
import numpy as np
import re
import time
from collections import deque

from agent import Agent
from logs.logger import logger  # import our custom logger
from plotter import MultiAgentPlotter
from read_config import get_config
from utils.reward import Reward
from groups_shared_infor import GroupsSharedInfo
from database.h5_database import OfflineRLDataset
from database.sumo_db import SumoDB

class MultiAgentManager:
    """
    This architectur:
    obs1 concat obs1^, ...
    => transformer
    => embedding
    => Q1, ...
    => Q total
    -> action selection from individual Q values

    obs1: current edge index, connectivity edges
    obs1^: density on current edge 

    
    """
    def __init__(self, graph, activation_prob=1):
        
        self.CONFIG = get_config()
        self.graph = graph
        self.edges = graph.edges
        self.trips = graph.trips  
        self.edges_dist_disaster = graph.edges_dist_disaster  # distance of all edges to mean of all disaster zones
        self.edges_distance = graph.edges_distance # distance of all edges to destination edges
        self.disaster_zones = graph.disaster_zones  # disaster zones in the network

        # Optional safety shield at inference (--hazard-mask): hazard edges are removed from the
        # feasible next edges whenever a safe alternative exists. Off by default.
        self.hazard_mask = bool(self.CONFIG.get('hazard_mask', False))
        self.hazard_edges = {e for edges in (self.disaster_zones or {}).values() for e in edges}
        self.hazard_mask_stats = {"decisions": 0, "masked_decisions": 0, "no_safe_alternative": 0}

        # Create reward instance, passing self as the agents manager  
        self.reward = Reward(self)  

        # Initialize the shared info manager to allow sharing information between agents
        self.shared_info = GroupsSharedInfo(graph) 

        # agents are grouped by their destination
        # agents share the same destination, so they are in the same group
        self.groups = {} 
        # configs for save and load model
        self.saved_folder = os.path.join(os.path.dirname(__file__), '..', 'new_models') 
        if self.CONFIG.get('models_subdir'):
            # --run-name: keep this run's models apart (own index counter, no overwriting)
            self.saved_folder = os.path.join(self.saved_folder, self.CONFIG['models_subdir'])

        # the dimension of state is index of [current edge],[feasible next edges], [flows of feasible next edges]
        # and [mean x disaster, mean y disaster, radius disaster]
        self.obs_dim = 1 + len(self.edges) + 1 # current edge + next feasible edges + density on current edge)
        self.action_dim = len(self.edges)  # action dimension based on the number of edges in the network
        
        ### create offlinedataset for RL if required
        # Create dataset
        try:        
            offline_data_path = self.CONFIG['offline_data_path']
            self.offline_dataset = OfflineRLDataset(filename=offline_data_path, 
                                           state_dim=self.obs_dim, action_dim=self.action_dim)
        except Exception:
            logger.warning("No offline data path for storing offline data collection")

        # init SUMO db to keep track of edge connectivity to destination edges
        self.SumoDB_instance = SumoDB()

        # Convert edges dict into a list
        # the index of the edge in the list is the encoded-number of the edge
        self.edges_list = list(self.edges.keys())
        self.max_vehicles_edge = self.CONFIG['max_vehicles_edge'] # maximum flows on in each edge
        self.max_travel_time = graph.max_travel_time  # maximum travel time for each edge
        self.destination_edges = self.CONFIG['destinations_edges']  # destination edges for the vehicles

        self.reward_type = self.CONFIG['reward_type'] # R1, R1R2
        # self.d_max = graph.max_edge_to_dest_distance  # maximum distance from an edge to edge distance for reward calculation
        self.w1 = self.CONFIG['reward']['R1R2']['w1']  # weight for travel time reward
        self.w2 = self.CONFIG['reward']['R1R2']['w2']  # weight distance reward

        self.d_weight = self.CONFIG['reward']['MSE']['d_weight']  # weight for distance reward
        self.max_speed_vehs = self.CONFIG['max_speed']  # maximum speed of vehicles on an edge

        self.dead_edge =  self.CONFIG['reward']['MSE']['dead_edge'] # penalty for dead edges
        self.disaster_info = graph.disaster_info # dict to hold the radius of each disaster zone
        self.dist_OD = graph.dist_OD # euclid distance of origin => dest
        
        # mode to activate/deactive disaster
        self.is_disaster_active = False  
        self.activation_prob = activation_prob # probability of activating disaster
        # plotter for visualizing agent performance.
        self.plotter = None  
        # set mode for training/ inference
        self.training = True
        self.offline_infer = False

        # Track timestep transitions for each group
        self.group_timestep_buffers = {}  # {destination: list of (vehicle_id, obs, action, next_obs, done)}
         # Track rewards for each group in CURRENT timestep
        self.current_timestep_rewards = {}  # {destination: [reward1, reward2, ...]}

        # ---- reward_mode 'evac_time' (rules A-C + team term); 'legacy' keeps get_reward_5 per decision ----
        self.reward_mode = str(self.CONFIG.get('reward_mode', 'legacy')).lower()
        sim_steps = float(self.CONFIG['simulation_steps'])
        self.reward_norm = float(self.CONFIG.get('reward_norm_s', 100))
        hp, nap = self.CONFIG.get('hazard_penalty_s'), self.CONFIG.get('no_arrival_penalty_s')
        self.hazard_penalty_s = sim_steps if hp is None else float(hp)
        self.no_arrival_penalty_s = sim_steps if nap is None else float(nap)
        team_cfg = self.CONFIG.get('team') or {}
        self.team_score = str(team_cfg.get('score', 'quartile')).lower()
        self.team_lambda = float(team_cfg.get('lambda', 0.0))
        self.grouping = str(self.CONFIG.get('grouping', 'same_second')).lower()
        if (self.CONFIG.get('shaping') or {}).get('enabled'):
            raise NotImplementedError("shaping.enabled = true is not implemented (decision D9: off)")
        # modes that hold each decision open until the vehicle's next decision or its end
        self.pending_mode = self.reward_mode in ('evac_time', 'legacy_end')
        # F1: enter the own shelter whenever it is a feasible next road (training and inference)
        self.shelter_rule = bool(self.CONFIG.get('shelter_rule', False))
        # B1: n-step returns per vehicle (n_step, default 1); B0: keep the next state's feasible roads (target_mask)
        self.n_step = int(self.CONFIG.get('n_step', 1) or 1)
        if self.n_step > 1 and self.CONFIG.get('gamma_per_second') is not None:
            raise NotImplementedError("n_step > 1 assumes one discount gamma per decision (gamma_per_second unset)")
        self.gamma = float(self.CONFIG['gamma'])
        self.target_mask = str(self.CONFIG.get('target_mask') or '').lower() == 'feasible'
        # time limit: 'fine' (rule C fine for cars still driving) | 'cutoff' (no fine, their open decision is dropped)
        self.time_limit_mode = str(self.CONFIG.get('time_limit_mode') or 'fine').lower()
        if self.time_limit_mode not in ('fine', 'cutoff'):
            raise ValueError(f"time_limit_mode must be 'fine' or 'cutoff', not {self.time_limit_mode!r}")
        # rwtH reward: risk-weighted remaining time to each shelter (utils/risk_time.py)
        self.rwt_table = None
        if self.CONFIG.get('reward_function') == 'get_reward_rwt':
            from utils.risk_time import RiskTimeTable
            self.rwt_table = RiskTimeTable(
                self, lam=float(self.CONFIG.get('rwt_lambda', 1.0)), r_safe=float(self.CONFIG.get('rwt_r_safe', 500.0)),
                fine_s=float(self.CONFIG['simulation_steps']) if self.CONFIG.get('rwt_fine_in_time', True) else 0.0,
                refresh_s=float(self.CONFIG.get('rwt_refresh_s', 10)))
        # memory branch + repeat penalty: each car's last K roads (window, oldest first)
        mem = self.CONFIG.get('memory') or {}
        self.mem_k = int(mem.get('window', 8)) if mem.get('enabled') else 0
        self.repeat_kappa = float(self.CONFIG.get('repeat_kappa', 0.0) or 0.0)
        self.window_k = self.mem_k or (int(self.CONFIG.get('repeat_window', 8)) if self.repeat_kappa > 0 else 0)
        if self.mem_k and self.repeat_kappa > 0 and int(self.CONFIG.get('repeat_window', self.mem_k)) != self.mem_k:
            raise ValueError("repeat_window must equal memory.window: the penalty only counts roads the memory sees")
        if self.mem_k and not self.pending_mode:
            raise NotImplementedError("memory.enabled needs reward_mode legacy_end or evac_time (decision-time observations)")
        self.edge_idx = {e: i for i, e in enumerate(self.edges_list)}
        self.window = {}
        # route baseline + delay penalty: Q = B + residual, B a fixed table per
        # shelter; the observation gets one more number, the car's delay D(e)/simulation_steps
        dcfg = self.CONFIG.get('delay') or {}
        self.delay_on = bool(dcfg.get('enabled', False))
        self.delay_cap = float(dcfg.get('cap_s', sim_steps))
        self.extra_dim = 1 if self.delay_on else 0
        if self.delay_on and (self.mem_k or not self.pending_mode):
            raise NotImplementedError("delay.enabled needs memory off and reward_mode legacy_end or evac_time")
        bcfg = self.CONFIG.get('baseline') or {}
        self.baseline = None
        if bcfg.get('enabled'):
            if not self.target_mask:
                raise ValueError("baseline.enabled needs target_mask: feasible (the target must not see B of roads the car cannot take)")
            from utils.route_baseline import RouteBaseline
            self.baseline = RouteBaseline(self, bcfg, float(self.CONFIG.get('reward_scale', 1.0)), self.gamma)
            print(f"route baseline: roads without a route per shelter {self.baseline.no_route}", flush=True)
        self.episodes_done = 0
        rs = self.CONFIG.get('resume') or {}
        self.beta_hold_from = (int(rs['from_episode']) if rs.get('from_episode') and str(rs.get('beta', 'hold')).lower() == 'hold'
                               else None)
        self.begin_episode()

    def set_training(self, mode: bool) -> None:
        """Turn on/off training mode."""
        self.training = mode

    def set_offline_infer(self, mode: bool) -> None:
        self.offline_infer = True

    def random_active_disaster(self) -> None:
        """
        Randomly activate/deactivate a disaster episode.

        If the config did not contain disaster information, the flag
        will stay False and a debug line is emitted.
        """
        if not self.disaster_info:                         # no disaster in cfg
            logger.warning("random_active_disaster(): "
                         "no disaster data configured - cannot activate.")
            self.is_disaster_active = False
            return

        self.is_disaster_active = np.random.random() < self.activation_prob
        if self.is_disaster_active:
            logger.warning("Disaster ACTIVATED for this episode.")
        else:
            logger.warning("No disaster for this episode.")

    def get_nearest_disaster(self, next_edge: str):
        """
        Return (zone_id, distance) of the nearest disaster zone to `edge_id`
        or (None, None) when nothing is active / configured.
        """
        # 1. Config check -------------------------------------------------
        if not self.disaster_info:
            logger.info("get_nearest_disaster(): no disaster configured.")
            return None, None

        # 2. Episode state check -----------------------------------------
        if not self.is_disaster_active:
            logger.info("get_nearest_disaster(): disaster not active.")
            return None, None

        # 3. Distance map check ------------------------------------------
        if not self.edges_dist_disaster:
            logger.warning("get_nearest_disaster(): "
                           "distance lookup table empty.")
            return None, None

        if next_edge not in self.edges_dist_disaster:
            logger.warning("get_nearest_disaster(): edge_id '%s' unknown.", next_edge)
            return None, None

        # 4. Retrieve nearest zone ---------------------------------------
        disaster_distances = self.edges_dist_disaster[next_edge]  
        # Find minimum distance and nearest zone
        min_distance = min(disaster_distances.values())  
        nearest_zone = [k for k, v in disaster_distances.items() if v == min_distance][0] 
        # return nearest disaster zone id and the distance to it 
        return nearest_zone, min_distance  
    
    def divide_trips_to_agent_groups(self):
        """Initialize groups"""
        for trip_id, trip_info in self.trips.items():
            destination = trip_info['to']
            
            if destination not in self.groups:
                self.groups[destination] = Agent(
                    self.obs_dim + getattr(self, 'mem_k', 0) + getattr(self, 'extra_dim', 0),   # + K road indices (memory), + delay
                    self.action_dim, 
                    group_id=destination
                )
                if getattr(self, 'baseline', None) is not None:
                    self.groups[destination].set_baseline(self.baseline.vector(destination))
                self.current_timestep_rewards[destination] = []
                self.group_timestep_buffers[destination] = []
            
            # self.groups[destination].add_vehicle(trip_id)

    # def step(self, vehicle_id, curr_edge, connected_edges, disaster_move=None):
    #     """
    #     Update each agent based on their actions and interactions in the environment.
    #     """
    #     dest_edge = self.trips[vehicle_id]['to']  # Get the destination edge of the vehicle
    #     agent = self.groups[dest_edge]['agent']  # Get the agent for the vehicle's group

    #     # 1-hot encoding for feasible next edges
    #     # feasible_actions = [1 if edge in connected_edges else 0 for edge in self.edges_list]
    #     feasible_actions = [1 if (edge in connected_edges and (not self.is_dead_edge(edge) or edge in self.destination_edges))  
    #                 else 0 for edge in self.edges_list]
        
    #     # logger.error(f"feasible actions: {feasible_actions}")
    #     # np_edge = np.array(self.edges_list)
    #     # np_actions = np.array(feasible_actions)
    #     # logger.error(f"feasibles actions to feasible edges {np_edge[np_actions ==1]}")
    #     # logger.error(f"current vehicle is {vehicle_id}")
        
    #     # TODO: if there no feasible actions, it means current edge wil lead to dead road
    #     if not any(feasible_actions):
    #         logger.error(f"all feasible actions from {curr_edge} is empty")
    #         return

    #     try:
    #         # Create observation for the agent (e.g., current node, travel time)
    #         encoded_observation = self.encode_oberservation(curr_edge, connected_edges) 
    #     except Exception as e:
    #         logger.error(f"Error creating observation for vehicle {vehicle_id}: {e}")
    
    #     try:
    #         # Select action using the GREEDY policy
    #         # action = agent.select_action(encoded_observation, feasible_actions) 

    #         # Select action using the UCB policy
    #         # action = agent.UCB_select_action(encoded_observation, feasible_actions)

    #         # if choose action according to offline model inference 
    #         if self.offline_infer: 
    #             action = agent.select_action_offline(encoded_observation, feasible_actions)

    #         # otherwise  using softmax action selection
    #         action = agent.softmax_select_action(encoded_observation, feasible_actions)
    #     except Exception as e:
    #         logger.error(f"Error selecting action for vehicle {vehicle_id}: {e}")
            
    #     # Execute joint action and obtain the next observation and reward
    #     if self.training:
    #         next_edge, reward = self.execute_action(action, curr_edge, dest_edge, vehicle_id, feasible_actions, disaster_move) 
    #         self.plotter.record_reward(agent.group_id, reward)  # Record the reward for the group agent
    #     else:
    #         # in inference mode, we do not calculate the reward
    #         next_edge = self.edges_list[action]

    #     # if current mode is training, we store the exp and update policy
    #     if self.training:
    #         # Store experience for training
    #         encoded_next_observation = self.encode_next_oberservation(next_edge)
    #         agent.replay_buffer.push((encoded_observation, action, encoded_next_observation, reward))

    #         # Update the agent's policy
    #         agent.update(self.plotter)

    #     # next state is the next edge chosend, so next state = next action = the next edge chosen
    #     return next_edge


    def step(self, vehicle_id, curr_edge, connected_edges, disaster_move=None):
        """Step for individual vehicle"""
        # inference only, config shelter_zone_s (test of an extended shelter rule): within that many seconds of the own
        # shelter (free-flow time of SUMO's shortest route from the current road), follow that route instead of the network
        zone = self.CONFIG.get('shelter_zone_s')
        if zone and not self.training:
            dest_edge = self.trips[vehicle_id]['to']
            route = traci.simulation.findRoute(curr_edge, dest_edge, routingMode=traci.constants.ROUTING_MODE_AGGREGATED)
            # free-flow time of that route (length / speed limit): a jam inside the loop must not hide the zone
            if not hasattr(self, '_ff_time'):
                self._ff_time = {}
            ff = 0.0
            for e in route.edges:
                if e not in self._ff_time:
                    self._ff_time[e] = traci.lane.getLength(e + "_0") / max(0.1, traci.lane.getMaxSpeed(e + "_0"))
                ff += self._ff_time[e]
            if len(route.edges) > 1 and ff <= float(zone):
                self.shelter_zone_used = getattr(self, 'shelter_zone_used', 0) + 1
                if self.shelter_zone_used % 500 == 1:
                    print(f"shelter zone used {self.shelter_zone_used} times", flush=True)
                return route.edges[1], list(route.edges[1:])
        req = self.prepare_step(vehicle_id, curr_edge, connected_edges)
        if req is None:
            return None, None
        try: 
            action = req['agent'].softmax_select_action(req['obs'], req['feasible'], req['dis'])
            self.count_select(1)

            # indices = np.where(np.array(feasible_actions) == 1)[0]
            # action = np.random.choice(indices)

        except Exception as e:
            self.count_select(0, 1, f"vehicle {vehicle_id}: {e}")
            logger.error(f"Error selecting action for vehicle {vehicle_id}: {e}")
            return None, None
        return self.finish_step(req, action, disaster_move)

    def count_select(self, ok, failed=0, err=None):
        """Decision and error counts for SELECT_SUMMARY (printed by simulation.py at each episode end, without --log);
        the first error is printed at once (a failing model must not look like a working run)."""
        self.n_decisions = getattr(self, 'n_decisions', 0) + ok
        self.n_select_errors = getattr(self, 'n_select_errors', 0) + failed
        if failed and self.n_select_errors == failed:
            print(f"SELECT_ERROR first: {err}", flush=True)

    def step_batch(self, requests, disaster_move=None):
        """S4 (batched_actions): requests = [(vid, prepare_step result or None)] of one second, in vehicle order.
        One network call per group, then finish_step per vehicle in the same order. Returns [(next_edge, append_route)]."""
        actions = {}
        by_group = {}
        for k, (vid, req) in enumerate(requests):
            if req is not None:
                by_group.setdefault(req['group'], []).append(k)
        for g, ks in by_group.items():
            reqs = [requests[k][1] for k in ks]
            try:
                acts = self.groups[g].batch_softmax_select_actions([r['obs'] for r in reqs], [r['feasible'] for r in reqs],
                                                                   [r['dis'] for r in reqs])
                self.count_select(len(ks))
            except Exception as e:
                self.count_select(0, len(ks), f"group {g}: {e}")
                logger.error(f"Error selecting actions for group {g}: {e}")
                acts = [None] * len(ks)
            actions.update(zip(ks, acts))
        out = []
        for k, (vid, req) in enumerate(requests):
            out.append((None, None) if req is None else self.finish_step(req, actions.get(k), disaster_move))
        return out

    def step_pool(self, reqs, second, disaster_move=None):
        """R2.1 oracle (inference_pool: oracle): reqs = prepare_step results of one
        second, in car order (None = the car cannot decide). The cars of one group are encoded together as one set (chunks of
        at most pool_max_set; unset = no cap), as in the training update; then each car samples its road from its own Q row
        IN CAR ORDER, so the random draws follow the order of ego-only. Also logs, per car, whether the best road with the
        pooled set differs from the best road alone (no random numbers). Returns one entry per request: (next_edge,
        append_route); (None, None) for a car without a choice; None after a selection error (counted)."""
        if self.CONFIG.get('shelter_zone_s'):
            raise RuntimeError("inference_pool oracle: shelter_zone_s is not supported")
        st = self.__dict__.setdefault('pool_stats', {'decisions': 0, 'errors': 0, 'sets': 0, 'size_sum': 0, 'max_set': 0,
                                                     'pooled': 0, 'flips': 0, 'tv': 0.0, 'sizes': []})
        cap = int(self.CONFIG.get('pool_max_set') or 0) or max(1, len(reqs))
        tau = float(self.CONFIG.get('inference_tau', 1) or 1)
        by_group = {}
        for k, r in enumerate(reqs):
            if r is not None:
                by_group.setdefault(r['group'], []).append(k)
        rows, failed = {}, set()
        for g, ks in by_group.items():
            agent = self.groups[g]
            for c in range(0, len(ks), cap):
                chunk = ks[c:c + cap]
                obs, dis = [reqs[k]['obs'] for k in chunk], [reqs[k]['dis'] for k in chunk]
                try:
                    qp = agent.q_pool(obs, dis)
                    qe = agent.q_ego(obs, dis) if len(chunk) > 1 else qp
                except Exception as e:
                    st['errors'] += len(chunk)
                    failed.update(chunk)
                    if st['errors'] == len(chunk):
                        print(f"POOL_ERROR first: group {g}, {len(chunk)} cars: {e}", flush=True)
                    continue
                st['sets'] += 1
                st['size_sum'] += len(chunk)
                st['max_set'] = max(st['max_set'], len(chunk))
                st['pooled'] += len(chunk) if len(chunk) > 1 else 0
                st['sizes'].append((second, g, len(chunk)))
                for j, k in enumerate(chunk):
                    rows[k] = qp[j]
                    f = torch.as_tensor(reqs[k]['feasible'], dtype=torch.bool, device=qp.device)
                    a, b = qp[j][f], qe[j][f]
                    st['flips'] += int(torch.argmax(a).item() != torch.argmax(b).item())
                    st['tv'] += 0.5 * float((torch.softmax(a / tau, 0) - torch.softmax(b / tau, 0)).abs().sum())
        out = []
        for k, r in enumerate(reqs):
            if r is None:
                out.append((None, None))
                continue
            if k in failed:
                self.n_select_errors = getattr(self, 'n_select_errors', 0) + 1
                out.append(None)
                continue
            try:
                action = r['agent'].sample_inference(rows[k], r['feasible'])
                st['decisions'] += 1
                self.n_decisions = getattr(self, 'n_decisions', 0) + 1
                out.append(self.finish_step(r, action, disaster_move))
            except Exception as e:   # per car (e.g. findRoute from a closed road): this car keeps its route
                st['route_errors'] = st.get('route_errors', 0) + 1
                if st['route_errors'] == 1:
                    print(f"POOL_ROUTE_ERROR first: vehicle {r['vid']}: {e}", flush=True)
                out.append(None)
        return out

    def pool_summary(self, out_dir):
        """R2.1: print POOL_SUMMARY and write pool_sizes.csv (second, group, set size) into out_dir."""
        st = getattr(self, 'pool_stats', None)
        if st is None:
            print("POOL_SUMMARY decisions=0 errors=0 sets=0", flush=True)
            return
        n = max(1, st['decisions'] + st['errors'])
        print(f"POOL_SUMMARY decisions={st['decisions']} errors={st['errors']} sets={st['sets']} "
              f"mean_set={st['size_sum'] / max(1, st['sets']):.2f} max_set={st['max_set']} "
              f"pooled_share={st['pooled'] / n:.4f} flips={st['flips']} flip_share={st['flips'] / n:.5f} "
              f"tv_mean={st['tv'] / n:.5f} route_errors={st.get('route_errors', 0)}", flush=True)
        with open(os.path.join(out_dir, "pool_sizes.csv"), "w") as f:
            f.write("second,group,set_size\n")
            for t, g, m in st['sizes']:
                f.write(f"{t},{g},{m}\n")

    def prepare_step(self, vehicle_id, curr_edge, connected_edges):
        """Observation, disaster features and feasible actions of one vehicle's decision; None if it cannot decide."""
        dest_edge = self.trips[vehicle_id]['to']
        group_agent = self.groups[dest_edge]

        ############## Obsevation
        try:
            # Create observation for the agent (e.g., current node, travel time)
            encoded_observation = self.encode_oberservation(curr_edge, connected_edges, vehicle_id=vehicle_id)
            encoded_disaster_obs = self.get_disaster_obs(curr_edge)
        except Exception as e:
            self.count_select(0, 1, f"observation of vehicle {vehicle_id}: {e}")
            logger.error(f"Error creating observation for vehicle {vehicle_id}: {e}")
            return None

        ##############  feasbile next actions 
        feasible_actions = [
            1 if (edge in connected_edges and 
                  (not self.is_dead_edge(edge) or edge in self.destination_edges))
            else 0 
            for edge in self.edges_list
        ]
        
        # mask out next edge options: remove edges that are not connected to dest edge
        feasible_actions = self.filter_edges_not_connected_dest(feasible_actions, dest_edge)

        if not any(feasible_actions):
            logger.warning(f"All feasible actions from {curr_edge} is empty")
            return None

        if getattr(self, 'shelter_rule', False):
            di = self.edges_list.index(dest_edge)
            if feasible_actions[di]:
                feasible_actions = [0] * len(feasible_actions)
                feasible_actions[di] = 1
                self.shelter_forced[dest_edge] = self.shelter_forced.get(dest_edge, 0) + 1

        if self.hazard_mask and not self.training:
            feasible_actions = self.mask_hazard_edges(feasible_actions)

        return {'vid': vehicle_id, 'curr_edge': curr_edge, 'group': dest_edge, 'agent': group_agent,
                'obs': encoded_observation, 'dis': encoded_disaster_obs, 'feasible': feasible_actions}

    def finish_step(self, req, action, disaster_move=None):
        """Act on the chosen action: reward, transition record, appended route (the second half of step)."""
        vehicle_id, curr_edge, dest_edge = req['vid'], req['curr_edge'], req['group']
        encoded_observation, encoded_disaster_obs, feasible_actions = req['obs'], req['dis'], req['feasible']
        if action is None:
            logger.warning(f"No action selected for vehicle {vehicle_id} on edge {curr_edge}")
            return None, None
        

        ##############  Excecute action => receive reward and next obsevation
        ############## Or do inference of action
        if self.training and self.pending_mode:
            # close this vehicle's previous decision (evac_time: its time cost is known now)
            t_now = traci.simulation.getTime()
            self.close_vehicle(vehicle_id, t_now, 'decided',
                               next_obs=encoded_observation, next_dis=encoded_disaster_obs,
                               next_feas=np.flatnonzero(np.asarray(feasible_actions)).astype(np.int32)
                               if self.target_mask else None)
            # get_reward_5 gives the appended route (routing unchanged); legacy_end also keeps its reward
            next_edge, base_reward, append_route = self.execute_action(
                action, curr_edge, dest_edge, vehicle_id, feasible_actions, disaster_move
            )
            if next_edge in self.hazard_edges:
                self.hazard_entries[vehicle_id] = self.hazard_entries.get(vehicle_id, 0) + 1
            seen = self.visits.setdefault(vehicle_id, {curr_edge})
            if next_edge in seen:
                self.repeat_cars.add(vehicle_id)
            seen.add(next_edge)
            rec = {'vid': vehicle_id, 'group': dest_edge, 'obs': encoded_observation, 'action': action,
                   'dis': encoded_disaster_obs, 't_start': t_now, 'base_reward': float(base_reward),
                   'hazard': next_edge in self.hazard_edges, 'slot': None}
            self.pending[vehicle_id] = rec
            self.group_timestep_buffers[dest_edge].append(rec)
            return next_edge, append_route
        if self.training:
            next_edge, individual_reward, append_route = self.execute_action(
                action, curr_edge, dest_edge, vehicle_id, feasible_actions, disaster_move
            )
            
            encoded_next_observation = self.encode_next_oberservation(next_edge)
            done = (next_edge in self.destination_edges)
            
            # Store in temporary buffer for this timestep
            self.group_timestep_buffers[dest_edge].append(
                (vehicle_id, encoded_observation, action, encoded_next_observation, done, encoded_disaster_obs)
            )

            # Accumulate reward for this timestep of current agent group
            self.current_timestep_rewards[dest_edge].append(individual_reward)
            return next_edge, append_route
        else:
            next_edge = self.edges_list[action]
            # fresh route with current travel times, as get_reward_5 does in training
            route = traci.simulation.findRoute(next_edge, dest_edge, routingMode=traci.constants.ROUTING_MODE_AGGREGATED)
            append_route = list(route.edges) if route.edges else self.inference_get_append_route(next_edge, dest_edge)
            return next_edge, append_route
        
    def end_timestep(self):
        """
        Call this at the end of each simulation timestep.
        Stores joint transitions and triggers training.
        """
        if not self.training:
            return
        
        start_time = time.time()
        if self.pending_mode:
            self._end_timestep_evac()
            return
        for destination, group_agent in self.groups.items():
            # Get all transitions for this group in current timestep

            try:
                transitions = self.group_timestep_buffers[destination]
            except Exception as e:
                logger.error(f"No transition for group {destination}: {e}. Skipping.")
                continue
            
            if len(transitions) > 0:
                # Calculate global reward for this group
                global_reward = self.calculate_group_reward(destination)
                logger.debug(f"Global reward for group {destination}: {global_reward}")
                
                # Store joint transition
                for vehicle_id, obs, action, next_obs, done, disaster_obs in transitions:
                    group_agent.store_transition_timestep(vehicle_id, obs, action, next_obs, done, disaster_obs)
                
                # Store joint transition with global reward
                group_agent.end_timestep(global_reward)
                
                # Update/training network with backpropagation
                loss = group_agent.update(plotter=self.plotter)
                
                if loss and self.plotter:
                    self.plotter.record_reward(destination, global_reward)
            
            # Clear buffer for next timestep
            self.group_timestep_buffers[destination] = []
        end_time = time.time()
        duration = end_time-start_time
        # print(f"Training takes: {duration}s")

    # ================================================================ evac_time reward (rules A-C + team)
    def begin_episode(self):
        """Reset the per-episode state of the evac_time reward."""
        self.pending = {}                                    # vid -> open decision record
        self.open_tr = {g: [] for g in self.groups}          # group -> group transitions not yet pushed
        self.final_slot = {}                                 # vid -> [(slot, discount)] that hold its final reward
        self.chain = {}                                      # vid -> slots still collecting their n-step return
        self.t_arr = {}                                      # vid -> arrival time at its own shelter
        self.outcome = {}                                    # vid -> 'arrived' | 'wrong_shelter' | 'failed'
        self.shelter_forced = {}                             # group -> times the shelter rule fired
        self.hazard_entries = {}                             # vid -> hazard edges entered
        self.censored = {}                                   # group -> open decisions dropped at the time limit (cutoff)
        self.visits = {}                                     # vid -> roads it has driven this episode
        self.window = {}                                     # vid -> deque of its last K road indices (memory / penalty)
        self.repeat_cars = set()                             # vids that chose a road they had already driven
        self.trip_plan = {}                                  # vid -> (departure time, planned trip time T1_0)
        if getattr(self, 'rwt_table', None) is not None:
            self.rwt_table.reset_episode()
        for g in self.group_timestep_buffers:
            self.group_timestep_buffers[g] = []

    def _end_timestep_evac(self):
        """Group this second's new decisions into group transitions, push finished ones, then train."""
        for destination, group_agent in self.groups.items():
            recs = self.group_timestep_buffers.get(destination, [])
            if recs:
                chunks = [recs] if self.grouping == 'same_second' else [[r] for r in recs]
                for chunk in chunks:
                    tr = _GroupTransition(group_agent.current_timestep, chunk)
                    group_agent.current_timestep += 1
                    self.open_tr.setdefault(destination, []).append(tr)
            # push transitions whose members are all closed; those with a final member wait for the episode end
            still_open = []
            for tr in self.open_tr.get(destination, []):
                if tr.complete() and not tr.has_final():
                    group_agent.push_group_transition(tr.to_dict())
                    if self.plotter:
                        self.plotter.record_reward(destination, tr.reward_sum())
                else:
                    still_open.append(tr)
            self.open_tr[destination] = still_open
            if recs:
                group_agent.maybe_update(plotter=self.plotter)
            self.group_timestep_buffers[destination] = []

    def close_vehicle(self, vehicle_id, t_now, outcome, next_obs=None, next_dis=None, next_feas=None):
        """Close the vehicle's open decision.

        outcome 'decided'       : it made a new decision (rule A + B, done = 0, next state = the state now)
                'arrived'       : it reached its own shelter (rule A + B, done = 1)
                'wrong_shelter' : it reached another group's shelter (rule C, done = 1)
                'failed'        : removed on a dead end, or still driving at the episode end (rule C, done = 1)
        """
        if not self.training or not self.pending_mode:
            return
        rec = self.pending.pop(vehicle_id, None)
        if rec is None:
            return
        dt = float(t_now) - rec['t_start']
        legacy = self.reward_mode == 'legacy_end'
        if legacy:
            r = rec['base_reward']                                       # B/C reward (get_reward_5), unscaled
        else:
            r = -dt / self.reward_norm                                   # rule A: time cost
            if rec['hazard']:
                r -= self.hazard_penalty_s / self.reward_norm            # rule B: hazard entry
        done = 0.0
        if outcome == 'arrived':
            done = 1.0
            self.t_arr[vehicle_id] = float(t_now)
        elif outcome in ('wrong_shelter', 'failed'):
            done = 1.0
            # rule C: not arrived (legacy_end: in get_reward_5 units, scaled by reward_scale in the update)
            r -= self.no_arrival_penalty_s if legacy else self.no_arrival_penalty_s / self.reward_norm
        if done:
            self.outcome[vehicle_id] = outcome
        slot = rec['slot']
        if slot is None:  # opened in this same second: give it its own transition now
            buf = self.group_timestep_buffers.get(rec['group'], [])
            if rec in buf:
                buf.remove(rec)
            tr = _GroupTransition(self.groups[rec['group']].current_timestep, [rec])
            self.groups[rec['group']].current_timestep += 1
            self.open_tr.setdefault(rec['group'], []).append(tr)
            slot = rec['slot']
        # B1: this decision's reward also enters the n-step returns of the vehicle's earlier, still open slots
        chain = self.chain.setdefault(vehicle_id, [])
        chain.append(slot)
        for sl in chain:
            sl.add(reward=r, dt=dt, done=done, gamma=self.gamma,
                   next_obs=next_obs if next_obs is not None else rec['obs'],
                   next_dis=next_dis if next_dis is not None else rec['dis'], next_feas=next_feas)
        if done:
            self.final_slot[vehicle_id] = [(sl, self.gamma ** (sl.steps - 1)) for sl in chain]
        for sl in chain:
            if done or sl.steps >= self.n_step:
                sl.closed = True
        self.chain[vehicle_id] = [sl for sl in chain if not sl.closed]
        if self.CONFIG.get('evac_debug') and self.plotter:
            self.plotter.write_evac_debug(f"{vehicle_id}, {float(t_now):.0f}, {dt:.0f}, {r:.6f}, {done:.0f}, {outcome}"
                                          + (f", {rec['base_reward']:.6f}" if legacy else ""))

    def end_episode(self, t_cap):
        """Episode end: rule C for every open decision, then the team term, then push all held transitions."""
        if not self.training or not self.pending_mode:
            return
        t_cap = float(t_cap)
        cutoff = self.time_limit_mode == 'cutoff'
        if cutoff:
            # still driving at the time limit: no fine, the open decision is dropped (its group transition never
            # completes, so it is not pushed); the logs still count the car as not arrived
            for vid, rec in list(self.pending.items()):
                self.outcome[vid] = 'failed'
                self.censored[rec['group']] = self.censored.get(rec['group'], 0) + 1
            self.pending = {}
        else:
            for vid in list(self.pending):
                self.close_vehicle(vid, t_cap, 'failed')
        for vid, chain in self.chain.items():  # B1: returns cut short without an end (bootstrap after their steps)
            for sl in chain:
                sl.closed = True
        self.chain = {}
        for g, group_agent in self.groups.items():
            vids = [v for v, info in self.trips.items() if info['to'] == g]
            T = np.array([self.t_arr.get(v, t_cap) - float(self.trips[v]['depart']) for v in vids])
            q1, q2, q3 = np.percentile(T, [25, 50, 75]) if len(T) else (0.0, 0.0, 0.0)
            if self.team_score == 'cvar25':
                worst = np.sort(T)[-max(1, len(T) // 4):]
                score = float(worst.mean()) if len(T) else 0.0
            else:  # 'quartile': Q2 + (Q3 - Q1)
                score = float(q2 + (q3 - q1))
            team_reward = -self.team_lambda * score / self.reward_norm
            share = team_reward / max(1, len(vids))
            for v in vids:
                if v in self.final_slot:
                    for sl, disc in self.final_slot[v]:
                        sl.reward += disc * share
                    if self.CONFIG.get('evac_debug') and self.plotter:
                        self.plotter.write_evac_debug(f"{v}, {t_cap:.0f}, 0, {share:.6f}, 1, team_share")
            n_pushed, reward_sum, dropped, lost = 0, 0.0, 0, 0
            for tr in self.open_tr.get(g, []):
                if tr.complete():
                    group_agent.push_group_transition(tr.to_dict())
                    reward_sum += tr.reward_sum()
                    n_pushed += 1
                    if self.plotter:
                        self.plotter.record_reward(g, tr.reward_sum())
                else:
                    dropped += 1
                    lost += sum(1 for sl in tr.slots if sl.closed)
            extra = {}
            if cutoff:
                extra.update(Censored=self.censored.get(g, 0), Dropped_Groups=dropped, Lost_Members=lost)
            if self.CONFIG.get('sumo_event_log'):
                extra['Repeat_Cars'] = sum(1 for v in vids if v in self.repeat_cars)
            if self.CONFIG.get('reward_function') in ('get_reward_5_inc', 'get_reward_rwt'):
                extra['Positive_Steps'] = self.reward.positive_steps.get(g, 0)
            if self.CONFIG.get('reward_function') == 'get_reward_5_clip_rep':
                extra.update(Repeat_Cars=sum(1 for v in vids if v in self.repeat_cars),
                             Repeat_Penalty_Sum=round(self.reward.repeat_penalty_sum.get(g, 0.0), 1))
            if self.CONFIG.get('reward_function') == 'get_reward_5_delay' or self.baseline is not None:
                delays = [self.t_arr[v] - self.trip_plan[v][0] - self.trip_plan[v][1] for v in vids
                          if self.outcome.get(v) == 'arrived' and v in self.t_arr and v in self.trip_plan]
                extra.update(Repeat_Cars=sum(1 for v in vids if v in self.repeat_cars),
                             Delay_Penalty_Sum=round(self.reward.delay_penalty_sum.get(g, 0.0), 1),
                             Median_Delay_Arrived=round(float(np.median(delays)), 1) if delays else -1,
                             Changed_Decisions_pct=round(group_agent.pop_changed_pct(), 2))
            if self.rwt_table is not None:
                extra.update(Repeat_Cars=sum(1 for v in vids if v in self.repeat_cars),
                             Rwt_Refresh_s=round(self.rwt_table.refresh_seconds, 2), Rwt_Refreshes=self.rwt_table.refreshes)
            outcomes = [self.outcome.get(v, 'never_decided') for v in vids]
            if self.plotter:
                self.plotter.write_episode_reward(
                    g, arrived=outcomes.count('arrived'), wrong_shelter=outcomes.count('wrong_shelter'),
                    failed=outcomes.count('failed'), q1=q1, q2=q2, q3=q3, team_reward=team_reward,
                    held_pushed=n_pushed, held_reward_sum=reward_sum,
                    shelter_forced=self.shelter_forced.get(g, 0),
                    hazard_vehicles=sum(1 for v in vids if self.hazard_entries.get(v, 0) > 0),
                    hazard_entries=sum(self.hazard_entries.get(v, 0) for v in vids),
                    clipped_steps=(self.reward.clipped_steps.get(g, 0)
                                   if self.CONFIG.get('reward_function') == 'get_reward_5_clip' else None),
                    extra=extra or None)
        self.reward.clipped_steps = {}
        self.reward.positive_steps = {}
        self.reward.repeat_penalty_sum = {}
        self.reward.delay_penalty_sum = {}
        self.begin_episode()

    def reset_timestep_rewards(self):
        """Reset reward buffers (call at start of new episode or timestep)"""
        for destination in self.current_timestep_rewards:
            self.current_timestep_rewards[destination] = []

    def calculate_group_reward(self, destination):
        """
        Calculate global reward as average of rewards from all vehicles 
        in this group that acted in the current timestep.
        """
        if destination not in self.current_timestep_rewards:
            return 0.0
        
        rewards = self.current_timestep_rewards[destination]
        
        if len(rewards) == 0:
            return 0.0
        
        # Average reward across all vehicles in group for this timestep
        avg_reward = np.mean(rewards)
        
        return avg_reward
    
    def collect_offline_data(self, vehicle_id, curr_edge, connected_edges, disaster_move):
        """
        Collect a single offline transition (S, A, R, S', group_id done)
        where action is the next edge chosen by an A* route.
        The transition is stored in the dataset.
        """
         
        dest_edge = self.trips[vehicle_id]['to']  # destination edge
        feasible_actions = [
            1 if (edge in connected_edges and 
                (not self.is_dead_edge(edge) or edge in self.destination_edges))
            else 0
            for edge in self.edges_list
        ]
        feasible_actions = self.filter_edges_not_connected_dest(feasible_actions, dest_edge)

        if not any(feasible_actions):
            logger.error(f"No feasible actions for vehicle {vehicle_id} on edge {curr_edge}")
            return None

        try:
            # Create observation for the agent (e.g., current node, travel time)
            encoded_observation = self.encode_oberservation(curr_edge, connected_edges) 
        except Exception as e:
            logger.error(f"Offline training. Error creating observation for vehicle {vehicle_id}: {e}")

        # --- Action selection: from A* path instead of RL agent ---
        # this is human imitation learning. Because we collect offline data from expert A* path
        # we first apply new edges weights to all edges (new edge weight = edge travel time + edge risk)
        # self.apply_new_edges_weight(disaster_move)
        route = traci.simulation.findRoute(curr_edge, dest_edge, routingMode=traci.constants.ROUTING_MODE_AGGREGATED)
        try:
            if len(route.edges) != 0:  
                next_edge = route.edges[1]  # take the first edge in the route as the next edge. the zero edge is current edge
            else:
                logger.error(f"A* could not find a route from {curr_edge} to {dest_edge} for vehicle {vehicle_id}")
                return None
        except Exception as e:
            logger.exception(f"The error route is {route.edges}. Error {e}")   
        
        encoded_next_observation = self.encode_next_oberservation(next_edge)
        action = self.edges_list.index(next_edge)  # map edge → discrete index
        
        # reward = self.compute_reward(curr_edge, next_edge, dest_edge, route, disaster_move) 
        # reward is current edge cost (already included travel time + risk)
        reward = -traci.edge.getTraveltime(next_edge) # reward should be reverse of cost

        # reward = self.reward._get_reward_5(curr_edge, next_edge, dest_edge, vehicle_id, feasible_actions)
        logger.info(f"Vehicle {action} took action: {next_edge}, received reward: {reward}")

        # --- Done flag ---
        done = next_edge in self.destination_edges

        # --- Store transition in HDF5 dataset ---
        # vehicles are grouped based on their similar destination => same destination => same group id
        group_id = dest_edge
        self.offline_dataset.append(encoded_observation, action, reward, encoded_next_observation, group_id, done)

        return next_edge
    
    def apply_new_edges_weight(self, disaster_move):
            """
            Reads all edge data, calculates new weights using NumPy vectorization,
            and writes them back to the simulation in a batch.
            """
            edge_ids = self.edges

            # ===================================================================
            # 1. READ PHASE: Gather all data from the simulation first
            # ===================================================================
            # Get all travel times and risk values. These loops are for I/O,
            # not calculation. This minimizes back-and-forth with SUMO.
            
            # Note: traci.edge.getTraveltime doesn't have a batch version, so
            # a list comprehension is the most efficient way to call it.
            travel_times_list = [traci.edge.getTraveltime(edge_id) for edge_id in edge_ids]

            # Process risks, handling cases where a risk might be None or 0.
            # The list comprehension directly translates original  'if not edge_risk' logic.
            risks_list = [disaster_move.get_risk_edge(edge_id) for edge_id in edge_ids]
            
            # Convert the Python lists into efficient NumPy arrays for calculation.
            travel_times = np.array(travel_times_list, dtype=np.float64)
            
            # This comprehension handles the `if not edge_risk: edge_risk = 0` logic.
            # It correctly processes values that are `None`, `0`, or `False`.
            risks = np.array([r if r else 0.0 for r in risks_list], dtype=np.float64)

            # ===================================================================
            # 2. CALCULATE PHASE: Use fast, vectorized NumPy operations
            # ===================================================================
            # This single line replaces the entire calculation loop. It operates on
            # all elements of the arrays at once, executed in highly optimized C code.
            new_weights = self.w1 * travel_times + self.w2 * risks

            # ===================================================================
            # 3. WRITE PHASE: Apply all changes back to the simulation
            # ===================================================================
            # Now, loop one last time just to send the results to SUMO.
            for i, edge_id in enumerate(edge_ids):
                # We use 'i' to get the corresponding weight from our calculated array.
                traci.edge.adaptTraveltime(edge_id, new_weights[i])
                # logger.debug(f"New cost for edge {edge_id}: {new_weights[i]}")
        
    def compute_reward(self, curr_edge, next_edge, dest_edge, route, disaster_move):

        # reward = self.reward.get_reward_5(curr_edge, next_edge, 
                                        #   dest_edge, vehicle_id, feasible_actions)

        reward = self.reward.get_reward_offline_training(curr_edge, next_edge, dest_edge, 
                                    route, disaster_move)
        return reward 


    def road_window(self, vehicle_id, curr_edge):
        """Add curr_edge to the car's window (once per road) and return the window as K indices, oldest first,
        padded at the front with len(edges_list)."""
        w = self.window.get(vehicle_id)
        if w is None:
            w = self.window[vehicle_id] = deque(maxlen=self.window_k)
        i = self.edge_idx.get(curr_edge, len(self.edges_list))
        if not w or w[-1] != i:
            w.append(i)
        return [len(self.edges_list)] * (self.window_k - len(w)) + list(w)

    def trip_plan_of(self, vehicle_id, curr_edge, dest_edge):
        """(t_dep, T1_0): departure time and planned trip time findRoute(road, shelter) at the car's first decision."""
        plan = self.trip_plan.get(vehicle_id)
        if plan is None:
            try:
                t_dep = float(traci.vehicle.getDeparture(vehicle_id))
            except traci.TraCIException:
                t_dep = float(self.trips[vehicle_id]['depart'])
            route = traci.simulation.findRoute(curr_edge, dest_edge, routingMode=traci.constants.ROUTING_MODE_AGGREGATED)
            t1_0 = float(route.travelTime) if route.edges else self.delay_cap
            plan = self.trip_plan[vehicle_id] = (t_dep, max(t1_0, 1.0))
        return plan

    def delay_now(self, vehicle_id, curr_edge):
        """D(e) = max(0, t_now - t_dep + T1(e) - T1_0) of the car on road e (cap when e has no route to its shelter)."""
        dest_edge = self.trips[vehicle_id]['to']
        t_dep, t1_0 = self.trip_plan_of(vehicle_id, curr_edge, dest_edge)
        route = traci.simulation.findRoute(curr_edge, dest_edge, routingMode=traci.constants.ROUTING_MODE_AGGREGATED)
        if not route.edges:
            return self.delay_cap
        return min(self.delay_cap, max(0.0, traci.simulation.getTime() - t_dep + float(route.travelTime) - t1_0))

    def encode_oberservation(self, curr_edge, connected_edges, vehicle_id=None):
        if getattr(self, 'window_k', 0) or getattr(self, 'delay_on', False):
            if vehicle_id is None:
                raise ValueError("memory / repeat penalty / delay: the observation needs the vehicle id")
            obs = self.encode(curr_edge, connected_edges)
            if getattr(self, 'delay_on', False):
                d = self.delay_now(vehicle_id, curr_edge) / float(self.CONFIG['simulation_steps'])
                obs = torch.cat([obs, torch.tensor([[d]], dtype=torch.float32)], dim=1)
            if getattr(self, 'window_k', 0):
                tail = self.road_window(vehicle_id, curr_edge)
                if self.mem_k:
                    obs = torch.cat([obs, torch.tensor(tail, dtype=torch.float32).unsqueeze(0)], dim=1)
            return obs
        """
        Transform the observation into a tensor format
        """
        return self.encode(curr_edge, connected_edges)

    def encode_next_oberservation(self, next_edge):
        """
        Transform then next observation into a tensor format
        """
        return self.encode(next_edge, connected_edges=self.edges[next_edge]["connected_edges"])

    def encode(self, curr_edge, connected_edges):
        """
        Encode the state of the current agent (obs):
        - [0] : Index of current edge
        - [1 : 1+num_edges] : one-hot connectivity edges from current edge
        
        then concat with [density of current edge] (obs^)

        Returns:
            obs (np.ndarray): state vector of fixed length
        """

        # Initialize state vector
        # we take obs dim -1 because we will concat with obs^ later
        obs = torch.zeros(self.obs_dim-1, dtype=torch.float32)

        # Encode current edge index (normalized)
        if curr_edge in self.edges_list:
            edge_idx = self.edges_list.index(curr_edge)
            obs[0] = edge_idx 
        else:
            logger.error(f"{curr_edge} not in edge list for encoding")
            obs[0] = -1.0  # use -1 if not on a known edge

        # Encode feasible edges (one-hot)
        for edge_id in connected_edges:
            if edge_id in self.edges_list:
                idx = self.edges_list.index(edge_id)
                obs[1 + idx] = 1.0

        # Encode obs^ - density on current edge
        num_vehicles = len(traci.edge.getLastStepVehicleIDs(curr_edge))  # Note: using curr_edge, not edge_id
        if num_vehicles > self.max_vehicles_edge:
            norm_flow = 1.0
        else:
            norm_flow = num_vehicles / self.max_vehicles_edge if self.max_vehicles_edge > 0 else 0.0

        # Create obs^ as a tensor
        obs_hat = torch.tensor([norm_flow], dtype=torch.float32)
        
        # Concatenate obs and obs^
        full_obs = torch.cat([obs, obs_hat], dim=0)
        
        return full_obs.unsqueeze(0)  # convert from 1D to 2D tensor (1, obs_dim)
    
    def get_disaster_obs(self, curr_edge):
        """
        Return a tensor MxD, where M is the number of disaster zones, D is the number of disaster features
        we can add binary mask/channel or full -1 tensors to indicate if there is no disaster zones
        """
        disaster_ids = list(self.disaster_info.keys())
        # if no disasters
        if len(disaster_ids) == 0:
            logger.warning("get_disaster_state(): no disaster zones configured.")
            return None

        # 2d vector
        distances = np.array([
                [self.edges_dist_disaster[curr_edge][disaster_id] ]
                for disaster_id in disaster_ids
        ]) # (4,))

        return torch.tensor(distances, dtype=torch.float32).unsqueeze(0) # convert from 2D to 3D tensor 
        ########### Add disaster info to the state: (mean x, mean y) and radius of disaster 
        # mean_x, mean_y = 0.0, 0.0
        # radius = 0.0 
        # nearest_zone, _ = self.get_nearest_disaster(curr_edge)  
        # if nearest_zone:
        #     mean_x, mean_y = self.disaster_info[nearest_zone]['coordinate']   
        #     radius = self.disaster_info[nearest_zone]['radius']  dius of disaster 
        # mean_x, mean_y = 0.0, 0.0
        # radius = 0.0 
        # nearest_zone, _ = self.get_nearest_disaster(curr_edge)  
        # if nearest_zone:
        #     mean_x, mean_y = self.disaster_info[nearest_zone]['coordinate']   
        #     radius = self.disaster_info[nearest_zone]['radius']  

        # ########## TODO: Normalize the state disaster
        # # Fixed indices for disaster information
        # disaster_start_idx = 1 + 2 * len(self.edges_list)  
        # self.obs[disaster_start_idx] = mean_x  
        # self.obs[disaster_start_idx + 1] = mean_y  
        # self.obs[disaster_start_idx + 2] = radius  
        # logger.error(f"disaster info state {mean_x} {mean_y} {radius}")

        #TODO: Add Disaster info to the state (disaster radius/distance to disaster)
        # nearest_zone, dist_disaster = self.get_nearest_disaster(curr_edge)
        # if nearest_zone:
        #     radius = self.disaster_info[nearest_zone]['radius']
        # disaster_state_index = 1 + 2 * len(self.edges_list)
        # disaster_state = 1 / (1+ np.sqrt( (radius/dist_disaster)**2 ))
        # self.obs[disaster_state_index] = disaster_state
        # logger.error(f"disaster info state {disaster_state}")

        # return self.obs.unsqueeze(0)  # convert from 1D to 2D tensor
        
    def execute_action(self, action, curr_edge, dest_edge, vehicle_id, feasible_actions,
                       disaster_move=None):
        """
        The agent executes the action in the environment and receives the next state and reward.
        The action is the index of next edge to traverse.
        """
        next_edge = self.edges_list[action]
        # reward = self.get_reward_MSE(next_edge, dest_edge)
        # reward = self.get_reward_based_d_weight(next_edge, dest_edge)
        # reward = self.get_reward_risk_model(next_edge, dest_edge)
        # reward = self.reward.get_reward_risk_model(next_edge, dest_edge)
        # reward = self.reward.get_reward_1(curr_edge, next_edge, dest_edge, vehicle_id, feasible_actions)
        # reward = self.reward.get_reward_3(curr_edge, next_edge, dest_edge, vehicle_id, feasible_actions)
        # reward = self.reward.get_reward_4(curr_edge, next_edge, dest_edge, vehicle_id, feasible_actions)
        # reward = self.reward.get_reward_5(curr_edge, next_edge, dest_edge, vehicle_id, feasible_actions)
        # if self.CONFIG['offline_training']: 
        #     reward = self.reward.get_reward_offline_training(curr_edge, next_edge, dest_edge, disaster_move)
        #     logger.success("Take reward using offline training")
        # else:
        # reward_function: 'get_reward_5' (default) | 'get_reward_5_clip'
        reward_fn = getattr(self.reward, self.CONFIG.get('reward_function', 'get_reward_5'))
        reward, append_route = reward_fn(curr_edge, next_edge, dest_edge, vehicle_id, feasible_actions)
        logger.debug(f"Vehicle {vehicle_id} took action: {next_edge}, received reward: {reward}")

        return next_edge, reward, append_route 

    def is_dead_edge(self, next_edge):
        """
        Check if the action chosen is a dead edge
        """
        # logger.debug(f"check {next_edge} is a dead edge")
        # logger.debug("connected edges of this edge: ", self.edges[next_edge]["connected_edges"])
        # there is no connectecs for the next edge, and next edge is not a destination edge
        if not self.edges[next_edge]["connected_edges"] and next_edge not in self.destination_edges:
            logger.warning(f"Check Dead edge: {next_edge}")
            return True
        return False      
    
    def mask_hazard_edges(self, feasible_actions):
        """Remove hazard edges from the feasible next edges, unless no safe edge would be left."""
        self.hazard_mask_stats["decisions"] += 1
        safe = [f if (f and self.edges_list[i] not in self.hazard_edges) else 0
                for i, f in enumerate(feasible_actions)]
        if not any(safe):
            self.hazard_mask_stats["no_safe_alternative"] += 1
            return feasible_actions
        if safe != list(feasible_actions):
            self.hazard_mask_stats["masked_decisions"] += 1
        return safe

    def inference_get_append_route(self, next_edge, dest_edge):
        return self.SumoDB_instance.get_append_route(next_edge, dest_edge)

    def filter_edges_not_connected_dest(self, feasible_actions, dest_edge):
        """
        Filter out next feasible actions that do not lead to the destination edge.
        feasible_actions: list[int]  # 1 means allowed, 0 means not allowed
        dest_edge: str               # target edge ID
        """
        # Find indices where the action is feasible (== 1)
        feasible_idxs = [i for i, v in enumerate(feasible_actions) if v == 1]

        a = np.where(np.array(feasible_actions) == 1)[0]
        logger.warning(f"Before filtering, feasible actions indices: {[self.edges_list[idx] for idx in a]}")

        # new result
        new_feasible_actions = list(feasible_actions)

        for idx in feasible_idxs:
            next_edge = self.edges_list[idx]

            check = self.SumoDB_instance.check_edge_connected_dest(next_edge, dest_edge)
            if not check:
                new_feasible_actions[idx] = 0
                logger.warning(f"Filter out edge {next_edge} not connected to dest {dest_edge}")

        b = np.where(np.array(new_feasible_actions) == 1)[0]
        logger.warning(f"After filtering, feasible actions indices: {[self.edges_list[idx] for idx in b]}")

        return new_feasible_actions

    def penalty_agents_loops(self, not_completed_trips, episode_checker):
        """
        Apply penalty for agents if they cannot reach their destination within the maximum simulation time.
        """
       # if current mode is training, we store the exp and update policy
        if self.training:
            for vehicle_id in not_completed_trips:

                dest_edge = self.trips[vehicle_id]['to']  # Get the destination edge of the vehicle
                agent = self.groups[dest_edge]  # Get the agent for the vehicle's group

                # Get the history edges of the vehicle
                history_edges = episode_checker.get_history_edges_trip(vehicle_id)
                if not history_edges:
                    logger.warning(f"No history edges found for vehicle {vehicle_id}. Skipping penalty.")
                    continue
                else:
                    curr_edge = history_edges[-1]
                    previous_edge = history_edges[-2] if len(history_edges) > 1 else None

                if not previous_edge:
                    logger.warning(f"No previous edge found for vehicle {vehicle_id}. Skipping penalty.")
                    continue

                # Store experience for training
                encoded_current_observation = self.encode_next_oberservation(curr_edge)
                encoded_previous_observation = self.encode_next_oberservation(previous_edge)

                # action is the index of current edge
                action = self.edges_list.index(curr_edge) if curr_edge in self.edges_list else -1
                if action == -1:
                    logger.warning(f"Current edge {curr_edge} not found in edges list for vehicle {vehicle_id}. Skipping penalty.")
                    continue
                
                # we take the reward as the -T (negative expected time to dest)
                # find route from selected next edge to destination edge
                logger.debug(f"Reward type {self.reward_type}")
                route = traci.simulation.findRoute(curr_edge, dest_edge, routingMode=traci.constants.ROUTING_MODE_AGGREGATED)
                if len(route.edges) != 0:  
                    travel_time = route.travelTime

                    if self.reward_type == "R1":
                        reward = self.reward.get_penalty_loop_R1(travel_time)
                    elif self.reward_type == "R1R2":
                        reward = self.reward.get_penalty_loop_R1R2(travel_time, curr_edge, dest_edge, vehicle_id)
        
                    logger.debug(f"Penalty agent loop: {reward} from edge {previous_edge}-> edge {curr_edge}")
                else:
                    
                    if self.reward_type == "R1":
                        reward = self.reward.get_penalty_loop_R1_in_dead_edge(self.CONFIG['simulation_steps'])
                    elif self.reward_type == "R1R2":
                        reward = self.reward.get_penalty_loop_R1R2_in_dead_edge()

                    logger.debug(f"Penalty agent loop for dead edge: {reward} from edge {previous_edge}-> edge {curr_edge}")

                agent.replay_buffer.push((encoded_previous_observation, action, encoded_current_observation, reward))

                # Update the agent's policy
                # agent.update(self.plotter)

    def add_reward_reaching_dest(self, completed_trips, episode_checker):
        """
        Apply reward for agent as the travel time when they complete the trips
        """
       # if current mode is training, we store the exp and update policy
        if self.training:
            for vehicle_id in self.trips.keys():
                
                if vehicle_id not in completed_trips:
                    continue

                dest_edge = self.trips[vehicle_id]['to']  # Get the destination edge of the vehicle
                agent = self.groups[dest_edge]['agent']  # Get the agent for the vehicle's group

                # Get the history edges of the vehicle
                history_edges = episode_checker.get_history_edges_trip(vehicle_id)
                if not history_edges:
                    logger.warning(f"No history edges found for vehicle {vehicle_id}. Skipping penalty.")
                    continue
                else:
                    curr_edge = history_edges[-1]
                    previous_edge = history_edges[-2] if len(history_edges) > 1 else None

                if not previous_edge:
                    logger.warning(f"No previous edge found for vehicle {vehicle_id}. Skipping penalty.")
                    continue

                # Store experience for training
                encoded_current_observation = self.encode_next_oberservation(curr_edge)
                encoded_previous_observation = self.encode_next_oberservation(previous_edge)

                # action is the index of current edge
                action = self.edges_list.index(curr_edge) if curr_edge in self.edges_list else -1
                if action == -1:
                    logger.error(f"Current edge {curr_edge} not found in edges list for vehicle {vehicle_id}. Skipping penalty.")
                    continue
                
                # we take the reward as the travel time when agemt complete the trip
                removed_time = episode_checker.get_time_vehicle_removed(vehicle_id)
                # depart_time = traci.vehicle.getDeparture(vehicle_id)
                depart_time = self.trips[vehicle_id]['depart']  # Get the departure time of the vehicle
                reward = depart_time - removed_time
                logger.debug(f"Add reward {reward} for vehicle {vehicle_id} reaching destination")

                agent.replay_buffer.push((encoded_previous_observation, action, encoded_current_observation, reward))

                # Update the agent's policy
                agent.update(self.plotter)

    def initialize_plotter(self, max_episodes, resume_episode=None):
        """
        Initialize the plotter for visualizing agent performance.
        """
        self.plotter = MultiAgentPlotter(list(self.groups.keys()), max_episodes, resume_episode)  # Initialize the plotter

    def end_episode_hooks(self):
        """After each training episode: step learning-rate schedules, log weight sizes and decision counts."""
        keys = ("input_proj.weight", "transformer.layers.0.self_attn.in_proj_weight",
                "transformer.layers.0.self_attn.out_proj.weight", "transformer.layers.0.linear1.weight",
                "transformer.layers.0.linear2.weight", "transformer.layers.1.self_attn.in_proj_weight",
                "transformer.layers.1.linear2.weight", "hidden_layers.0.weight", "output_layer.weight")
        for group_id, agent in self.groups.items():
            params = dict(agent.q_network.named_parameters())
            for k in keys:
                if k in params:
                    rms = params[k].detach().float().pow(2).mean().sqrt().item()
                    self.plotter.write_weight_stat(group_id, k, rms)
            self.plotter.write_decision_counts(group_id, agent.decision_counts)
            agent.decision_counts = [0, 0, 0, 0]
            if getattr(agent, 'stat_updates', 0):
                n = agent.stat_updates
                loss_raw = float(agent.stat_loss_raw) / n
                w_mean = float(agent.stat_weight) / n if not isinstance(agent.stat_weight, float) else float('nan')
                st = agent.replay_buffer.stats() if hasattr(agent.replay_buffer, 'stats') else {}
                self.plotter.write_train_stats(group_id, n, loss_raw, w_mean,
                                               agent.stat_done_picks / max(1, agent.stat_picks),
                                               st.get('done_size', -1), st.get('size', -1), st.get('beta', float('nan')))
                if getattr(agent, 'vdn_stats_on', False):
                    m = max(1.0, float(agent.stat_multi))
                    self.plotter.write_vdn_stats(group_id, float(agent.stat_loss_group) / n, float(agent.stat_loss_car) / n,
                                                 float(agent.stat_gnorm) / n, 100.0 * float(agent.stat_clipped) / n,
                                                 100.0 * float(agent.stat_hidden) / m, int(float(agent.stat_multi)))
            if getattr(agent, 'td_debug', False) and agent.td_acc is not None:
                self.plotter.write_td_stats(group_id, agent.td_acc.cpu().tolist())
            if hasattr(agent, 'reset_train_stats'):
                agent.reset_train_stats()
            if agent.lr_scheduler is not None:
                agent.lr_scheduler.step()
            if hasattr(agent.replay_buffer, 'set_beta'):
                agent.replay_buffer.set_beta(self.episodes_done + 1, int(self.CONFIG.get('num_episodes', 100)))
                if self.beta_hold_from is not None and self.episodes_done + 1 >= self.beta_hold_from:
                    agent.replay_buffer.beta = agent.replay_buffer.beta_end   # resume.beta: hold
            if getattr(agent, 'tau_end', None) is not None:
                tau = agent.set_episode_tau(self.episodes_done + 1)
        self.episodes_done += 1
        if self.CONFIG['softmax'].get('tau_end') is not None:
            print(f"softmax tau for episode {self.episodes_done}: {tau:.3f}", flush=True)

    def save_arrived_vehicles(self, arrived_vehicles):
        self.plotter.write_arrived_vehicles(arrived_vehicles)

    def save_all_models(self, episode=None):  
        """Save models for all agents in all groups"""  
        os.makedirs(self.saved_folder, exist_ok=True)  
        
        # Create temp directory if it doesn't exist  
        os.makedirs(self.saved_folder+"/temp", exist_ok=True)  
        
        # Read or create temp file for this episode  
        temp_file_path = self.saved_folder+f"/temp/temp_{episode}.txt"  
        try:  
            # Try to read existing index  
            with open(temp_file_path, 'r') as f:  
                N = int(f.read().strip())  
        except (FileNotFoundError, ValueError):  
            # Create new file with index 0 if file doesn't exist or has invalid content  
            N = 0  
            with open(temp_file_path, 'w') as f:  
                f.write(str(N))  
            
        # Increment index  
        index = N + 1  
        
        # Write the incremented index back to the file  
        with open(temp_file_path, 'w') as f:  
            f.write(str(index))  
        
        # Create a metadata file for this save session  
        metadata = {  
            'episode': episode,  
            'index': index,  
            'groups': {}  
        }  
        
        # Save each agent's model  
        for dest_edge, group in self.groups.items():  
            agent = group
            # Create model directory if it doesn't exist  
            os.makedirs(self.saved_folder+"/model", exist_ok=True) 
            model_path = f"{self.saved_folder}/model/agent_{dest_edge}_ep{episode}_{index}.pt"  
            agent.save_model(model_path)  
            
            # Record in metadata  
            metadata['groups'][dest_edge] = {  
                'model_path': model_path
      
                # 'avg_reward': self.plotter.get_avg_reward(agent.group_id) if self.plotter else None  
            }  
      
        # Create meta directory if it doesn't exist  
        os.makedirs(self.saved_folder+"/meta", exist_ok=True)  
        with open(f"{self.saved_folder}/meta/metadata_ep{episode}_{index}.json", 'w') as f:
            json.dump(metadata, f, indent=4)
        self.save_run_state(episode, index)
            
        logger.success(f"All models saved. Episode: {episode}, Index: {index}")  
        return str(index)  # Return index as string instead of timestamp

    def save_run_state(self, episode, index):
        """save_state: random-number states (rng: true, every checkpoint) and replay buffers (buffer: end = last episode,
        every = each checkpoint) in <saved_folder>/state/, so a later resume continues exactly."""
        import pickle, random
        ss = self.CONFIG.get('save_state') or {}
        d = os.path.join(self.saved_folder, "state")
        if ss.get('rng'):
            os.makedirs(d, exist_ok=True)
            torch.save({"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                       os.path.join(d, f"rng_ep{episode}_{index}.pt"))
        mode = str(ss.get('buffer', 'off')).lower()
        if mode == 'every' or (mode == 'end' and int(episode) >= int(self.CONFIG.get('num_episodes', 0))):
            os.makedirs(d, exist_ok=True)
            for g, agent in self.groups.items():
                if not hasattr(agent.replay_buffer, 'state_dict'):
                    continue
                t0, p = time.time(), os.path.join(d, f"buffer_{g}_ep{episode}_{index}.pkl")
                with open(p, 'wb') as f:
                    pickle.dump(agent.replay_buffer.state_dict(), f, protocol=5)
                print(f"saved buffer {g}: {agent.replay_buffer.size} items, {os.path.getsize(p) / 1e6:.1f} MB, "
                      f"{time.time() - t0:.1f} s", flush=True)

    def load_run_state(self, episode, folder, seed):
        """resume: restore the random-number states and replay buffers saved at `episode` in `folder` (newest index).
        No RNG state saved: reseed (resume.reseed: same = seed, offset = seed + episode). No buffer saved: start empty."""
        import pickle, random
        d = os.path.join(folder, "state")
        rng = sorted(glob.glob(os.path.join(d, f"rng_ep{episode}_*.pt")), key=os.path.getmtime)
        if rng:
            st = torch.load(rng[-1], weights_only=False)
            random.setstate(st["python"]); np.random.set_state(st["numpy"]); torch.set_rng_state(st["torch"])
            if st.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(st["cuda"])
            rng_msg = f"restored {os.path.basename(rng[-1])}"
        else:
            s = int(seed) if seed is not None else 0
            if str((self.CONFIG.get('resume') or {}).get('reseed', 'same')).lower() == 'offset':
                s += int(episode)
            random.seed(s); np.random.seed(s); torch.manual_seed(s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(s)
            rng_msg = f"not saved, reseeded with {s}"
        buf_msg = []
        for g, agent in self.groups.items():
            f = sorted(glob.glob(os.path.join(d, f"buffer_{g}_ep{episode}_*.pkl")), key=os.path.getmtime)
            if f and hasattr(agent.replay_buffer, 'load_state_dict'):
                with open(f[-1], 'rb') as fh:
                    agent.replay_buffer.load_state_dict(pickle.load(fh))
                buf_msg.append(f"{g}: {agent.replay_buffer.size} items")
            else:
                buf_msg.append(f"{g}: starts empty")
            if self.beta_hold_from is not None and hasattr(agent.replay_buffer, 'beta_end'):
                agent.replay_buffer.beta = agent.replay_buffer.beta_end
        return rng_msg, "; ".join(buf_msg)

    def load_models(self, index=None, episode=None, eval_mode=True, fine_tune_mode=False, resume_mode=False, folder=None):
        """
        Load saved models for all agents
        resume_mode: continue training (see Agent.load_model); folder: another run's model folder (resume.from_run)

        Args:  
            index: Specific index to load models from (matches the new saving logic)  
            episode: Specific episode to load models from  
            eval_mode: Whether to set agents to evaluation mode  
        """  
        saved_folder = folder or self.saved_folder
        # Ensure folder exists
        if not os.path.exists(saved_folder):
            logger.success(f"Saved models folder {saved_folder} not found")
            return False

        # Find the metadata file based on criteria  
        metadata_path = None  

        # If index and episode are provided (primary lookup, matches new save pattern)  
        if index and episode:  
            metadata_path = f"{saved_folder}/meta/metadata_ep{episode}_{index}.json"  

        # If only index is provided  
        elif index:  
            files = glob.glob(f"{saved_folder}/meta/metadata_ep*_{index}.json")  
            if files:  
                metadata_path = files[0]  

        # If only episode is provided  
        elif episode:  
            files = glob.glob(f"{saved_folder}/meta/metadata_ep{episode}_*.json")  
            if files:  
                # Sort by modification time (newest first)  
                files.sort(key=lambda x: os.path.getmtime(x), reverse=True)  
                metadata_path = files[0]  

        # Find the latest metadata file if neither is provided  
        else:  
            files = glob.glob(f"{saved_folder}/meta/metadata_ep*.json")  
            if files:  
                files.sort(key=lambda x: os.path.getmtime(x), reverse=True)  
                metadata_path = files[0]  

        if not metadata_path or not os.path.exists(metadata_path):  
            logger.warning(f"No saved models found matching criteria: index={index}, episode={episode}")  
            return False  

        # Load metadata  
        try:  
            with open(metadata_path, 'r') as f:  
                metadata = json.load(f)  
        except json.JSONDecodeError:  
            logger.error(f"Failed to parse metadata file {metadata_path}")  
            return False  

        logger.info(f"Loading models from: {metadata_path}")  
        logger.info(f"Episode: {metadata['episode']}, Index: {metadata.get('index', 'unknown')}")  # updated key  

        # # Load each agent's model  
        # success = True  
        # for agent_edge, group_meta in metadata['groups'].items():  
        #     if agent_edge in self.groups:  
        #         model_path = group_meta['model_path']  
        #         if os.path.exists(model_path):  
        #             logger.info(f"Loading model for agent {agent_edge} from {model_path}")  
        #             agent = self.groups[agent_edge]  
        #             if agent.load_model(model_path) and eval_mode:  
        #                 agent.set_eval_mode()  
        #                 logger.info("Set inference mode for models successuly")
        #         else:  
        #             logger.error(f"Model file not found: {model_path}")  
        #             success = False  
        #     else:  
        #         logger.warning(f"Group {agent_edge} not found in current simulation.")  
        success = True
        for agent_edge, group_meta in metadata['groups'].items():
            if agent_edge in self.groups:
                model_path = group_meta['model_path']
                # use the file next to this metadata when it exists: the stored path is absolute and points to the machine
                # that trained the model (released weights are read from new_models/<run>/model/)
                local_path = os.path.join(saved_folder, "model", os.path.basename(model_path.replace("\\", "/")))
                if resume_mode or os.path.exists(local_path):
                    model_path = local_path
                if os.path.exists(model_path):
                    logger.info(f"Loading model for agent {agent_edge} from {model_path}")
                    agent = self.groups[agent_edge]

                    # NEW: pass fine_tune_mode flag
                    if agent.load_model(model_path, fine_tune_mode=fine_tune_mode, resume_mode=resume_mode):
                        self.loaded_model_paths = getattr(self, 'loaded_model_paths', []) + [model_path]
                        if resume_mode:
                            logger.info("Model loaded in resume mode (training continues).")
                        elif fine_tune_mode:
                            logger.info("Model loaded in fine-tune mode.")
                        elif eval_mode:
                            agent.set_eval_mode()
                            logger.info("Set inference mode for models successfully.")
                    else:  # architecture mismatch or unreadable file: do not run with random weights (noenc_ablation_plan U4)
                        logger.error(f"Model could not be loaded: {model_path}")
                        success = False
                else:
                    logger.error(f"Model file not found: {model_path}")
                    success = False
            else:
                logger.warning(f"Group {agent_edge} not found in current simulation.")

        return success
    
    def load_offline_models(self, eval_mode=True):

        model_dir = self.CONFIG['offline_models_path']
        # Ensure folder exists  
        if not os.path.exists(model_dir):  
            logger.error(f"Saved offline models folder {model_dir} not found")  
            return
        
        pattern = re.compile(r'^(?P<agent_name>.+)_eps_(?P<eps>\d+)_time_.*$')

        # find models with latest episodes
        latest_models = {}
        for filename in os.listdir(model_dir):
            match = pattern.match(filename)
            if match:
                agent_name = match.group('agent_name')
                eps = int(match.group('eps'))
                filepath = os.path.join(model_dir, filename)

                # Keep only the model with the highest eps
                if agent_name not in latest_models or eps > latest_models[agent_name]['eps']:
                    latest_models[agent_name] = {'eps': eps, 'path': filepath}

        success = True
        # Load models
        for agent_name, info in latest_models.items():
            if agent_name in self.groups:
                agent = self.groups[agent_name]['agent']
                if agent.load_offline_model(info['path'], agent_name) and eval_mode:  
                        agent.set_eval_mode()  
                        logger.info("Set inference mode for model {agent_name} successuly")
            else:
                success = False
                logger.warning("Agent {agent_name} not in group of agents")

        return success


class _Slot:
    """One vehicle inside a group transition (evac_time mode)."""

    def __init__(self, rec):
        self.vid, self.obs, self.action, self.dis = rec['vid'], rec['obs'], rec['action'], rec['dis']
        self.reward, self.dt, self.done = 0.0, 0.0, 0.0
        self.next_obs, self.next_dis, self.next_feas = None, None, None
        self.steps = 0
        self.closed = False

    def add(self, reward, dt, done, gamma, next_obs, next_dis, next_feas=None):
        """Add the vehicle's next decision outcome: reward discounted by gamma^steps, next state moves on (B1).
        With n_step = 1 this is called once, giving exactly the one-step transition."""
        self.reward += (gamma ** self.steps) * float(reward)
        self.dt += float(dt)
        self.done = float(done)
        self.next_obs, self.next_dis, self.next_feas = next_obs, next_dis, next_feas
        self.steps += 1


class _GroupTransition:
    """Vehicles of one group that decided in the same second; pushed when all members are closed."""

    def __init__(self, timestep, recs):
        self.timestep = timestep
        self.slots = []
        for rec in recs:
            slot = _Slot(rec)
            rec['slot'] = slot
            self.slots.append(slot)

    def complete(self):
        return all(sl.closed for sl in self.slots)

    def has_final(self):
        return any(sl.done for sl in self.slots)

    def reward_sum(self):
        return float(sum(sl.reward for sl in self.slots))

    def to_dict(self):
        # member tuple: (vid, obs, action, next_obs, done, disaster_obs, reward, dt, next_disaster_obs,
        #                next_feasible_indices (B0, None when off), steps (B1))
        vehicles = [(sl.vid, sl.obs, sl.action, sl.next_obs, sl.done, sl.dis, sl.reward, sl.dt, sl.next_dis,
                     sl.next_feas, sl.steps)
                    for sl in self.slots]
        return {'timestep': self.timestep, 'vehicles': vehicles,
                'global_reward': self.reward_sum(), 'per_vehicle': True}
