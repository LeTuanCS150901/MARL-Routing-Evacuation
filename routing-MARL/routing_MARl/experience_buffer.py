# import random
# from collections import deque

# class JointExperienceReplayBuffer:
#     """
#     Stores joint transitions: all vehicles' experiences from the same timestep.
#     """
#     def __init__(self, capacity):
#         self.buffer = deque(maxlen=capacity)
#         self.capacity = capacity
    
#     def push(self, joint_transition):
#         """
#         Store a joint transition.
        
#         Args:
#             joint_transition: dict with keys:
#                 - 'timestep': int
#                 - 'vehicles': list of (vehicle_id, obs, action, next_obs, done)
#                 - 'global_reward': float (shared reward for all vehicles)
#         """
#         self.buffer.append(joint_transition)
    
#     def sample(self, batch_size):
#         """Sample batch_size joint transitions"""
#         return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
#     def __len__(self):
#         return len(self.buffer)


import random

import numpy as np


class JointExperienceReplayBuffer:
		"""
		Prioritized Joint Experience Replay Buffer.

		Same main interface as old buffer:
			self.replay_buffer.push(joint_transition)
			batch = self.replay_buffer.sample(batch_size)
			len(self.replay_buffer)

		Extra optional method:
			self.replay_buffer.update_priorities(td_errors)

		Main idea:
			- high TD-error transitions are sampled more often
			- some uniform sampling is kept for stability
			- some recent sampling is kept for adaptation
		"""

		def __init__(
			self,
			capacity,
			alpha=0.6,
			priority_eps=1e-6,
			priority_ratio=0.70,
			uniform_ratio=0.20,
			recent_ratio=0.10,
			recent_window=None,
			skip_empty=True,
		):
			self.capacity = capacity

			self.buffer = []
			self.priorities = np.zeros(capacity, dtype=np.float32)

			self.position = 0
			self.size = 0

			# priority strength
			# alpha = 0 means uniform replay
			# alpha = 1 means strong prioritization
			self.alpha = alpha
			self.priority_eps = priority_eps

			# sampling mixture
			total = priority_ratio + uniform_ratio + recent_ratio
			self.priority_ratio = priority_ratio / total
			self.uniform_ratio = uniform_ratio / total
			self.recent_ratio = recent_ratio / total

			if recent_window is None:
				recent_window = max(100, capacity // 10)

			self.recent_window = recent_window

			# This stores indices of the last sampled batch.
			# It allows update_priorities(td_errors) without changing sample output.
			self.last_sampled_indices = []

			# If True, do not store invalid transitions with no vehicles.
			self.skip_empty = skip_empty

		def __len__(self):
			return self.size

		def push(self, joint_transition):
			"""
			Store one joint transition.

			joint_transition should contain:
				- 'vehicles'
				- 'global_reward'
				- other optional fields
			"""

			if self.skip_empty:
				vehicles = joint_transition.get("vehicles", [])
				if vehicles is None or len(vehicles) == 0:
					return

			# New samples should be seen at least once.
			if self.size > 0:
				max_priority = float(np.max(self.priorities[:self.size]))
				if max_priority <= 0:
					max_priority = 1.0
			else:
				max_priority = 1.0

			if self.size < self.capacity:
				self.buffer.append(joint_transition)
				idx = self.size
				self.size += 1
			else:
				idx = self.position
				self.buffer[idx] = joint_transition

			self.priorities[idx] = max_priority
			self.position = (self.position + 1) % self.capacity

		def _priority_probs(self):
			"""
			Compute sampling probabilities from priorities.
			"""

			priorities = self.priorities[:self.size].astype(np.float64)
			priorities = np.maximum(priorities, self.priority_eps)

			scaled = priorities ** self.alpha
			total = np.sum(scaled)

			if total <= 0 or not np.isfinite(total):
				return np.ones(self.size, dtype=np.float64) / self.size

			return scaled / total

		def _recent_indices(self):
			"""
			Return indices of the most recent transitions.
			"""

			if self.size == 0:
				return []

			n_recent = min(self.size, self.recent_window)

			indices = []
			for i in range(n_recent):
				idx = (self.position - 1 - i) % self.capacity
				if idx < self.size:
					indices.append(idx)

			return indices

		def sample(self, batch_size):
			"""
			Sample a batch.

			Return format is exactly the same as old buffer:
				batch = list of joint_transition
			"""

			if self.size == 0:
				self.last_sampled_indices = []
				return []

			batch_size = min(batch_size, self.size)

			n_priority = int(batch_size * self.priority_ratio)
			n_recent = int(batch_size * self.recent_ratio)
			n_uniform = batch_size - n_priority - n_recent

			sampled_indices = []

			# 1. Prioritized samples
			if n_priority > 0:
				probs = self._priority_probs()

				indices = np.random.choice(
					self.size,
					size=n_priority,
					replace=True,
					p=probs,
				)

				sampled_indices.extend([int(i) for i in indices])

			# 2. Uniform samples
			if n_uniform > 0:
				n = min(n_uniform, self.size)
				indices = random.sample(range(self.size), n)
				sampled_indices.extend(indices)

			# 3. Recent samples
			if n_recent > 0:
				recent_indices = self._recent_indices()

				if len(recent_indices) > 0:
					if len(recent_indices) >= n_recent:
						indices = random.sample(recent_indices, n_recent)
					else:
						indices = random.choices(recent_indices, k=n_recent)

					sampled_indices.extend(indices)

			# If something went wrong or batch is too small, fill with uniform samples.
			while len(sampled_indices) < batch_size:
				sampled_indices.append(random.randrange(self.size))

			# Shuffle to avoid ordered batch structure.
			random.shuffle(sampled_indices)

			self.last_sampled_indices = sampled_indices

			batch = [self.buffer[i] for i in sampled_indices]

			return batch

		def update_priorities(self, td_errors):
			"""
			Update priorities using TD errors from the last sampled batch.

			Args:
				td_errors:
					array-like of shape [batch_size]
					usually abs(q_total_current - target)
			"""

			if len(self.last_sampled_indices) == 0:
				return

			td_errors = np.asarray(td_errors, dtype=np.float32).reshape(-1)

			n = min(len(self.last_sampled_indices), len(td_errors))

			for idx, td_error in zip(self.last_sampled_indices[:n], td_errors[:n]):
				priority = abs(float(td_error)) + self.priority_eps

				if np.isfinite(priority):
					self.priorities[idx] = priority

		def stats(self):
			if self.size == 0:
				return {
					"size": 0,
					"capacity": self.capacity,
					"max_priority": 0.0,
					"mean_priority": 0.0,
				}

			return {
				"size": self.size,
				"capacity": self.capacity,
				"max_priority": float(np.max(self.priorities[:self.size])),
				"mean_priority": float(np.mean(self.priorities[:self.size])),
				"priority_ratio": self.priority_ratio,
				"uniform_ratio": self.uniform_ratio,
				"recent_ratio": self.recent_ratio,
			}

class PrioritizedReplayBuffer:
    """Proportional prioritized experience replay (Schaul et al., 2016).

    priority      p_j = |delta_j| + eps          (new items: the current max priority)
    probability   P(j) = p_j^alpha / sum_k p_k^alpha
    weight        w_j = (N P(j))^-beta / max_k (N P(k))^-beta
    beta is moved from beta_start to beta_end over the episodes with set_beta().
    A sum-tree and a min-tree over p^alpha give O(log N) sampling and updates.

    push(transition); sample(B) -> (batch, indices, weights); update_priorities(indices, td_errors)
    """

    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_end=1.0, eps=1e-6,
                 uniform_ratio=0.0, done_ratio=0.0):
        self.capacity = int(capacity)
        # F3c: share of each batch drawn uniformly; F3b: share drawn from transitions that end a trip
        self.uniform_ratio, self.done_ratio = float(uniform_ratio), float(done_ratio)
        self.is_done = np.zeros(self.capacity, dtype=bool)
        self.done_list, self.done_pos = [], {}   # done index set with O(1) add / remove
        self.last_n_done = 0
        self.alpha, self.eps = float(alpha), float(eps)
        self.beta_start, self.beta_end = float(beta_start), float(beta_end)
        self.beta = self.beta_start
        self.tree_size = 1
        while self.tree_size < self.capacity:
            self.tree_size *= 2
        self.sum_tree = np.zeros(2 * self.tree_size, dtype=np.float64)
        self.min_tree = np.full(2 * self.tree_size, np.inf, dtype=np.float64)
        self.buffer = [None] * self.capacity
        self.position = 0
        self.size = 0
        self.max_priority = 1.0
        self.last_sampled_indices = []

    def __len__(self):
        return self.size

    def _set_leaf(self, idx, value):
        i = idx + self.tree_size
        self.sum_tree[i] = value
        self.min_tree[i] = value
        i //= 2
        while i >= 1:
            self.sum_tree[i] = self.sum_tree[2 * i] + self.sum_tree[2 * i + 1]
            self.min_tree[i] = min(self.min_tree[2 * i], self.min_tree[2 * i + 1])
            i //= 2

    def _find(self, mass):
        """Index of the leaf where the running sum of p^alpha passes mass."""
        i = 1
        while i < self.tree_size:
            left = 2 * i
            if mass <= self.sum_tree[left]:
                i = left
            else:
                mass -= self.sum_tree[left]
                i = left + 1
        return i - self.tree_size

    def push(self, joint_transition):
        vehicles = joint_transition.get("vehicles", [])
        if vehicles is None or len(vehicles) == 0:
            return
        idx = self.position
        if self.is_done[idx]:  # the ring overwrites a done item: drop it from the done set
            self._done_remove(idx)
        self.buffer[idx] = joint_transition
        self._set_leaf(idx, self.max_priority ** self.alpha)
        if any(float(v[4]) == 1.0 for v in vehicles):
            self._done_add(idx)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _done_add(self, idx):
        self.is_done[idx] = True
        self.done_pos[idx] = len(self.done_list)
        self.done_list.append(idx)

    def _done_remove(self, idx):
        self.is_done[idx] = False
        k = self.done_pos.pop(idx)
        last = self.done_list.pop()
        if last != idx:
            self.done_list[k] = last
            self.done_pos[last] = k

    def sample(self, batch_size):
        if self.size == 0:
            self.last_sampled_indices = []
            return [], [], np.zeros(0, dtype=np.float32)
        batch_size = min(batch_size, self.size)
        n_done = int(round(batch_size * self.done_ratio)) if self.done_list else 0
        n_uni = int(round(batch_size * self.uniform_ratio))
        n_per = max(0, batch_size - n_done - n_uni)
        total = self.sum_tree[1]
        indices = []
        segment = total / max(1, n_per)
        for k in range(n_per):  # prioritized part, stratified as in the paper
            mass = np.random.uniform(k * segment, (k + 1) * segment)
            idx = self._find(min(mass, total - 1e-12))
            if idx >= self.size:  # numerical edge: fall back to a stored item
                idx = np.random.randint(self.size)
            indices.append(idx)
        indices += [int(i) for i in np.random.randint(self.size, size=n_uni)]                 # F3c
        indices += [self.done_list[i] for i in np.random.randint(len(self.done_list), size=n_done)] if n_done else []  # F3b
        idx_arr = np.array(indices)
        p_per = self.sum_tree[idx_arr + self.tree_size] / total
        if n_uni == 0 and n_done == 0:
            # pure prioritized replay: weights normalised by the buffer-wide maximum (Schaul et al.)
            p_min = self.min_tree[1] / total
            weights = (self.size * p_per) ** (-self.beta) / (self.size * p_min) ** (-self.beta)
        else:
            # mixture: P_mix = share_per * P_per + share_uni / N + share_done * 1[done] / |D|; batch-max normalisation
            sp, su, sd = n_per / batch_size, n_uni / batch_size, n_done / batch_size
            p_mix = sp * p_per + su / self.size + (sd * self.is_done[idx_arr] / len(self.done_list) if n_done else 0.0)
            w = (self.size * p_mix) ** (-self.beta)
            weights = w / w.max()
        self.last_sampled_indices = indices
        self.last_n_done = n_done
        return [self.buffer[i] for i in indices], indices, weights.astype(np.float32)

    def update_priorities(self, indices, td_errors):
        td_errors = np.abs(np.asarray(td_errors, dtype=np.float64).reshape(-1))
        for idx, err in zip(indices, td_errors):
            p = float(err) + self.eps
            if not np.isfinite(p):
                continue
            self.max_priority = max(self.max_priority, p)
            self._set_leaf(int(idx), p ** self.alpha)

    def set_beta(self, episode, total_episodes):
        frac = min(1.0, episode / max(1, total_episodes - 1))
        self.beta = self.beta_start + (self.beta_end - self.beta_start) * frac

    def state_dict(self):
        """Everything needed to continue sampling as before (resume): items, ring position, leaves p^alpha, done set."""
        return {"capacity": self.capacity, "buffer": self.buffer[:self.size], "position": self.position, "size": self.size,
                "max_priority": self.max_priority, "beta": self.beta,
                "leaves": self.sum_tree[self.tree_size:self.tree_size + self.size].copy(), "done_list": list(self.done_list)}

    def load_state_dict(self, d):
        if int(d["capacity"]) != self.capacity:
            raise ValueError(f"buffer capacity {d['capacity']} in the file != {self.capacity} in the config")
        n = int(d["size"])
        self.buffer = list(d["buffer"]) + [None] * (self.capacity - n)
        self.position, self.size = int(d["position"]), n
        self.max_priority, self.beta = float(d["max_priority"]), float(d["beta"])
        # rebuild both trees bottom-up from the leaves
        self.sum_tree[:] = 0.0
        self.min_tree[:] = np.inf
        self.sum_tree[self.tree_size:self.tree_size + n] = d["leaves"]
        self.min_tree[self.tree_size:self.tree_size + n] = d["leaves"]
        for i in range(self.tree_size - 1, 0, -1):
            self.sum_tree[i] = self.sum_tree[2 * i] + self.sum_tree[2 * i + 1]
            self.min_tree[i] = min(self.min_tree[2 * i], self.min_tree[2 * i + 1])
        self.is_done[:] = False
        self.done_list = [int(i) for i in d["done_list"]]
        self.done_pos = {idx: k for k, idx in enumerate(self.done_list)}
        self.is_done[self.done_list] = True

    def stats(self):
        if self.size == 0:
            return {"size": 0, "capacity": self.capacity, "beta": self.beta}
        leaves = self.sum_tree[self.tree_size:self.tree_size + self.size]
        return {"size": self.size, "capacity": self.capacity, "beta": self.beta,
                "max_priority": self.max_priority, "done_size": len(self.done_list),
                "mean_p_alpha": float(leaves.mean()), "min_p_alpha": float(leaves.min())}
