"""Risk-weighted remaining time to each shelter for the rwtH reward.

T~H_lam(e) = min over routes e -> shelter of sum_k [ t_k * (1 + lam * exp(-d_k / r_safe)) + F * 1[k is a hazard road] ]
  t_k = aggregated SUMO driving time of road k (findRoute(k, k, AGGREGATED), the source of T_rem in get_reward_5)
  d_k = distance from road k to the nearest disaster zone (edges_dist_disaster; 0 on hazard roads)
The value includes the road's own cost, as findRoute(e, shelter).travelTime does. One reverse Dijkstra per shelter,
refreshed every refresh_s simulated seconds."""
import time

import numpy as np
import traci
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


class RiskTimeTable:
    def __init__(self, manager, lam, r_safe=500.0, fine_s=0.0, refresh_s=10.0):
        self.lam, self.r_safe, self.fine_s, self.refresh_s = float(lam), float(r_safe), float(fine_s), float(refresh_s)
        self.roads = list(manager.edges_list)
        self.idx = {e: i for i, e in enumerate(self.roads)}
        n = len(self.roads)
        self.t_free = np.array([max(float(manager.edges[e]['travel_time']), 1e-3) for e in self.roads])
        rows, cols = [], []                     # reversed arcs: n <- e for every connection e -> n
        for e in self.roads:
            for nx in set(manager.edges[e]['connected_edges']):
                if nx in self.idx:
                    rows.append(self.idx[nx]); cols.append(self.idx[e])
        self.arc_r, self.arc_c = np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64)
        self.hazard = np.array([e in manager.hazard_edges for e in self.roads])
        d = np.full(n, np.inf)
        zones = list(manager.disaster_info.keys()) if manager.disaster_info else []
        for i, e in enumerate(self.roads):
            dd = manager.edges_dist_disaster.get(e) or {}
            if zones and dd:
                d[i] = min(float(dd[z]) for z in zones)
        d[self.hazard] = 0.0
        self.h = np.exp(-d / self.r_safe)       # 0 where no distance is known (no zones)
        self.shelters = list(manager.destination_edges)
        self.sources = [self.idx[s] for s in self.shelters]
        self.T = {}
        self.last = None
        self.refresh_seconds = 0.0              # wall time spent in refresh() this episode
        self.refreshes = 0

    def reset_episode(self):
        self.last = None
        self.refresh_seconds = 0.0
        self.refreshes = 0

    def edge_times(self):
        """Aggregated driving time of every road (free-flow time if SUMO gives no route for it)."""
        t = self.t_free.copy()
        mode = traci.constants.ROUTING_MODE_AGGREGATED
        for i, e in enumerate(self.roads):
            try:
                r = traci.simulation.findRoute(e, e, routingMode=mode)
                if r.edges:
                    t[i] = max(float(r.travelTime), 1e-3)
            except traci.TraCIException:
                pass
        return t

    def compute(self, t):
        """T~H for every road and shelter from road times t (also used by the tests)."""
        cost = t * (1.0 + self.lam * self.h) + self.fine_s * self.hazard
        n = len(self.roads)
        A = csr_matrix((cost[self.arc_c], (self.arc_r, self.arc_c)), shape=(n, n))
        D = dijkstra(A, directed=True, indices=self.sources)
        return {s: D[k] + cost[self.sources[k]] for k, s in enumerate(self.shelters)}

    def refresh(self, now):
        t0 = time.perf_counter()
        self.T = self.compute(self.edge_times())
        self.last = now
        self.refresh_seconds += time.perf_counter() - t0
        self.refreshes += 1

    def maybe_refresh(self, now):
        if self.last is None or now - self.last >= self.refresh_s:
            self.refresh(now)

    def value(self, road, shelter):
        i = self.idx.get(road)
        return float(self.T[shelter][i]) if i is not None and shelter in self.T else float('inf')
