"""Static route baseline B for Q = B + residual.

For each shelter g and every road a (free-flow road times t_k = edges[k]['travel_time'], computed once):
  T1(a)   = fastest time from a to g, own road included (as findRoute(a, g).travelTime)
  n*(a)   = next road on the fastest route that avoids hazards (route cost t_k + route_fine_s on hazard roads)
  R0(a)   = -w1*T1(a) + w2*T2(a) - P*1[a is a hazard road]      (get_reward_5 without the delay penalty)
            include_t2: false drops the w2*T2 bonus: every R0 < 0, so greedy-by-B moves to a strictly higher B at
            each step and can never loop (with the bonus, U-turn pairs next to the shelter pay and B loops)
  B(g)    = s*R0(g);   B(a) = s*R0(a) + gamma*B(n*(a))           (s = reward_scale)
B(a) is the discounted, scaled return of stepping onto a and then following that route. Roads with no route to g
get B = s*(-P). The table never changes during training."""
import heapq

import numpy as np


class RouteBaseline:
    def __init__(self, manager, cfg, reward_scale, gamma):
        self.roads = list(manager.edges_list)
        self.idx = {e: i for i, e in enumerate(self.roads)}
        self.w1, self.w2 = float(manager.w1), float(manager.w2)
        self.P = float(manager.CONFIG['simulation_steps'])
        self.s, self.gamma = float(reward_scale), float(gamma)
        self.route_fine = float(cfg.get('route_fine_s', self.P))
        self.include_t2 = bool(cfg.get('include_t2', True))
        self.t = {e: max(float(manager.edges[e]['travel_time']), 1e-3) for e in self.roads}
        self.hazard = set(manager.hazard_edges)
        self.rev = {}
        for e in self.roads:
            for n in set(manager.edges[e]['connected_edges']):
                if n in self.idx:
                    self.rev.setdefault(n, []).append(e)
        self.t2 = {e: self._time_to_disaster(manager, e) for e in self.roads}
        self.no_route = {}
        self.T1, self.next_road, self.B = {}, {}, {}
        for g in manager.destination_edges:
            self.T1[g], _ = self._dijkstra(g, 0.0)
            route_cost, self.next_road[g] = self._dijkstra(g, self.route_fine)
            self.B[g] = self._values(g, route_cost)

    @staticmethod
    def _time_to_disaster(manager, e):
        """T2(e): distance to the nearest disaster zone / median speed of that zone's roads (as get_reward_5)."""
        if not manager.disaster_info or e not in manager.edges_dist_disaster:
            return 0.0
        ids = list(manager.disaster_info.keys())
        d = [manager.edges_dist_disaster[e][i] for i in ids]
        k = int(np.argmin(d))
        return d[k] / np.median([manager.edges[x]['speed'] for x in manager.disaster_zones[ids[k]]])

    def _dijkstra(self, goal, fine):
        """Fastest time from every road to goal (own road included); next road on that route."""
        c0 = self.t[goal] + (fine if goal in self.hazard else 0.0)
        T, nxt, pq = {goal: c0}, {goal: None}, [(c0, goal)]
        while pq:
            d, e = heapq.heappop(pq)
            if d > T[e]:
                continue
            for pr in self.rev.get(e, []):
                c = d + self.t[pr] + (fine if pr in self.hazard else 0.0)
                if c < T.get(pr, float('inf')):
                    T[pr] = c; nxt[pr] = e; heapq.heappush(pq, (c, pr))
        return T, nxt

    def r0(self, g, a):
        bonus = self.w2 * self.t2[a] if self.include_t2 else 0.0
        return -self.w1 * self.T1[g][a] + bonus - (self.P if a in self.hazard else 0.0)

    def _values(self, g, route_cost):
        B = np.full(len(self.roads), -self.s * self.P, dtype=np.float64)
        known = {}
        for a in sorted(route_cost, key=route_cost.get):      # the next road on the route always comes first
            if a not in self.T1[g]:
                continue
            n = self.next_road[g][a]
            known[a] = self.s * self.r0(g, a) + (self.gamma * known[n] if n is not None else 0.0)
            B[self.idx[a]] = known[a]
        self.no_route[g] = len(self.roads) - len(known)
        return B.astype(np.float32)

    def vector(self, g):
        """B of every road (edges_list order) for shelter g."""
        return self.B[g]
