"""Inference-time hazard avoidance. Off unless the config says so.

hazard_mask: true   -> the agent may not choose a hazard road, nor a road from which its shelter can only be reached
                       through a hazard road (reverse search from the shelter over hazard-free connections). If every
                       candidate is masked, the unmasked candidates are kept (fallback, counted).
hazard_close: true  -> after SUMO starts (before any car departs), every lane of a hazard road is closed to passenger
                       cars (traci.lane.setDisallowed), so SUMO's router never plans through them: not the first route
                       at departure, not the route completed after each agent choice (findRoute, aggregated mode).
                       A long travel time alone is not enough: the paper code uses ROUTING_MODE_AGGREGATED, which
                       ignores traci.edge.adaptTraveltime.
Hazard roads = CONFIG['hazard_close_edges'] when set (closing only; the model's disaster input stays as trained),
else every edge of CONFIG['disaster']['fire']['disaster_zones']."""
import atexit
import collections
import os
import re

import numpy as np

CLOSED_TRAVEL_TIME = 1e6   # s; makes a hazard road the last resort for SUMO's router


def hazard_edges(config):
    if config.get('hazard_close_edges'):
        return set(config['hazard_close_edges'])
    zones = ((config.get('disaster') or {}).get('fire') or {}).get('disaster_zones') or {}
    return {e for edges in zones.values() for e in (edges or [])}


def close_hazard_roads(config, traci):
    edges = sorted(hazard_edges(config))
    lanes = []
    for e in edges:
        traci.edge.adaptTraveltime(e, CLOSED_TRAVEL_TIME)
        for i in range(traci.edge.getLaneNumber(e)):
            traci.lane.setDisallowed(f"{e}_{i}", ["passenger"]); lanes.append(f"{e}_{i}")
    print(f"hazard_close: {len(edges)} hazard roads closed to passenger cars ({len(lanes)} lanes): {edges}", flush=True)


class HazardFilter:
    def __init__(self, config):
        self.hazard = hazard_edges(config)
        net = open(os.path.abspath(config['net_file']), encoding='utf-8').read()
        self.into = collections.defaultdict(set)   # to-edge -> from-edges, normal edges only
        for a, b in re.findall(r'<connection from="([^":][^"]*)" to="([^":][^"]*)"', net):
            self.into[b].add(a)
        self.safe = {}                              # shelter -> edges with a hazard-free path to it
        self.counts = collections.Counter()
        atexit.register(lambda: print(f"hazard_mask summary: {dict(self.counts)}", flush=True))
        print(f"hazard_mask: {len(self.hazard)} hazard roads {sorted(self.hazard)}", flush=True)

    def safe_edges(self, dest):
        if dest not in self.safe:
            seen, stack = {dest}, [dest]
            while stack:
                for a in self.into[stack.pop()]:
                    if a not in seen and a not in self.hazard:
                        seen.add(a); stack.append(a)
            self.safe[dest] = seen
        return self.safe[dest]

    def apply(self, feasible_actions, edges_list, dest):
        safe = self.safe_edges(dest)
        masked = [v if (v == 1 and edges_list[i] not in self.hazard and edges_list[i] in safe) else 0
                  for i, v in enumerate(feasible_actions)]
        self.counts['decisions'] += 1
        if not any(masked):
            self.counts['fallback'] += 1
            return feasible_actions
        if masked != list(feasible_actions):
            self.counts['masked'] += 1
        return masked
