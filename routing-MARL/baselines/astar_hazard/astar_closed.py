"""Dynamic A* baseline: every car is re-routed with A* on live travel times, hazard roads get a large cost.

The defaults are (--trigger edge --cost smoothed --routing-mode 4 --adaptation-interval 1 --rules vdo
--hazard-cost 1e6 --steps 43200):
- a car is re-routed when it enters a new road (and once at departure);
- the travel time of a road is SUMO's moving average of the last 4 speed samples, one sample per second;
- the hazard roads (rings_<testcase>.json, ring D1) get +1e6 s, so A* avoids them whenever another way exists;
- SUMO rules as Coop-Eva: lights off, no internal links, junction blocker -1, collision warn.

Usage (from routing-MARL):
  python baselines/astar_hazard/astar_closed.py --testcase randers --route-file randers_fw_N2000.rou.xml --output-dir runs/astar/randers_N2000
Outputs in --output-dir: run.sumocfg, tripinfo.xml, summary.xml, routes.xml, sumo.log, metrics.json.
"""
import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import libsumo as traci

ROOT = Path(__file__).resolve().parents[2]  # routing-MARL/
HERE = Path(__file__).resolve().parent
CLOSED = 1e9
EXPECTED_SUMO = "1.26.0"
PAPER_START = ["--device.rerouting.adaptation-steps", "4", "--collision.action", "none",
               "--collision.mingap-factor", "1.0", "--ignore-junction-blocker", "1.0"]
VDO_START = ["--device.rerouting.adaptation-steps", "4", "--collision.action", "warn",
             "--collision.mingap-factor", "1.0", "--ignore-junction-blocker", "-1"]  # karl_vdo_old_rules_retrain_plan
VDI_START = VDO_START + ["--collision.check-junctions", "true"]  # vdo with internal links on
NEW_START = ["--device.rerouting.adaptation-steps", "4", "--collision.action", "teleport",
             "--collision.mingap-factor", "1.0", "--ignore-junction-blocker", "-1",
             "--collision.check-junctions", "true", "--time-to-teleport", "300"]


def set_value(parent, tag, value):
    node = parent.find(tag)
    if node is None:
        node = ET.SubElement(parent, tag)
    node.set("value", str(value))


BASE_CFG = {"karl": "karl/karl_kc.sumocfg", "manhattan2": "manhattan2/manhattan2.sumocfg", "randers": "randers/randers_kc.sumocfg"}


def write_run_config(testcase, out, seed, sumocfg=None, rules="old", route_file=None):
    """Copy the base sumocfg (default: the network's, else the demand's) into the run folder, set as the paper run's run.sumocfg
    (rules new: lights on and internal links on instead)."""
    base = Path(sumocfg).resolve() if sumocfg else ROOT / "network" / BASE_CFG.get(testcase, f"{testcase}/{testcase}.sumocfg")
    tree = ET.parse(base)
    root = tree.getroot()
    inp = root.find("input")
    for tag in ("net-file", "route-files", "additional-files"):
        node = inp.find(tag) if inp is not None else None
        if node is not None and node.get("value"):
            node.set("value", ",".join(str((base.parent / v.strip()).resolve()) for v in node.get("value").split(",")))
    if route_file:   # demand file: a name in the network folder, or a path
        rf = Path(route_file) if Path(route_file).exists() else base.parent / route_file
        if not rf.exists():
            raise SystemExit(f"route file not found: {route_file}")
        inp.find("route-files").set("value", str(rf.resolve()))
    output = root.find("output")
    if output is None:
        output = ET.SubElement(root, "output")
    for child in list(output):
        output.remove(child)
    set_value(output, "summary-output", out / "summary.xml")
    set_value(output, "tripinfo-output", out / "tripinfo.xml")
    set_value(output, "vehroute-output", out / "routes.xml")
    set_value(output, "vehroute-output.exit-times", "true")
    set_value(output, "vehroute-output.last-route", "true")
    processing = root.find("processing")
    if processing is None:
        processing = ET.SubElement(root, "processing")
    set_value(processing, "threads", 64)
    set_value(processing, "tls.all-off", "False" if rules == "new" else "True")
    set_value(processing, "no-internal-links", "False" if rules in ("new", "vdi") else "True")
    routing = root.find("routing")
    if routing is None:
        routing = ET.SubElement(root, "routing")
    set_value(routing, "routing-algorithm", "astar")
    rnd = root.find("random_number")
    if rnd is None:
        rnd = ET.SubElement(root, "random_number")
    set_value(rnd, "seed", seed)
    report = root.find("report")
    if report is None:
        report = ET.SubElement(root, "report")
    set_value(report, "log", out / "sumo.log")
    cfg = out / "run.sumocfg"
    ET.indent(tree, space="    ")
    tree.write(cfg, encoding="utf-8", xml_declaration=True)
    return cfg


def load_rings(testcase):
    rings = json.loads((HERE / f"rings_{testcase}.json").read_text(encoding="utf-8"))
    return {k: set(v) for k, v in rings.items()}


def update_costs(edges, closed, cost=CLOSED):
    """Global A* cost table: current travel time, + `cost` (default 1e9) on the hazard roads."""
    for e in edges:
        traci.edge.adaptTraveltime(e, traci.edge.getTraveltime(e) + (cost if e in closed else 0.0))


def reroute(vehicles):
    failed = 0
    for v in vehicles:
        try:
            traci.vehicle.rerouteTraveltime(v, currentTravelTimes=False)
        except traci.TraCIException:
            failed += 1
    return failed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--testcase", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--steps", type=int, default=43200, help="step limit; the run stops once every car has arrived")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--sumocfg", help="base sumocfg; default the network's (karl_kc / manhattan2 / randers_kc .sumocfg)")
    p.add_argument("--route-file", help="demand file (name in the network folder, or a path); default the one in the sumocfg")
    p.add_argument("--reroute-period", type=int, default=10)
    p.add_argument("--cost", choices=["snapshot", "smoothed"], default="smoothed")
    p.add_argument("--adaptation-interval", type=float, default=1.0, help="s between edge speed samples (smoothed cost)")
    p.add_argument("--rerouting-device", action="store_true", help="equip every car with a rerouting device, period 0")
    p.add_argument("--routing-mode", type=int, default=4, help="traci routing mode set on each car (smoothed cost)")
    p.add_argument("--trigger", choices=["period", "stagger", "edge", "none"], default="edge",
                   help="period: all cars every reroute_period s (+ at departure); stagger: every reroute_period s on "
                        "each car's own clock from its departure; edge: when a car enters a new road (+ at departure); "
                        "none: never rerouted (route given at insertion)")
    p.add_argument("--rules", choices=["old", "new", "vdo", "vdi"], default="vdo", help="old = paper rules; new = vdnpc training rules; vdo = old + blocker -1, collision warn")
    p.add_argument("--allow-version", action="store_true")
    p.add_argument("--hazard-cost", type=float, default=1e6, help="travel time (s) added to the hazard roads (paper: 1e6)")
    a = p.parse_args()
    start_opts = {"new": NEW_START, "vdo": VDO_START, "vdi": VDI_START}.get(a.rules, PAPER_START)
    if a.trigger != "period" and a.cost == "snapshot":
        raise SystemExit("--trigger stagger/edge needs --cost smoothed (snapshot costs are updated on the period)")

    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = write_run_config(a.testcase, out, a.seed, a.sumocfg, a.rules, a.route_file)
    rings = load_rings(a.testcase)
    closed = rings["D1"]
    ring_of = {e: k for k in ("D3", "D2", "D1") for e in rings[k]}

    wall = time.time()
    extra = ["--device.rerouting.adaptation-interval", f"{a.adaptation_interval:g}"]
    if a.rerouting_device:
        extra += ["--device.rerouting.probability", "1", "--device.rerouting.period", "0"]
    traci.start(["sumo", "-c", str(cfg)] + start_opts + extra)
    version = traci.getVersion()[1]
    print(f"SUMO: {version}; rules {a.rules}; start options {start_opts + extra}; cost {a.cost}; trigger {a.trigger}; hazard cost {a.hazard_cost:g} s; "
          f"closed roads {sorted(closed)}", flush=True)
    if EXPECTED_SUMO not in version and not a.allow_version:
        traci.close()
        raise SystemExit(f"SUMO {EXPECTED_SUMO} expected (Coop-Eva runs), got {version}")

    edges = [e for e in traci.edge.getIDList() if not e.startswith(":")]
    entered = {"D1": set(), "D2": set(), "D3": set()}
    reroute_failures = reroutes = 0
    arrived = teleports = 0
    dep_step, last_road = {}, {}
    if a.cost == "snapshot":
        update_costs(edges, closed, a.hazard_cost)
    else:
        for e in closed:
            traci.edge.adaptTraveltime(e, a.hazard_cost)

    step = 0
    while step < a.steps:
        traci.simulationStep()
        step += 1
        arrived += traci.simulation.getArrivedNumber()
        teleports += traci.simulation.getStartingTeleportNumber()
        departed = traci.simulation.getDepartedIDList()
        if a.routing_mode:
            for v in departed:
                traci.vehicle.setRoutingMode(v, a.routing_mode)
        if a.trigger == "period":
            if step % a.reroute_period == 0:
                if a.cost == "snapshot":
                    update_costs(edges, closed, a.hazard_cost)
                due = traci.vehicle.getIDList()
            else:
                due = departed
        elif a.trigger == "none":      # static: never rerouted
            due = []
        elif a.trigger == "stagger":   # every reroute_period s on each car's own clock, from its departure
            for v in departed:
                dep_step[v] = step
            due = [v for v in traci.vehicle.getIDList() if (step - dep_step.get(v, step)) % a.reroute_period == 0]
        else:                          # "edge": when the car enters a new road (and at departure)
            due = []
            for v in traci.vehicle.getIDList():
                r = traci.vehicle.getRoadID(v)
                if r != last_road.get(v) and not r.startswith(":"):
                    due.append(v)
                last_road[v] = r
        reroutes += len(due)
        reroute_failures += reroute(due)
        for e, ring in ring_of.items():
            vs = traci.edge.getLastStepVehicleIDs(e)
            if vs:
                entered[ring].update(vs)
        if step > 1 and traci.simulation.getMinExpectedNumber() == 0:
            break

    stuck = traci.simulation.getMinExpectedNumber()
    traci.close()
    metrics = {
        "testcase": a.testcase, "sumocfg": a.sumocfg or "default", "route_file": a.route_file, "seed": a.seed, "penalty": f"D1 closed ({a.hazard_cost:g}) only", "hazard_cost": a.hazard_cost, "reroute_period": a.reroute_period, "cost": a.cost,
        "adaptation_interval": a.adaptation_interval, "rerouting_device": a.rerouting_device, "routing_mode": a.routing_mode,
        "trigger": a.trigger, "reroutes": reroutes, "rules": a.rules,
        "sumo_version": version, "start_options": start_opts + extra, "steps_simulated": step, "step_limit": a.steps,
        "loaded": arrived + stuck, "arrived": arrived, "stuck": stuck, "teleports": teleports,
        "reroute_failures": reroute_failures, "vehicles_entered_D1": len(entered["D1"]),
        "vehicles_entered_D2": len(entered["D2"]), "vehicles_entered_D3": len(entered["D3"]),
        "wall_seconds": round(time.time() - wall, 1),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"done {a.testcase} seed={a.seed}: arrived {arrived}/{arrived + stuck}, teleports {teleports}, "
          f"entered D1 {len(entered['D1'])}, steps {step}, {metrics['wall_seconds']} s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
