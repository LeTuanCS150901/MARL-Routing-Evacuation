"""Run a trained model on one demand file and one seed (decentralized execution: each car decides from its own observation).

Examples (from routing-MARL):
  python infer.py --config randers --model r200_s1 --episode 190 --demand randers_fw_N2000.rou.xml --seed 23
  python infer.py --config randers --model r200_s1 --episode 190 --demand randers_fw_N6000.rou.xml --pool group
  python infer.py --config manhattan_noenc --model mne_s1 --episode 120 --demand manhattan2_jm_N3000.rou.xml
  python infer.py --algo mat --config karlsruhe --model kmat_s1 --episode 70 --demand karl_fw_N4000.rou.xml
  python infer.py ... --demands 2000,3000,4000,5000,6000 --seeds 23,25,29   # a demand sweep, one run after the other

The model is read from new_models/<model>/model/agent_<shelter>_ep<episode>_<index>.pt
(baselines/mappo_mat/new_models/<model>/ for MAPPO / MAT).
Outputs: runs/inference/<name>/  sumo/{tripinfo,summary,routes}.xml, sumo/sumo.log, stdout.log, metrics.json, done.txt
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):   # the code prints a few non-ASCII signs; never fail on a narrow console
    sys.stdout.reconfigure(errors="replace")

import metrics

ROOT = Path(__file__).resolve().parent
BL = ROOT / "baselines" / "mappo_mat"
BL_CONFIG = {"karlsruhe": "bl_karl", "manhattan": "bl_man", "randers": "bl_randers"}
FOLDER = {"karlsruhe": "karl", "manhattan": "manhattan2", "randers": "randers"}
PATTERN = {"karlsruhe": "karl_fw_N{}.rou.xml", "manhattan": "manhattan2_jm_N{}.rou.xml", "randers": "randers_fw_N{}.rou.xml"}

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--config", required=True, help="config name (karlsruhe, manhattan, randers, *_noenc, *_lstm)")
p.add_argument("--model", required=True, help="training run name = folder in new_models/")
p.add_argument("--episode", type=int, required=True, help="checkpoint episode, e.g. 190")
p.add_argument("--index", type=int, default=1, help="checkpoint index (the _1 in agent_<shelter>_ep190_1.pt)")
p.add_argument("--algo", choices=["coopeva", "mappo", "mat"], default="coopeva")
p.add_argument("--demand", default=None, help="demand file in the network folder, e.g. randers_fw_N2000.rou.xml")
p.add_argument("--demands", default=None, help="sweep: comma-separated car counts, e.g. 2000,3000 (uses the *_N<count> files)")
p.add_argument("--seed", type=int, default=1)
p.add_argument("--seeds", default=None, help="sweep: comma-separated seeds")
p.add_argument("--steps", type=int, default=43200, help="time limit in simulated seconds; the run ends earlier once all cars arrived")
p.add_argument("--pool", choices=["ego", "group"], default="ego",
               help="ego: each car alone (paper); group: cars of a group deciding in the same second are encoded together")
p.add_argument("--tau", type=float, default=1.0, help="softmax temperature at inference (paper: 1)")
p.add_argument("--hazard-close", action="store_true", help="close the hazard roads to cars (default: open, the agent avoids them)")
p.add_argument("--name", default=None, help="run folder name (default: <model>_ep<episode>_<demand>_s<seed>[_group])")
p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
p.add_argument("--torch-threads", type=int, default=2)
p.add_argument("--gui", action="store_true", help="watch in sumo-gui (slow; uses TraCI)")
a = p.parse_args()

net = a.config.split("_")[0]
if a.demands:
    demands = [PATTERN[net].format(int(n)) for n in a.demands.split(",")]
elif a.demand:
    demands = [a.demand]
else:
    sys.exit("give --demand <file> or --demands <counts>")
seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else [a.seed]
env = dict(os.environ, PYTHONIOENCODING="utf-8", TORCH_THREADS=str(a.torch_threads))
env.pop("PYTHONPATH", None)

rc_all = 0
for demand in demands:
    for seed in seeds:
        tag = Path(demand).name.replace(".rou.xml", "")
        name = a.name if (a.name and len(demands) * len(seeds) == 1) else \
            f"{a.model}_ep{a.episode}_{tag}_s{seed}" + ("_group" if a.pool == "group" else "")
        run_dir = ROOT / "runs" / "inference" / name
        run_dir.mkdir(parents=True, exist_ok=True)
        common = ["--mode", "inference", "--index", str(a.index), "--episode", str(a.episode), "--route-file", demand,
                  "--steps", str(a.steps), "--seed", str(seed), "--output-dir", str(run_dir / "sumo")] + (["--gui"] if a.gui else [])
        sets = [f"saved_dir=inference/{name}", f"inference_tau={a.tau:g}"] + (["hazard_close=true"] if a.hazard_close else []) + a.set
        if a.algo == "coopeva":
            cmd = [sys.executable, str(ROOT / "run.py"), "--testcase", a.config, *common, "--tls-off", "--no-internal-links",
                   "--set", f"models_subdir={a.model}", "--pool", "oracle" if a.pool == "group" else "ego", "--inference-tau", f"{a.tau:g}"]
            cwd = ROOT / "routing_MARl"
        else:
            cmd = [sys.executable, str(BL / "routing_MARl" / "main.py"), "--testcase", BL_CONFIG[net], *common, "--run-name", name,
                   "--set", f"algorithm={a.algo}", "--set", f"models_from={a.model}", "--set", "dist_validate_args=false"]
            cwd = BL / "routing_MARl"
        for s in sets:
            cmd += ["--set", s]
        print(f"[{name}] {subprocess.list2cmdline(cmd)}", flush=True)
        with open(run_dir / "stdout.log", "w", encoding="utf-8") as log:
            rc = subprocess.call(cmd, cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT)
        done = run_dir / "done.txt"   # Coop-Eva runs: 'ok' once the run finished (see run.py for why not the exit code)
        if a.algo == "coopeva" and done.exists():
            rc = 0 if done.read_text(encoding="utf-8", errors="replace").startswith("ok") else (rc or 1)
        demand_path = ROOT / "network" / FOLDER[net] / Path(demand).name
        if rc == 0 and (run_dir / "sumo" / "tripinfo.xml").exists():
            m, _ = metrics.compute(run_dir, demand_path if demand_path.exists() else None)
            (run_dir / "metrics.json").write_text(__import__("json").dumps(m, indent=1), encoding="utf-8")
            print(f"[{name}] arrived {m['arrived']}/{m['cars']}  mean {m.get('mean_trip')} s  median {m.get('median_trip')} s  "
                  f"max {m.get('max_trip')} s  last arrival {m.get('last_arrival')} s", flush=True)
        else:
            print(f"[{name}] FAILED (exit {rc}); see {run_dir / 'stdout.log'}", flush=True)
            rc_all = rc or 1
sys.exit(rc_all)
