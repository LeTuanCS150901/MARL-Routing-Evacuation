"""Train one model (Coop-Eva and its ablations, or the MAPPO / MAT baselines).

Examples (from routing-MARL):
  python train.py --config randers --run-name r200_s1                         # Coop-Eva 
  python train.py --config randers --demand randers_train120.rou.xml --run-name r120_s1
  python train.py --config manhattan_noenc --run-name mne_s1                  # ablation: no encoder
  python train.py --config karlsruhe_lstm --run-name kls_s1                   # ablation: LSTM encoder
  python train.py --algo mappo --config karlsruhe --run-name kmappo_s1         # baseline
  python train.py --config randers --run-name smoke --episodes 1 --steps 300  # quick check

Outputs:
  runs/train/<run-name>/        stdout.log (console copy), settings.json, pid.txt, done.txt, sumo/ (last episode's SUMO files)
  result/<saved_dir>/<run-name>/ training curves (CSV); see README "Results during training"
  new_models/<run-name>/        checkpoints every --save-freq episodes
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):   # the code prints a few non-ASCII signs; never fail on a narrow console
    sys.stdout.reconfigure(errors="replace")

ROOT = Path(__file__).resolve().parent                     # routing-MARL
BL = ROOT / "baselines" / "mappo_mat"
BL_CONFIG = {"karlsruhe": "bl_karl", "manhattan": "bl_man", "randers": "bl_randers"}

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--config", required=True, help="config name in config/ (karlsruhe, manhattan, randers, *_noenc, *_lstm)")
p.add_argument("--run-name", required=True, help="name of the run: folder names for logs, checkpoints and SUMO files")
p.add_argument("--algo", choices=["coopeva", "mappo", "mat"], default="coopeva")
p.add_argument("--demand", default=None, help="training demand file in the network folder (default: route_file in the config)")
p.add_argument("--seed", type=int, default=1)
p.add_argument("--episodes", type=int, default=None, help="default: num_episodes in the config")
p.add_argument("--steps", type=int, default=None, help="simulated seconds per episode (default: simulation_steps in the config)")
p.add_argument("--save-freq", type=int, default=10, help="save a checkpoint every N episodes")
p.add_argument("--resume-from", type=int, default=None, help="Coop-Eva: continue this run from its checkpoint of episode N")
p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value (repeatable), e.g. learning_rate=1e-5")
p.add_argument("--sumo-threads", type=int, default=1)
p.add_argument("--torch-threads", type=int, default=4)
p.add_argument("--gui", action="store_true", help="watch the simulation in sumo-gui (slow; uses TraCI)")
a = p.parse_args()

run_dir = ROOT / "runs" / "train" / a.run_name
run_dir.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, PYTHONIOENCODING="utf-8", TORCH_THREADS=str(a.torch_threads))
env.pop("PYTHONPATH", None)

common = ["--mode", "train", "--seed", str(a.seed), "--run-name", a.run_name, "--save_freq", str(a.save_freq),
          "--output-dir", str(run_dir / "sumo")]
common += ["--episodes", str(a.episodes)] if a.episodes else []
common += ["--steps", str(a.steps)] if a.steps else []
if not a.demand:   # always pass the demand file: SUMO and the agents must read the same one
    import yaml
    cfg_file = (BL / "config" / f"{BL_CONFIG.get(a.config.split('_')[0], a.config)}.yaml") if a.algo != "coopeva" else ROOT / "config" / f"{a.config}.yaml"
    a.demand = Path(yaml.safe_load(cfg_file.read_text(encoding="utf-8"))["route_file"]).name
common += ["--route-file", a.demand]
common += ["--gui"] if a.gui else []
sets = list(a.set)

if a.algo == "coopeva":
    if a.resume_from:
        sets.append(f"resume.from_episode={a.resume_from}")
    cmd = [sys.executable, str(ROOT / "run.py"), "--testcase", a.config, *common,
           "--tls-off", "--no-internal-links", "--sumo-threads", str(a.sumo_threads)]
    cwd = ROOT / "routing_MARl"
else:
    net = a.config.split("_")[0]
    if net not in BL_CONFIG:
        sys.exit(f"--algo {a.algo}: --config must be karlsruhe, manhattan or randers")
    cmd = [sys.executable, str(BL / "routing_MARl" / "main.py"), "--testcase", BL_CONFIG[net], *common,
           "--set", f"algorithm={a.algo}", "--set", "dist_validate_args=false"]
    cwd = BL / "routing_MARl"
for s in sets:
    cmd += ["--set", s]

(run_dir / "settings.json").write_text(json.dumps({"args": vars(a), "command": cmd, "started": time.strftime("%Y-%m-%d %H:%M:%S")},
                                                  indent=1), encoding="utf-8")
print("command:", subprocess.list2cmdline(cmd), flush=True)
print(f"console copy: {run_dir / 'stdout.log'}", flush=True)
with open(run_dir / "stdout.log", "w", encoding="utf-8") as log:
    proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    for line in proc.stdout:
        sys.stdout.write(line)
        log.write(line)
        log.flush()
    rc = proc.wait()
done = run_dir / "done.txt"   # Coop-Eva runs: 'ok' once training finished and saved (see run.py for why not the exit code)
if a.algo == "coopeva" and done.exists():
    rc = 0 if done.read_text(encoding="utf-8", errors="replace").startswith("ok") else (rc or 1)
print(f"exit {rc}; checkpoints in {ROOT / 'new_models' if a.algo == 'coopeva' else BL / 'new_models'} / {a.run_name}", flush=True)
sys.exit(rc)
