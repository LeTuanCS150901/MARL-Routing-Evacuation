import os
import time
import random
import numpy as np
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from read_config import read_config

from group_agents import MultiAgentManager
from graph_manager import GraphManager
from logs.logger import logger, setLoggerEnable  # import our custom logger

from simulation import run_simulation, reset_simulation, run_simulation_inference
from disaster_infor.disaster_move import DisasterMove

import torch
torch.set_num_threads(int(os.environ.get("BL_TORCH_THREADS", "16")))  # r22: launcher sets fewer threads per run

#  CUDA/GPU Setup
if torch.cuda.is_available():
    # Print to confirm Pytorch sees the GPU
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")

    # Optional: Optimizes performance if input sizes (obs_dim) don't change
    torch.backends.cudnn.benchmark = True
else:
    print("CUDA is NOT available. Running on CPU.")

SCRIPT_DIR = Path(__file__).resolve().parent


def set_seed(seed):
    """r22: run seed for random, NumPy, torch and CUDA (ported from the paper main.py, 33187d9).
    Default 42 = the value the tip fixed (only when CUDA was found)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_config_path(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else (SCRIPT_DIR / path).resolve()


def _ensure_child(parent, tag):
    child = parent.find(tag)
    if child is None:
        child = ET.SubElement(parent, tag)
    return child


def _set_child_value(parent, tag, value):
    _ensure_child(parent, tag).set("value", str(value))


def prepare_run_sumo_config(CONFIG, output_dir, seed):
    """r22: write <output_dir>/run.sumocfg from the config's .sumocfg (as the paper main.py, 33187d9):
    absolute net path, route file = CONFIG['route_file'], outputs and SUMO log in output_dir, the run seed,
    and the processing options of the config block `sumocfg_processing` (vdo rules: tls.all-off,
    no-internal-links, threads)."""
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    source = resolve_config_path(CONFIG["sumo_config"])
    tree = ET.parse(source)
    root = tree.getroot()
    input_node = _ensure_child(root, "input")
    net = input_node.find("net-file")
    net.set("value", str((source.parent / net.get("value")).resolve()))
    _set_child_value(input_node, "route-files", resolve_config_path(CONFIG["route_file"]))
    output_node = _ensure_child(root, "output")
    for tag, name in (("summary-output", "summary.xml"), ("tripinfo-output", "tripinfo.xml"),
                      ("vehroute-output", "routes.xml")):
        _set_child_value(output_node, tag, output_path / name)
    _set_child_value(_ensure_child(root, "report"), "log", output_path / "sumo.log")
    _set_child_value(_ensure_child(root, "random_number"), "seed", seed)
    processing = root.findall("processing")
    for tag, value in (CONFIG.get("sumocfg_processing") or {}).items():
        node = None
        for proc in processing:          # change the option where the source file has it
            node = proc.find(tag)
            if node is not None:
                break
        if node is None:
            parent = processing[0] if processing else ET.SubElement(root, "processing")
            node = ET.SubElement(parent, tag)
        node.set("value", str(value))
    run_cfg = output_path / "run.sumocfg"
    ET.indent(tree, space="    ")
    tree.write(run_cfg, encoding="utf-8", xml_declaration=True)
    CONFIG["sumo_config"] = str(run_cfg)
    CONFIG["saved_route"] = str(output_path / "routes.xml")
    return run_cfg


def _early_stop_check(CONFIG, multi_agents, episode):
    """r22 early stop (user, 2026-11-07): after episode `episode` of the config block (1-based), stop when the
    mean arrival rate of the last `window` episodes is below `min_arrival`. Returns that rate, else None."""
    es = CONFIG.get("early_stop") or {}
    at = int(es.get("episode") or 0)
    if not at or episode + 1 != at:
        return None
    window = int(es.get("window", 10))
    threshold = float(es.get("min_arrival", 0.2))
    hist = multi_agents.arrival_history[-window:]
    rate = float(np.mean(hist)) if hist else 0.0
    print(f"EARLY_STOP_CHECK ep{at} arrival={rate * 100:.1f}% (mean of episodes {at - window + 1}-{at}), "
          f"threshold {threshold * 100:.0f}%", flush=True)
    return rate if rate < threshold else None

def initialize(CONFIG):
    net_file = CONFIG['net_file']
    route_file = CONFIG['route_file']
    graph =  GraphManager(route_file, net_file)
    # calculate distance of edges to destination edges
    graph.calculate_distances(net_file)
    # calculate distance of edges to disaster zones
    graph.calculate_distance_to_disaster_zones(net_file)
    # calculate distance from O->d
    graph.calculate_dist_OD(net_file)

    logger.info(f"Graph loaded. Number of nodes: {graph.network.number_of_nodes()}")
    logger.info(f"Number of edges: {graph.network.number_of_edges()}")

    # Global dictionary to track each vehicle's last processed edge.
    # visited_edge = {vehicle_id: [] for vehicle_id in graph.trips.keys()}
    visited_edge = {}
    multi_agents = MultiAgentManager(graph)
    multi_agents.divide_trips_to_agent_groups()

    # Init disaster object
    poi_file = None
    try:
        poi_file = CONFIG['disaster']['POI_path'] # path to file contains disaster coordinates
    except Exception:
        logger.warning("No POI file of disaster found => disater object = None")
    disaster_move = None
    if poi_file:
        logger.success("ACtivate offline training for data collection")
        disaster_move = DisasterMove(poi_file, graph)
        disaster_move.init_disaster_params()

    init = {
        "edges": graph.edges,
        "trips": graph.trips,
        "visited_edge": visited_edge,
        "multi_agents": multi_agents,
        "disaster_move": disaster_move,
    }
    return init

def train(num_episodes, simulation_steps, save_freq, CONFIG=None):  
    """Training function with model saving"""  
    # Initialize environment and agents  
    flag_save_freq = True
    while flag_save_freq:
        try:
            save_freq = int(save_freq)
            if save_freq <= 0:
                int(None)
            flag_save_freq = False
            print("Thank you very much !")
        except:
            save_freq = input("Please input a positive value for the save frequency !\n")

    result = initialize(CONFIG)
    edges = result["edges"]
    trips = result["trips"]
    last_edge = result["visited_edge"]
    multi_agents = result["multi_agents"]
    disaster_move = result["disaster_move"]

    # Initialize plotter  
    multi_agents.initialize_plotter(max_episodes=num_episodes)

    # r22 continue training (user, 2026-11-15): --set resume_from=<run> --set resume_episode=<N> loads that run's
    # checkpoint ep<N> (weights + Adam state, index 1) and goes on with episode N+1 ... num_episodes.
    # The PPO buffer (cleared after each update) and the random-number state start fresh.
    start = 0
    if CONFIG.get("resume_from"):
        start = int(CONFIG["resume_episode"])
        own = multi_agents.saved_folder
        multi_agents.saved_folder = str((SCRIPT_DIR / ".." / "new_models" / CONFIG["resume_from"]).resolve())
        ok = multi_agents.load_models(index=1, episode=start, eval_mode=False, fine_tune_mode=False)
        multi_agents.saved_folder = own
        if not ok:
            raise SystemExit(f"RESUME_FAILED {CONFIG['resume_from']} ep{start}")
        for agent in multi_agents.groups.values():
            agent.set_train_mode()          # load_model() leaves the network in eval mode
        multi_agents.plotter.current_episode = start
        print(f"RESUME_FROM {CONFIG['resume_from']} ep{start} -> episodes {start + 1}-{num_episodes}, "
              f"models to {own}", flush=True)

    for episode in range(start, num_episodes):
        start_time = time.time()
        logger.info(f"Starting episode {episode + 1}/{num_episodes}")  
        
        # Reset the simulation  
        reset_simulation(episode, CONFIG)  
        last_edge.clear()  
        
        # random activate/deactiate disaster for a new episode
        # if no disaster provided. The function do nothing
        multi_agents.random_active_disaster()  
        
        # Run one episode  
        run_simulation(edges=edges, trips=trips, last_edge=last_edge,  
                      multi_agents=multi_agents, CONFIG=CONFIG, simulation_steps=simulation_steps,
                      disaster_move=disaster_move)  
        
        logger.info(f"Episode {episode + 1} completed.")  
        multi_agents.plotter.increase_episode()  

        # track the time taken for each epoch
        end_time = time.time()
        epoch_duration = end_time - start_time
        logger.trace(f"Epoch {episode}/{num_episodes} took {epoch_duration:.2f} seconds")
        print(f"Epoch {episode}/{num_episodes} took {epoch_duration} seconds")
        
        # r22: checkpoints at episodes save_freq, 2*save_freq, ... (labels ep50, ep100, ...), not on the final one
        if (episode + 1) % save_freq == 0 and episode < num_episodes - 1:
            multi_agents.save_all_models(episode=episode+1)

        rate = _early_stop_check(CONFIG, multi_agents, episode)
        if rate is not None:
            print(f"EARLY_STOP {CONFIG.get('_run_name')} ep{episode + 1} arrival={rate * 100:.1f}%", flush=True)
            return None

        time.sleep(1)

    # Final save at the end of training (only once)  
    timestamp = multi_agents.save_all_models(episode=num_episodes)  
    # multi_agents.plotter.write_to_file() 
    
    logger.info(f"Training completed. Models saved with timestamp: {timestamp}")  
    return timestamp  

def run_inference(index=None, episode=None, simulation_steps=None, CONFIG=None, offline_infer=None):  
    """Run inference using saved models"""  
    if simulation_steps is None:  
        simulation_steps = CONFIG['simulation_steps']  
    
    # Initialize environment and agents  
    result = initialize(CONFIG)
    edges = result["edges"]
    trips = result["trips"]
    last_edge = result["visited_edge"]
    multi_agents = result["multi_agents"]
    disaster_move = result["disaster_move"]
        
    # initialize plotter
    multi_agents.initialize_plotter(max_episodes=1)  
     
    # set mode as inference
    multi_agents.set_training(mode=False)

    # load models, either from offline training or online training
    if offline_infer:
        # set mode for selecting action in offline way
        multi_agents.set_offline_infer(mode=True)
        success = multi_agents.load_offline_models()
    else:
        success = multi_agents.load_models(index=index, episode=episode, eval_mode=True)  
    
    if not success:  
        logger.error("Failed to load models. Cannot run inference.")  
        return  
    
    logger.info("Models loaded successfully. Running inference...")  

    # Reset simulation  
    reset_simulation(episode=0, CONFIG=CONFIG)  
    last_edge.clear()  
    
    # random activate/deactiate disaster for a new episode
    # if no disaster provided. The function do nothing
    multi_agents.random_active_disaster()  
        
    # Run simulation with loaded models (no learning) 
    # For now no disaster is added into the model
    run_simulation_inference(edges=edges, trips=trips, last_edge=last_edge,  
                  multi_agents=multi_agents, CONFIG=CONFIG, simulation_steps=simulation_steps, disaster_move=disaster_move)  
    
    logger.info("Inference completed")  

def fine_tune(index, num_finetune_episodes, max_episode, simulation_steps, save_freq, CONFIG=None):
    """
    Fine-tuning function:
    - Loads pretrained models from checkpoints.
    - Freezes encoders, trains MLP/output layers with updated reward weighting.
    - Updates optimizer (small lr, fresh).
    """
    logger.info("===== Starting Fine-Tuning Mode =====")
    flag_save_freq = True
    while flag_save_freq:
        try:
            save_freq = int(save_freq)
            if save_freq <= 0:
                int(None)
            flag_save_freq = False
            print("Thank you very much !")
        except:
            save_freq = input("Please input a positive value for the save frequency !\n")

    # Step 1: Initialize environment and agents
    result = initialize(CONFIG)
    edges = result["edges"]
    trips = result["trips"]
    last_edge = result["visited_edge"]
    multi_agents = result["multi_agents"]
    disaster_move = result["disaster_move"]

    # Step 2: Load pretrained models, and freeze encoder, unfreeze MLP, create fine-tune optimizer
    success = multi_agents.load_models(index=index, episode=num_finetune_episodes, eval_mode=False, fine_tune_mode=True)  
    if not success:
        logger.error("Failed to load pretrained models. Cannot start fine-tuning.")
        return

    # Step 4: Initialize plotter
    multi_agents.initialize_plotter(max_episodes=num_finetune_episodes)

    # Step 5: Run fine-tuning episodes
    for episode in range(num_finetune_episodes+1, max_episode):
        start_time = time.time()
        logger.info(f"[Fine-tune] Episode {episode}/{num_finetune_episodes}")

        reset_simulation(episode, CONFIG)
        last_edge.clear()

        # random activate/deactiate disaster for fine-tuning
        multi_agents.random_active_disaster()

        # Use the same simulation logic, but with learning ON and new reward weights
        run_simulation(edges=edges, trips=trips, last_edge=last_edge,
                    multi_agents=multi_agents, CONFIG=CONFIG, 
                    simulation_steps=simulation_steps, disaster_move=disaster_move)

        multi_agents.plotter.increase_episode()

        end_time = time.time()
        logger.trace(f"Fine-tune episode took {end_time - start_time:.2f}s")

        # Save fine-tuned models periodically
        if episode % save_freq == 0:
            timestamp = multi_agents.save_all_models(episode=f"ft_{episode}")

    logger.info(f"Fine-tuning completed. Models saved with timestamp: {timestamp}")

    return timestamp


def main():  
    parser = argparse.ArgumentParser(description='RL for Traffic Routing')  
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference', 'finetune'],  
                       help='Run mode: train or inference')  
    parser.add_argument('--max_episode', type=int, default=None,
                       help='Max episode number for fine-tuning')
    parser.add_argument('--episodes', type=int, default=None,  
                       help='Number of episodes for training')  
    parser.add_argument('--testcase', type=str, default=None,  
                       help='The testcase config yaml file in config folder')  
    parser.add_argument('--steps', type=int, default=None,  
                       help='Number of simulation steps per episode')  
    parser.add_argument('--save_freq', type=int, default=None,  
                       help='Save frequency (episodes)')  
    parser.add_argument('--index', type=int, default=None,  
                       help='index is the run number of training. E.g, traning number 1,2,3,...')  
    parser.add_argument('--episode', type=int, default=None,  
                       help='Episode number of saved models to load for inference/fine-tuning')  
    parser.add_argument('--gui',       action='store_true',
                        help='Launch SUMO with GUI (sumo-gui) instead of sumo') # boolean flag for gui
    parser.add_argument('--offline_infer',  action='store_true',
                        help='Run offline model for inference') # boolean flag for offline_inference
    parser.add_argument('--log', action='store_true',
                        help='Enable Loguru console output')  # <-- added option
    # r22 options
    parser.add_argument('--seed', type=int, default=42,
                        help='Run seed: random, NumPy, torch, CUDA and SUMO (default 42 = the tip value)')
    parser.add_argument('--route-file', type=str, default=None,
                        help='Demand file: a file name in the config route folder, or a path')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Logs in result/<saved_dir>/<run-name>/, models in new_models/<run-name>/')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Folder for run.sumocfg and the SUMO outputs (default result/<saved_dir>/<run-name>/sumo)')
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                        help='Override a config value (repeatable; nested keys with dots, e.g. model.PPO.batch_size=5000)')

    args = parser.parse_args()

    # Set the config file
    CONFIG = read_config(config_filename=args.testcase)
    import yaml
    for item in args.set:
        key, _, raw = item.partition("=")
        node = CONFIG
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
        print(f"config override: {key} = {node[parts[-1]]!r}", flush=True)
    if args.route_file:
        route_path = Path(args.route_file)
        if not route_path.is_absolute() and not resolve_config_path(route_path).exists():
            route_path = Path(CONFIG["route_file"]).parent / args.route_file
        if not resolve_config_path(route_path).exists():
            raise FileNotFoundError(f"Route file not found: {args.route_file}")
        CONFIG["route_file"] = str(route_path)
    run_name = args.run_name or f"{args.mode}_s{args.seed}"
    CONFIG["_run_name"] = run_name
    CONFIG["saved_dir"] = str(Path(CONFIG["saved_dir"]) / run_name)
    CONFIG["models_dir"] = str((SCRIPT_DIR / ".." / "new_models" / run_name).resolve())
    if CONFIG.get("models_from"):       # inference: load the models of another run (folder new_models/<run>)
        CONFIG["models_dir"] = str((SCRIPT_DIR / ".." / "new_models" / CONFIG["models_from"]).resolve())
    out_dir = args.output_dir or str((SCRIPT_DIR / ".." / "result" / CONFIG["saved_dir"] / "sumo").resolve())
    if CONFIG.get("dist_validate_args") is False:
        # r22 speed switch (D6, P1/P2): skip torch.distributions argument checks; no change in values or random draws
        torch.distributions.Distribution.set_default_validate_args(False)
    set_seed(args.seed)
    CONFIG["_seed"] = args.seed
    run_cfg = prepare_run_sumo_config(CONFIG, out_dir, seed=args.seed)
    algo = str(CONFIG.get("algorithm", "mat")).lower()
    m = CONFIG.get("model") or {}
    pp, mt = m.get("PPO") or {}, m.get("MAT") or {}
    print("RUN_CONFIG " + " ".join(f"{k}={v}" for k, v in [
        ("run", run_name), ("mode", args.mode), ("algorithm", algo), ("seed", args.seed),
        ("demand", Path(CONFIG["route_file"]).name), ("net", Path(CONFIG["net_file"]).name),
        ("sumocfg", run_cfg), ("episodes", args.episodes or CONFIG["num_episodes"]),
        ("steps", args.steps or CONFIG["simulation_steps"]), ("lr", CONFIG["learning_rate"]),
        ("grad_clip", CONFIG["max_grad_norm"]),
        ("buffer", pp.get("batch_size") if algo == "mappo" else mt.get("memory_buffer_size")),
        ("gae_lambda", pp.get("gae_lamda") if algo == "mappo" else "unused"),
        ("entropy", pp.get("entropy_coef") if algo == "mappo" else mt.get("entropy_coef")),
        ("w1", CONFIG["reward"]["R1R2"]["w1"]), ("w2", CONFIG["reward"]["R1R2"]["w2"]),
        ("zones", len(CONFIG["disaster"]["fire"].get("disaster_zones") or {})),
        ("late_fine", CONFIG.get("late_fine", False)), ("shelter_rule", CONFIG.get("shelter_rule", False)),
        ("hazard_close", CONFIG.get("hazard_close", False)), ("sumo_rules", CONFIG.get("sumo_rules")),
        ("sumocfg_processing", CONFIG.get("sumocfg_processing")), ("models_dir", CONFIG["models_dir"]),
        ("torch_threads", torch.get_num_threads()), ("dist_validate_args", CONFIG.get("dist_validate_args", True))]),
        flush=True)

    # Override CONFIG values if specified  
    num_episodes = args.episodes if args.episodes is not None else CONFIG['num_episodes']  
    simulation_steps = args.steps if args.steps is not None else CONFIG['simulation_steps']  
    # Set which SUMO binary to call
    CONFIG['sumo_binary'] = 'sumo-gui' if args.gui else 'sumo'

    offline_infer = False
    if args.offline_infer:
        offline_infer = True

    # enable loguru console output
    if args.log:
        setLoggerEnable(True)

    if args.mode == 'train':  
        logger.info(f"Starting training for {num_episodes} episodes, {simulation_steps} steps per episode")  
        train(num_episodes, simulation_steps, save_freq=args.save_freq, CONFIG=CONFIG)  

    if args.mode == 'inference':  # inference mode  
        logger.info("Starting inference with saved models")  
        run_inference(index=args.index, episode=args.episode, simulation_steps=simulation_steps, CONFIG=CONFIG,
                      offline_infer=offline_infer)  
        
    if args.mode == 'finetune':
        logger.info("Starting fine tune with saved models")  
        fine_tune(index=args.index, num_finetune_episodes=args.episode, max_episode=args.max_episode, simulation_steps=simulation_steps, 
                  save_freq=args.save_freq, CONFIG=CONFIG)

if __name__ == "__main__":  
    main()  
    
