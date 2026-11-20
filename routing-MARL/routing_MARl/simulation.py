import traci
import sys
import os
import time
import numpy as np

from logs.logger import logger  # import our custom logger
from utils.check_end_episode import CheckEndEpisode

class SumoEventLog:
    """sumo_event_log: collisions and teleports summed per 500 s of the episode, with the running and halted cars at the end of each block."""
    EVERY = 500

    def __init__(self, multi_agents, CONFIG):
        self.plotter = multi_agents.plotter if CONFIG.get('sumo_event_log') else None
        self.t_start, self.collisions, self.teleports, self.t = 0, 0, 0, 0

    def step(self, t):
        if self.plotter is None:
            return
        self.t = t + 1
        self.collisions += len(traci.simulation.getCollisions())
        self.teleports += traci.simulation.getStartingTeleportNumber()
        if self.t - self.t_start >= self.EVERY:
            self.flush()

    def flush(self):
        if self.plotter is None or self.t <= self.t_start:
            return
        ids = traci.vehicle.getIDList()
        self.plotter.write_sumo_events(self.t_start, self.collisions, self.teleports, len(ids),
                                       sum(1 for v in ids if traci.vehicle.getSpeed(v) < 0.1))
        self.t_start, self.collisions, self.teleports = self.t, 0, 0


# --- SETUP ---
if 'SUMO_HOME' not in os.environ:
    logger.error("Please set the SUMO_HOME environment variable")
    sys.exit("Please set the SUMO_HOME environment variable")

sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

def set_routing_mode(trips, vehicle_ids):
    """
    Set the routing for vehicles to always routing based on smoothed travel time of traffic condition
    """
    # logger.debug("Setting mode for vehicles routing based on smoothed travel time")
    # Get the list of active vehicle IDs in the simulation.
    # vehicle_ids = traci.vehicle.getIDList()
    # filter trips that we control. Other trips background are controlled default by SUMO
    controlled_vehicle_ids = np.intersect1d(vehicle_ids, list(trips.keys())) 
    for vid in controlled_vehicle_ids:
        traci.vehicle.setRoutingMode(vid, traci.constants.ROUTING_MODE_AGGREGATED)

    return controlled_vehicle_ids

def run_simulation(edges, trips, last_edge, multi_agents, CONFIG, simulation_steps=700, disaster_move=None):
    """
    Launch SUMO simulation using TraCI, and interact with vehicles
    by looking up edges in a NetworkX graph for re-routing.
    """
    # instance to check if the episode has ended
    episode_checker = CheckEndEpisode(trips)
    latest_depart_time = max(int(info['depart'])
                             for info in trips.values()) + 10
    multi_agents.begin_episode()  # evac_time reward: reset its per-episode state
    # run simultion
    event_log = SumoEventLog(multi_agents, CONFIG)
    for step in range(simulation_steps):

        logger.info(f"Step {step}")
        # Advance the simulation by one step.
        traci.simulationStep()
        event_log.step(step)

        if disaster_move:
            # simulate disaster evolution
            disaster_move.simulate_points(step)
            # update risk parameters after 30s interval
            if step % 30 ==0:
                disaster_move.compute_risks_numba()
                logger.success(f"Recalculated risks and Update new edge costs at step {step}")
                multi_agents.apply_new_edges_weight(disaster_move)

        # simulate vehicles routing
        try:
            # # Get the list of active vehicle IDs in the simulation.
            # vehicle_ids = traci.vehicle.getIDList()
            # # filter trips that we control. Other trips background are controlled default by SUMO
            # controlled_vehicle_ids = np.intersect1d(vehicle_ids, list(trips.keys())) 
            vehicle_ids = traci.vehicle.getIDList()
            controlled_vehicle_ids = set_routing_mode(trips, vehicle_ids)

            # S4 (batched_actions): collect this second's decisions, one network call per group, then act in order
            batched = bool(CONFIG.get('batched_actions', False)) and multi_agents.training
            requests = []
            for vid in controlled_vehicle_ids:
                if episode_checker.is_trip_completed(vid):
                    logger.info(f"Vehicle {vid} has already completed its trip. Skipping re-routing.")
                    continue

                # Retrieve the current edge (i.e., road segment) where the vehicle is located.
                curr_edge = traci.vehicle.getRoadID(vid)

                # If the current edge is internal (which starts with ":"), skip re-routing.
                if curr_edge.startswith(":"):
                    logger.warning(f"Current edge {curr_edge} is internal. Keeping original  route for vehicle {vid}.")
                    continue
                
                # If the vehicle is still on the same edge as last time,
                # skip processing this vehicle for re-routing.
                if vid in last_edge and last_edge[vid] == curr_edge:
                    # logger.info(f"Vehicle {vid} still on edge {curr_edge} from previous step; skipping re-route update.")
                    continue
                # otherwise, Record the current edge for the vehicle.
                logger.info(f"[Step {step}] Vehicle {vid} is on edge: {curr_edge}")
                last_edge[vid] = curr_edge    
                # Save the current edge into history edges of vehicle 
                episode_checker.set_edge_history_trip(vid, curr_edge)

                # Check if the current edge exists within our edge mapping (created by our network graph).
                if curr_edge in edges:
                    # Check if the vehicle has reached its destination.  
                    if curr_edge in CONFIG['destinations_edges']:
                        # if this is a true dest
                        if curr_edge == trips[vid]['to']:
                            logger.success(f"Vehicle {vid} has reached its dest: {curr_edge}. Stopping vehicle.")
                            episode_checker.mark_trip_completed(vid) # mark trip as completed
                            multi_agents.close_vehicle(vid, traci.simulation.getTime(), 'arrived')
                        # if this is a wrong dest
                        else:
                            logger.warning(f"Vehicle {vid} has reached wrong dest: {curr_edge}. Stopping vehicle.")
                            multi_agents.close_vehicle(vid, traci.simulation.getTime(), 'wrong_shelter')
                        
                        current_time = traci.simulation.getTime() 
                        episode_checker.mark_time_vehicle_removed(vid, current_time)
                        traci.vehicle.remove(vid)  # Remove the vehicle from the simulation  
                        # mark time when vehicle is removed from simulation
                        continue  # Skip further processing since the vehicle has arrived.  

                    # Get the list of possible next edges that are connected to the current edge.
                    connected_edges = edges[curr_edge]["connected_edges"]
                    if not connected_edges:
                        logger.warning(f"No connected edges found for {curr_edge}.")
                        logger.warning(f"Vehicle {vid} dissapear at {curr_edge}.")
                        multi_agents.close_vehicle(vid, traci.simulation.getTime(), 'failed')
                        continue
                    logger.info(f"Vehicle {vid} can take these next edges: {connected_edges}")
                    # Randomly select one of the connected edges as the next edge.
                    # next_edge = random.choice(connected_edges) if connected_edges else None
                    
                    try: # check if we run offline training for data collection
                        offline_training = CONFIG['offline_training']
                    except:
                        logger.warning("No offline training config found, set to False")
                        offline_training = False

                    # if do offline training
                    if offline_training:
                        next_edge = multi_agents.collect_offline_data(vid, curr_edge, connected_edges, disaster_move)
                    elif batched:
                        requests.append((vid, multi_agents.prepare_step(vid, curr_edge, connected_edges)))
                        continue
                    else:
                        next_edge, append_route = multi_agents.step(vid, curr_edge, connected_edges, disaster_move)
                    # TODO: check if offline RL => then collect data for training
                    logger.info(f"selected next_edge: {next_edge}")
                    
                else:
                    # This is dead edge
                    # Log a warning if the current edge is not found in our edge mapping.
                    logger.warning(f"{curr_edge} not found in edge map")
                    continue

                if next_edge:    
                    # Build a new route for the vehicle consisting of the current edge followed by the chosen next edge + subsequence edges.
                    # this help to avoid SUMO bug of removing vehicles when we do edge-based routing
                    # but in reality, the vehicle only go to next chosen edge, and the 
                    # rerouting will be repeated again and not follow subsequence edges
                    if append_route:
                        new_route = [curr_edge] + append_route
                        logger.success(f"Append route successfully to veh {vid}")
                    else:
                        new_route = [curr_edge, next_edge] 

                    traci.vehicle.setRoute(vid, new_route)

                else:
                    # If no valid next edge is found => current edge lead to dead edge
                    logger.warning(f"Current edge {curr_edge} of vehicle {vid} leads to dead edge. Remove vehicle from simulation")
                    multi_agents.close_vehicle(vid, traci.simulation.getTime(), 'failed')
                    traci.vehicle.remove(vid)

            if requests:
                for (vid, req), (next_edge, append_route) in zip(requests, multi_agents.step_batch(requests, disaster_move)):
                    if next_edge:
                        new_route = [req['curr_edge']] + append_route if append_route else [req['curr_edge'], next_edge]
                        traci.vehicle.setRoute(vid, new_route)
                    else:
                        logger.warning(f"Vehicle {vid} leads to dead edge. Remove vehicle from simulation")
                        multi_agents.close_vehicle(vid, traci.simulation.getTime(), 'failed')
                        traci.vehicle.remove(vid)

            # After the current simulation step, trigger training of transitions of all current vehicles
            multi_agents.end_timestep()
            multi_agents.reset_timestep_rewards()
        except Exception as e:
            # Log any exceptions that occur during the simulation step.
            logger.exception(f"Error during simulation step {step}: {e}")
            continue
        
        if episode_checker.is_ended_episode(trips):
            logger.success("-------------------All trips have arrrived. Episode ended.")
            logger.success(f"Break the SIMULATION TIME at {step} steps")
            break  

        # TODO: if all the controlled vehicles disappear (reach dest, or go to dead loop), break the simulation loop
        if step > latest_depart_time:
            if len(np.intersect1d(vehicle_ids, list(trips.keys()))) == 0:
                logger.success("-------------------All trips have disappeard. Episode ended.")
                logger.success(f"Break the SIMULATION TIME at {step} steps")
                break
    
    # evac_time reward: rule C for vehicles still driving, team term, push the held transitions
    multi_agents.end_episode(CONFIG['simulation_steps'])

    # TODO: After the end of episode, add -Penalty for vehicle not reach destination in time limit
    not_completed_trips = episode_checker.get_trips_not_completed(trips)
    if not_completed_trips:
        logger.info(f"Add Penalty for trips not completed after episode: {len(not_completed_trips)} / {len(trips)}")
        # multi_agents.penalty_agents_loops(not_completed_trips, episode_checker)

    # After the end of episode, add Reward for vehicle that reached destination in time limit
    completed_trips = episode_checker.get_trips_completed()
    if completed_trips:
        logger.info(f"Add reward for trips completed after episode: {len(completed_trips)} / {len(trips)}")
        # multi_agents.add_reward_reaching_dest(completed_trips, episode_checker)

    # Write the number of arrived vehicles at the end of episode
    # TOOD: user traci.getArrived instead
    multi_agents.save_arrived_vehicles(len(completed_trips))
  

    # Close the TraCI connection after the simulation ends.
    # traci.close(wait=True)
    event_log.flush()
    traci.close()
    print(f"SELECT_SUMMARY decisions={getattr(multi_agents, 'n_decisions', 0)} errors={getattr(multi_agents, 'n_select_errors', 0)} (since start)", flush=True)
    multi_agents.plotter.write_survived_vehicles()

def run_simulation_inference(edges, trips, last_edge, multi_agents, CONFIG, simulation_steps=1000, disaster_move=None):
    """
    Launch SUMO simulation using TraCI, and interact with vehicles
    by looking up edges in a NetworkX graph for re-routing.
    """
    # instance to check if the episode has ended
    episode_checker = CheckEndEpisode(trips)

    latest_depart_time = max(int(info['depart']) 
                             for info in trips.values()) 
    arrived_own, wrong_shelter = set(), set()

    # run simultion
    event_log = SumoEventLog(multi_agents, CONFIG)
    # R2.1: inference_pool oracle = same-second cars of a group encoded together
    pool = str(CONFIG.get('inference_pool', 'ego')).lower() == 'oracle'
    step_errors_car = str(CONFIG.get('step_errors', '')).lower() == 'car'
    n_step_errors = 0
    print(f"POOL_MODE inference_pool={CONFIG.get('inference_pool', 'ego')} inference_tau={CONFIG.get('inference_tau', 1)} "
          f"pool_max_set={CONFIG.get('pool_max_set')} step_errors={CONFIG.get('step_errors', 'paper')}", flush=True)
    for step in range(simulation_steps):

        logger.info(f"Step {step}")
        # Advance the simulation by one step.
        traci.simulationStep()
        event_log.step(step)

        if disaster_move:
            # simulate disaster evolution
            disaster_move.simulate_points(step)
            # update risk parameters after 30s interval
            if step % 30 ==0:
                disaster_move.compute_risks_numba()
                logger.success(f"Recalculated risks and Update new edge costs at step {step}")
                multi_agents.apply_new_edges_weight(disaster_move)
        try:
            # # Get the list of active vehicle IDs in the simulation.
            # vehicle_ids = traci.vehicle.getIDList()
            # # filter trips that we control. Other trips background are controlled default by SUMO
            # controlled_vehicle_ids = np.intersect1d(vehicle_ids, list(trips.keys())) 
            vehicle_ids = traci.vehicle.getIDList()
            controlled_vehicle_ids = set_routing_mode(trips, vehicle_ids)
            pending = []   # R2.1 oracle: this second's decisions

            for vid in controlled_vehicle_ids:

                if episode_checker.is_trip_completed(vid):
                    logger.info(f"Vehicle {vid} has already completed its trip. Skipping re-routing.")
                    continue

                # Retrieve the current edge (i.e., road segment) where the vehicle is located.
                curr_edge = traci.vehicle.getRoadID(vid)
                # curr_edge =  traci.vehicle.getLaneID(vid)

                # If the current edge is internal (which starts with ":"), skip re-routing.
                if curr_edge.startswith(":"):
                    logger.warning(f"Current edge {curr_edge} is internal. Keeping original  route for vehicle {vid}.")
                    continue
                
                # If the vehicle is still on the same edge as last time,
                # skip processing this vehicle for re-routing.
                if vid in last_edge and last_edge[vid] == curr_edge:
                    # logger.info(f"Vehicle {vid} still on edge {curr_edge} from previous step; skipping re-route update.")
                    continue
                # otherwise, Record the current edge for the vehicle.
                logger.info(f"[Step {step}] Vehicle {vid} is on edge: {curr_edge}")
                last_edge[vid] = curr_edge    
                # Save the current edge into history edges of vehicle 
                episode_checker.set_edge_history_trip(vid, curr_edge)

                # Check if the current edge exists within our edge mapping (created by our network graph).
                if curr_edge in edges:
                    # Check if the vehicle has reached its destination.  
                    if curr_edge in CONFIG['destinations_edges']:
                        # if this is a true dest
                        if curr_edge == trips[vid]['to']:  
                            logger.success(f"Vehicle {vid} has reached its dest: {curr_edge}. Stopping vehicle.")  
                            arrived_own.add(vid)
                        # if this is a wrong dest
                        else:
                            logger.warning(f"Vehicle {vid} has reached wrong dest: {curr_edge}. Stopping vehicle.") 
                            wrong_shelter.add(vid)
                        # end the trip on this shelter edge (normal SUMO arrival, written to tripinfo);
                        # the car no longer gets decisions. Wrong shelter = not arrived in the metrics.
                        traci.vehicle.setRoute(vid, [curr_edge])
                        episode_checker.mark_trip_completed(vid)
                        continue  # Skip further processing since the vehicle has arrived.  

                    # Get the list of possible next edges that are connected to the current edge.
                    connected_edges = edges[curr_edge]["connected_edges"]
                    if not connected_edges:
                        logger.warning(f"No connected edges found for {curr_edge}.")
                        logger.warning(f"Vehicle {vid} dissapear at {curr_edge}.")
                        continue
                    logger.info(f"Vehicle {vid} can take these next edges: {connected_edges}")
                    # Randomly select one of the connected edges as the next edge.
                    # next_edge = random.choice(connected_edges) if connected_edges else None
                    if pool:
                        try:   # per car, as the oracle's choices (e.g. a route check from a closed road)
                            req = multi_agents.prepare_step(vid, curr_edge, connected_edges)
                        except Exception as e:
                            n_step_errors += 1
                            continue
                        pending.append((vid, curr_edge, req))
                        continue
                    if step_errors_car:   # R2.1 test switch: an error ends only this car's decision (paper code: the second's loop)
                        try:
                            res = multi_agents.step(vid, curr_edge, connected_edges, disaster_move)
                            next_edge, append_route = res if res is not None else (None, None)
                            if res is None:
                                continue
                        except Exception as e:
                            n_step_errors += 1
                            continue
                    else:
                        next_edge, append_route = multi_agents.step(vid, curr_edge, connected_edges, disaster_move)
                    logger.info(f"selected next_edge: {next_edge}")
                    
                else:
                    # This is dead edge
                    # Log a warning if the current edge is not found in our edge mapping.
                    logger.warning(f"{curr_edge} not found in edge map")
                    continue

                if next_edge:    
                     # Build a new route for the vehicle consisting of the current edge followed by the chosen next edge.
                    if append_route:
                        new_route = [curr_edge] + append_route
                        logger.success(f"Append route successfully to veh {vid}")
                    else:
                        new_route = [curr_edge, next_edge]
                                  
                    traci.vehicle.setRoute(vid, new_route)
                    logger.info(f"Vehicle {vid} choose new edge: {next_edge}")
                    
                else:
                    # If no valid next edge is found => current edge lead to dead edge
                    logger.warning(f"Current edge {curr_edge} of vehicle {vid} leads to dead edge. Remove vehicle from simulation")
                    traci.vehicle.remove(vid)              

            if pending:
                for (vid, curr_edge, req), res in zip(pending, multi_agents.step_pool([p[2] for p in pending], step, disaster_move)):
                    if res is None:
                        continue   # selection or route error (counted): the car keeps its route
                    next_edge, append_route = res
                    if next_edge:
                        new_route = [curr_edge] + append_route if append_route else [curr_edge, next_edge]
                        traci.vehicle.setRoute(vid, new_route)
                    else:
                        logger.warning(f"Current edge {curr_edge} of vehicle {vid} leads to dead edge. Remove vehicle from simulation")
                        traci.vehicle.remove(vid)
            # Sleep briefly between simulation steps to slow execution for debugging.
            time.sleep(0.1)
        except Exception as e:
            # Log any exceptions that occur during the simulation step.
            logger.exception(f"Error during simulation step {step}: {e}")
            continue
        
        # if step >= latest_depart_time +640:
        #     logger.success("Remove remaining vehs")
        #     [traci.vehicle.remove(veh, reason=2) for veh in controlled_vehicle_ids]

        # after all trips are maked as reach destination
        if episode_checker.is_ended_episode(trips):
            # we check if all the controlled vehicle removed from the simulation or not
            done_simulation = True
            for trip in list(trips.keys()):
                if trip in controlled_vehicle_ids:
                    done_simulation = False

            if done_simulation: 
                # if all the controlled vehicle removed from the simulation
                logger.error("-------------------All trips have arrrived. Episode ended.")
                logger.error(f"Break the SIMULATION TIME at {step} steps") 
                break 
        
        # check trips not completed after simulation
        not_completed_trips = episode_checker.get_trips_not_completed(trips)
        if step > latest_depart_time and not_completed_trips:
            logger.info(f"Trips not completed: {not_completed_trips}")
                
    
    print(f"own shelter: {len(arrived_own)}, wrong shelter: {len(wrong_shelter)}, "
          f"still driving: {len(trips) - len(arrived_own) - len(wrong_shelter)}", flush=True)
    if step_errors_car or pool:
        print(f"STEP_ERRORS per car: {n_step_errors}", flush=True)
    if pool:
        multi_agents.pool_summary(os.path.dirname(os.path.abspath(CONFIG['sumo_config'])))
    # Close the TraCI connection after the simulation ends.
    event_log.flush()
    traci.close()
    print(f"SELECT_SUMMARY decisions={getattr(multi_agents, 'n_decisions', 0)} errors={getattr(multi_agents, 'n_select_errors', 0)}", flush=True)

def reset_simulation(episode, CONFIG):  
    """  
    Resets the SUMO simulation by starting a new simulation instance.  
    """ 
    logger.info(f"Starting SUMO simulation at episode {episode + 1}...")
    
    sumo_cmd = [
                CONFIG['sumo_binary'] , "-c",
                CONFIG['sumo_config'], "--start", 
                "--routing-algorithm", CONFIG['routing_algorithm'],
                "--device.rerouting.adaptation-steps", "4",
                "--collision.action", "none",
                "--collision.mingap-factor", "1.0",
                "--ignore-junction-blocker", "1.0"
            ]
    # sumo_rules (optional config block): junction and collision rules; without it, the old rules above
    rules = CONFIG.get('sumo_rules') or {}
    def _set(opt, val):
        if opt in sumo_cmd:
            sumo_cmd[sumo_cmd.index(opt) + 1] = str(val)
        else:
            sumo_cmd.extend([opt, str(val)])
    if 'ignore_junction_blocker' in rules:
        _set("--ignore-junction-blocker", rules['ignore_junction_blocker'])
    if 'collision_action' in rules:
        _set("--collision.action", rules['collision_action'])
    if 'collision_check_junctions' in rules:
        _set("--collision.check-junctions", str(bool(rules['collision_check_junctions'])).lower())
    if 'time_to_teleport' in rules:
        _set("--time-to-teleport", rules['time_to_teleport'])
    if rules.get('collision_output'):
        _set("--collision-output", rules['collision_output'])
    if rules.get('statistic_output'):
        _set("--statistic-output", rules['statistic_output'])
    if rules:
        logger.info(f"sumo_rules applied: {rules}")

    traci.start(sumo_cmd)  # Start a new instance of SUMO  
    if CONFIG.get('hazard_close'):
        from hazard_mask import close_hazard_roads
        close_hazard_roads(CONFIG, traci)
