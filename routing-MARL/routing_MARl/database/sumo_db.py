import traci 
from logs.logger import logger
from collections import defaultdict

class SumoDB:

    def __init__(self):
        self.edge_connected_dest = defaultdict(dict)
        # appended route per (next edge, destination); one route per edge alone mixed up the groups
        self.append_routes = {}


    def check_edge_connected_dest(self, next_edge: str, dest_edge: str) -> bool:
        """
        Check if an edge is connected to a destination edge.
        Uses SUMO's traci API to get the outgoing edges and checks if dest_edge is among them.
        Caches results for efficiency.

        Args:
            next_edge (str): The next edge ID
            dest_edge (str): The destination edge ID
        Returns:
            bool: True if edge_id is connected to dest_edge, False otherwise
        """
        if next_edge not in self.edge_connected_dest or dest_edge not in self.edge_connected_dest[next_edge]:
            self.add_edge_connected_dest(next_edge, dest_edge)
    
        check = self.edge_connected_dest[next_edge][dest_edge]
        logger.info(f"Edge {next_edge} -> {dest_edge}: {check}")
        return check
       
    def add_edge_connected_dest(self, next_edge: str, dest_edge: str):

        route = traci.simulation.findRoute(
                next_edge,
                dest_edge,
                routingMode=traci.constants.ROUTING_MODE_AGGREGATED
            )
        
        # If route.edges is empty or no route found
        if len(route.edges) == 0:
            self.edge_connected_dest[next_edge][dest_edge] = False
         
        else:
            self.edge_connected_dest[next_edge][dest_edge] = True 
            self.append_routes[(next_edge, dest_edge)] = list(route.edges)

    
    def get_append_route(self, next_edge: str, dest_edge: str):
        route = self.append_routes.get((next_edge, dest_edge))
        if route is None:
            logger.error(f"No append route from {next_edge} to {dest_edge}")
        return route
        
        # logger.info(f"Edge {next_edge} to destination {dest_edge} connected: {self.edge_connected_dest[next_edge][dest_edge]}")

            

