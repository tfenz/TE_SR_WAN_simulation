import time

from algorithm.generic_sr import GenericSR
from algorithm.segment_routing.equal_split_shortest_path import EqualSplitShortestPath


class InverseCapacity(GenericSR):
    def __init__(self, nodes: list, links: list, demands: list, weights: dict = None, waypoints: dict = None, **kwargs):
        super().__init__(nodes, links, demands, weights, waypoints)

        self.__nodes = nodes  # [i, ...]
        self.__links = links  # [(i,j,c), ...]
        self.__demands = demands  # {idx: (s,t,d), ...}
        self.__weights = None
        self.__waypoints = waypoints


    def solve(self) -> dict:
        """ set weights to inverse capacity and use shortest path algorithm """

        # add random waypoint for each demand
        t = time.process_time()
        pt_start = time.process_time()  # count process time (e.g. sleep excluded)

        # set link weights to inverse capacity scaled by max capacity
        max_c = max([c for _, _, c in self.__links])
        self.__weights = {(i, j): max_c / c for i, j, c in self.__links}

        post_processing = EqualSplitShortestPath(nodes=self.__nodes, links=self.__links, demands=self.__demands,
                                                 split=True, weights=self.__weights, waypoints=self.__waypoints)
        solution = post_processing.solve()

        pt_duration = time.process_time() - pt_start
        exe_time = time.process_time() - t

        # update execution time
        solution["execution_time"] = exe_time
        solution["process_time"] = pt_duration
        return solution

    def get_name(self):
        """ returns name of algorithm """
        return f"inverse_capacity"
