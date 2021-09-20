"""this algorithm is used as a subroutine for multiple algorithms - simply computing the shortest path """

import time

import networkx as nx

from algorithm.generic_sr import GenericSR
from algorithm.segment_routing.sr_utility import SRUtility


class EqualSplitShortestPath(GenericSR):
    def __init__(self, nodes: list, links: list, demands: list, weights: dict = None, waypoints: dict = None,
                 split: bool = True, **kwargs):
        super().__init__(nodes, links, demands, weights, waypoints)

        self.__nodes = nodes
        self.__links = links  # list with [(i,j,c)]
        if waypoints is not None:
            segmented_demands = SRUtility.get_segmented_demands(waypoints, demands)
            self.__demands = {idx: (s, t, d) for idx, (s, t, d) in enumerate(segmented_demands)}  # dict {idx:(s,t,d)}
            self.__segments = waypoints  # dict with {idx:(p,q)}
        else:
            self.__demands = {idx: (s, t, d) for idx, (s, t, d) in enumerate(demands)}
            self.__segments = {idx: [(p, q)] for idx, (p, q, _) in enumerate(demands)}

        self.__weights = weights if weights else {(i, j): 1 for i, j, _ in links}

        self.__split = split
        self.__all_shortest_paths_generators = dict()
        self.__all_shortest_paths = dict()
        self.__nx_graph = nx.DiGraph()
        self.__flow_sum = dict()

        self.__create_nx_graph()
        self.__init_flow_sum_map()
        return

    def __create_nx_graph(self):
        for i, j, c in self.__links:
            w = int(self.__weights[i, j]) * 100  # '*100' to reduce computing errors
            self.__nx_graph.add_edge(i, j, weight=w, capacity=c)
        return

    def __get_all_shortest_paths_generator(self):
        for s in self.__nx_graph.nodes:
            for t in self.__nx_graph.nodes:
                if s == t:
                    continue
                self.__all_shortest_paths_generators[s, t] = nx.all_shortest_paths(
                    self.__nx_graph, source=s, target=t, weight='weight')
        return

    def __init_flow_sum_map(self):
        for i, j, _ in self.__links:
            self.__flow_sum[(i, j)] = 0
        return

    def __add_demand_val_to_path(self, path: list, demand: float):
        for idx in range(len(path) - 1):
            i = path[idx]
            j = path[idx + 1]

            self.__flow_sum[(i, j)] += demand
        return

    def __add_demand_update_objective(self, src, dst, demand):
        if (src, dst) not in self.__all_shortest_paths:
            self.__all_shortest_paths[src, dst] = list(self.__all_shortest_paths_generators[src, dst])

        if self.__split:
            n_splits = len(self.__all_shortest_paths[src, dst])
            split_demand = demand / n_splits
            for shortest_path in self.__all_shortest_paths[src, dst]:
                self.__add_demand_val_to_path(shortest_path, split_demand)
        else:
            # take first shortest path if multiple
            shortest_path = self.__all_shortest_paths[src, dst][0]
            self.__add_demand_val_to_path(shortest_path, demand)
        return

    def solve(self) -> dict:
        """
        Computes shortest path routes and determines max link utilization and the utilization map
        on fixed weight settings. This algorithms supports postprocessing of several
        algorithms e.g. segment_lp_gurobi_relax
        """

        t_start = time.time()  # sys wide time
        pt_start = time.process_time()  # count process time (e.g. sleep excluded)
        self.__get_all_shortest_paths_generator()
        for idx in self.__demands:
            s, t, d = self.__demands[idx]
            self.__add_demand_update_objective(s, t, d)
        pt_duration = time.process_time() - pt_start
        t_duration = time.time() - t_start
        utilization = {(i, j): self.__flow_sum[i, j] / self.__nx_graph[i][j]["capacity"] for i, j, _ in
                       self.__links}
        solution = {
            "objective": max(utilization.values()),
            "execution_time": t_duration,
            "process_time": pt_duration,
            "waypoints": self.__segments,
            "weights": self.__weights,
            "loads": utilization,
        }
        return solution

    def get_name(self):
        """ returns name of algorithm """
        return f"equal_split_shortest_paths"
