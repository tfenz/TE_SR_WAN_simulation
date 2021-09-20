import os
import time

import gurobipy as gp

from algorithm.generic_sr import GenericSR
from utility import utility


class SegmentILP(GenericSR):
    def __init__(self, nodes: list, links: list, demands: list, weights: dict = None, waypoints: dict = None,
                 waypoint_count: int = 1, method: str = "JOINT", splitting_factor: int = 15,
                 model_name: str = "sr_ilp", max_weight=None, log_file_name: str = "gurobi.log", time_out=None,
                 log_to_console=None, threads=None, **kwargs):
        super().__init__(nodes, links, demands, weights, waypoints)
        self.__is_build = False

        assert type(waypoint_count) is int and waypoint_count > 0, "waypoint_count must be greater than 0"
        assert method.upper() in ["JOINT", "WAYPOINTS", "WEIGHTS"], "must be from ['JOINT', 'WAYPOINTS', 'WEIGHTS']"
        assert type(splitting_factor) is int and splitting_factor > 0, "splitting factor must be greater than 0"

        self.__time_out = time_out if time_out else utility.TIME_LIMIT
        self.__threads = threads if threads else utility.MAX_THREADS
        self.__log_to_console = log_to_console if log_to_console else utility.LOGTOCONSOLE
        self.__nodes = nodes  # [i, ...]
        self.__links = links  # [(i,j,c), ...]
        self.__demands = {idx: (s, t, d) for idx, (s, t, d) in enumerate(demands)}  # dict with: {idx:(s, t, d), ...}
        self.__waypoint_count = waypoint_count  # allowed number of waypoints per demand
        self.__max_weight = max_weight if max_weight else gp.GRB.INFINITY  # Note: discrete set of possible weights in [1, 2, .., max_weight]

        # Choose from [JOINT, WAYPOINTS, WEIGHTS]
        #   WAYPOINTS: optimize routes only by adding waypoints (constant link weights)
        #   WEIGHTS: optimize routes by adjusting link weights (no waypoints can be set)
        #   JOINT: optimize routes by adding waypoints and adjusting link weights
        self.__method = method.upper()

        # Splitting factor for constraint (10) (last revision of MIP from 2021-03-10)
        #   restricts the number of parts a flow to destination t can split. (splitting_factor=1: for single flow)
        self.__splitting_factor = splitting_factor
        self.__name = model_name
        self.__log_file_name = os.path.abspath(log_file_name)

        # See ILP definition for M constant - ILP (1)
        self.__M = max(sum(d for s, t, d in demands), 2 * len(links), 100)
        self.__segments = [(u, v) for u in self.__nodes for v in self.__nodes if u != v]  # ILP (2)

        # Gurobi variables:
        self.__model = None
        self.__max_util = None
        self.__utilization = None  # helper var (no influence in objective)
        self.__d_segments = None
        self.__f_link = None
        self.__segments_flows = None
        self.__x = None
        self.__f_segment = None
        self.__distance = None
        self.__w = None

        self.setup_constraints()
        return

    def __gp_model(self):
        """creates a named Gurobi model for lps with some options"""
        self.__model = gp.Model(self.__name)
        self.__model.setParam('LogToConsole', self.__log_to_console)
        self.__model.setParam('TimeLimit', self.__time_out)
        self.__model.setParam('threads', self.__threads)
        self.__model.setParam('LogFile', self.__log_file_name)
        return

    def __gp_vars(self):
        links_only = [(u, v) for u, v, c in self.__links]

        # util := objective (= minimize maximal link utilisation)
        self.__max_util = self.__model.addVar(lb=0, ub=self.__M, vtype=gp.GRB.CONTINUOUS)  # ok
        self.__utilization = self.__model.addVars(links_only, lb=0, ub=self.__M, vtype=gp.GRB.CONTINUOUS)  # helper var
        # D := sum of demands on segment (p,q)
        self.__d_segments = self.__model.addVars(self.__segments, lb=0, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
        # f := the fractional amount of flow for segment l (p, q) on the link l.
        self.__f_link = self.__model.addVars(
            self.__segments, links_only, lb=0, ub=self.__M, vtype=gp.GRB.CONTINUOUS)  # ok
        # S := binary variable indicating whether segment (p, q) is active for demand i
        self.__segments_flows = self.__model.addVars(self.__segments, self.__demands.keys(), vtype=gp.GRB.BINARY)  # ok
        # x := binary variables indicating whether link l is on a shortest path to node t.
        self.__x = self.__model.addVars(self.__nodes, links_only, vtype=gp.GRB.BINARY)  # ok
        # f_seg := the total flow of segment (p, q) leaving a node
        self.__f_segment = self.__model.addVars(self.__segments, lb=0, ub=self.__M, vtype=gp.GRB.CONTINUOUS)  # ok
        # shortest path distance from v to t.
        self.__distance = self.__model.addVars(
            self.__nodes, self.__nodes, lb=0, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)  # ok
        # weight of link l.
        self.__w = self.__model.addVars(links_only, lb=1, ub=self.__max_weight,
                                        vtype=gp.GRB.INTEGER)  # todo change to continuous if performance suffers
        return

    def __gp_objective(self):
        self.__model.setObjective(self.__max_util, gp.GRB.MINIMIZE)

    def __gb_c_demands_segments(self):
        """ This method sets the demands on segments constraints - ILP: (4) """
        self.__model.addConstrs(self.__d_segments[p, q] == gp.quicksum(
            self.__segments_flows[p, q, i] * d for i, (s, t, d) in self.__demands.items()) for p, q in self.__segments)
        return

    def __gp_c_flows(self):
        """ This method sets the flow conservation - ILP: (5)"""
        # (5-i) i != p,q
        self.__model.addConstrs(
            self.__f_link.sum(p, q, '*', v) - self.__f_link.sum(p, q, v, '*') == 0 for v in self.__nodes for p, q in
            self.__segments if v != p and v != q)

        # (5-ii) s == i
        self.__model.addConstrs(
            self.__f_link.sum(p, q, '*', v) - self.__f_link.sum(p, q, v, '*') == -self.__d_segments[p, q] for
            v in self.__nodes for p, q in self.__segments if v == p)

        # (5-iii) t == i
        self.__model.addConstrs(
            self.__f_link.sum(p, q, '*', v) - self.__f_link.sum(p, q, v, '*') == self.__d_segments[p, q] for
            v in self.__nodes for p, q in self.__segments if v == q)
        return

    def __gp_c_segments_paths(self):
        """ This method sets the segment paths - ILP: (6) and (7)"""
        # (6-i) v != s,t
        self.__model.addConstrs(
            self.__segments_flows.sum('*', v, i) - self.__segments_flows.sum(v, '*', i) == 0 for v in self.__nodes for
            i, (s, t, _) in self.__demands.items() if v != s and v != t)

        # (6-ii) v == s
        self.__model.addConstrs(
            self.__segments_flows.sum('*', v, i) - self.__segments_flows.sum(v, '*', i) == -1 for v in self.__nodes for
            i, (s, t, _) in self.__demands.items() if v == s)

        # (6-iii) v == t
        self.__model.addConstrs(
            self.__segments_flows.sum('*', v, i) - self.__segments_flows.sum(v, '*', i) == 1 for v in self.__nodes for
            i, (s, t, _) in self.__demands.items() if v == t)

        # (7)
        self.__model.addConstrs(
            self.__segments_flows.sum('*', '*', i) <= self.__waypoint_count + 1 for i in self.__demands)
        return

    def __gp_c_fix_segments(self):
        """ (optional) This method adds constraints that all segments are fixed from s to t.
        No additional waypoints are allowed. Used for WEIGHTS optimization only - ILP (not mentioned yet)"""
        self.__model.addConstrs(self.__segments_flows[s, t, i] == 1 for i, (s, t, _) in self.__demands.items())
        return

    def __gp_c_capacity(self):
        """ sets the capacity constraints - ILP (8)"""
        self.__model.addConstrs(self.__f_link.sum('*', '*', i, j) <= self.__max_util * c for i, j, c in self.__links)
        self.__model.addConstrs(
            self.__utilization[i, j] == self.__f_link.sum('*', '*', i, j) / c for i, j, c in self.__links)
        return

    def __gp_c_shortest_path_tree(self):
        """sets the shortest path constraint - ILP (9)"""
        self.__model.addConstrs(
            self.__f_link[p, q, i, j] <= self.__M * self.__x[q, i, j] for p, q in self.__segments for i, j, _ in
            self.__links)
        return

    def __gp_c_set_splitting_factor(self):
        """restricts the number of paths a flow can take - ILP (10)"""
        self.__model.addConstrs(
            self.__x.sum(t, v, '*') <= self.__splitting_factor for v in self.__nodes for t in self.__nodes if t != v)
        return

    def __gp_c_even_split(self):
        """sets the even split (between links) constraints - ILP (11)"""
        # (11-i)
        self.__model.addConstrs(self.__f_link.sum('*', t, i, j) <= self.__f_segment[t, i]
                                for v, t in self.__segments for i, j, _ in self.__links if i != t)
        # (11-ii)
        self.__model.addConstrs(
            self.__f_segment[t, i] - self.__f_link.sum('*', t, i, j) <= self.__M * (1 - self.__x[t, i, j])
            for v, t in self.__segments for i, j, _ in self.__links if i != t)
        return

    def __gp_c_weights(self):
        """sets the weight constraints - ILP (12)"""

        # ILP (12-i)
        self.__model.addConstrs(
            self.__distance[u, t] <= self.__distance[v, t] + self.__w[u, v] for u, v, _ in self.__links for t in
            self.__nodes if t != u)

        # ILP (12-ii)
        self.__model.addConstrs(
            self.__distance[v, t] - self.__distance[u, t] + self.__w[u, v] <= self.__M * (1 - self.__x[t, u, v]) for
            u, v, _ in self.__links for t in self.__nodes if t != u)

        # ILP (12-iii)
        self.__model.addConstrs(
            1 - self.__x[t, u, v] <= self.__M * (self.__distance[v, t] - self.__distance[u, t] + self.__w[u, v]) for
            u, v, _ in self.__links for t in self.__nodes if t != u)
        return

    def __gp_c_fix_weights(self):
        """ (optional) adds constraints that fix all link weights at 1. Used for WAYPOINTS optimization only
            - ILP (not mentioned yet)"""
        self.__model.addConstrs(self.__w[i, j] == 1 for i, j, c in self.__links)
        return

    def setup_constraints(self):
        self.__gp_model()
        self.__gp_vars()  # ILP (1), (2), (3)
        self.__gp_objective()

        self.__gb_c_demands_segments()  # ILP (4)
        self.__gp_c_flows()  # ILP (5)
        self.__gp_c_segments_paths()  # ILP (6), (7)
        if self.__method == "WEIGHTS":
            self.__gp_c_fix_segments()  # ILP (not mentioned in formulation)
        self.__gp_c_capacity()  # ILP (8)
        self.__gp_c_shortest_path_tree()  # ILP (9)
        self.__gp_c_set_splitting_factor()  # ILP (10)
        self.__gp_c_even_split()  # ILP (11)
        self.__gp_c_weights()  # ILP (12)
        if self.__method == "WAYPOINTS":  # ILP (not mentioned in formulation)
            self.__gp_c_fix_weights()

    def solve(self) -> dict:
        """ Solves the MIP
        :return: dict with execution time, objective, segments and weight-assignment
        """

        t_start = time.time()  # sys wide time
        pt_start = time.process_time()  # count process time (e.g. sleep excluded)
        self.__model.optimize()
        objective = self.__model.objVal
        pt_duration = time.process_time() - pt_start
        t_duration = time.time() - t_start

        waypoints = {idx: [(p, q) for p, q, idx_s in self.__segments_flows if
                           idx_s == idx and self.__segments_flows[p, q, idx_s].X == 1] for idx in self.__demands}
        weights = {(u, v): self.__w[u, v].X for u, v in self.__w}
        loads = {(u, v): self.__utilization[u, v].X for u, v in self.__utilization}
        solution = {
            "objective": objective,
            "execution_time": t_duration,
            "process_time": pt_duration,
            "waypoints": waypoints,
            "weights": weights,
            "loads": loads,
        }
        return solution

    def get_name(self):
        """ returns name of algorithm """
        return f"segment_ilp"
