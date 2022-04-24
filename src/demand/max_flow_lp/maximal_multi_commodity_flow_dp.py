""" Max flow demand generator; randomly choose src/dst pairs and compute MCF LP to determine maximal flow sizes;"""

import gurobipy as gb
import numpy as np

from demand.generic_demand_provider import GenericDemandProvider
from utility import utility


class McfDP(GenericDemandProvider):
    def __init__(self, n: int, links: list, seed: float = 0, fixed_total: float = 1, number_samples: int = 10,
                 active_pairs_fraction=0.5, flows_per_pair: int = 3, method: str = "MAXIMAL",
                 unscaled_dm_sets: dict = None, **kwargs):
        """
        creates a traffic matrix by choosing a fraction of connection pairs at random and assign traffic
            which magnitude is determined using optimal routing algorithm
        :param n: # nodes
        :param links: the links including capacity information with [(i, j, capacity), ..]
        :param seed: seed for generating random numbers
        :param fixed_total: the sum of all gets scaled to fixed_total
        :param number_samples: number of samples
        :param active_pairs_fraction: define how many src/dst pairs have non-zero demand
        :param flows_per_pair: number of flows per src/dst
        """
        methods = ["MAXIMAL", "MAXIMAL_CONCURRENT", "UNIFORM_MAXIMAL_CONCURRENT"]
        assert method.upper() in methods, f"method must be in: {methods}"
        assert method.upper() != "MAXIMAL_CONCURRENT" or method.upper() == "MAXIMAL_CONCURRENT" and unscaled_dm_sets is not None, "for method MAXIMAL_CONCURRENT: unscaled_demands must not be None"

        self.__method = method.upper()
        self.__seed = seed
        self.__n = n
        self.__n_active_pairs = int(self.__n * (self.__n - 1) * active_pairs_fraction)
        self.__fixed_total = fixed_total
        self.__n_samples = number_samples
        if method.upper() == "MAXIMAL_CONCURRENT":
            self.__n_samples = len(unscaled_dm_sets)

        self.__flows_per_pair = flows_per_pair
        self.__capacities = {(i, j): c for i, j, c in links}
        self.__links = list(self.__capacities.keys())
        self.__unscaled_dms = unscaled_dm_sets

        # random src/dst pairs:
        self.__connection_pairs = None
        self.__current_unscaled_dm = None

        # Gurobi variables:
        self.__model = None
        self.__flows = None
        self.__demands = None
        self.__fraction = None

        # set of results
        self.__demand_sequence_sets = dict()
        self.__demand_matrix_sets = dict()
        return

    def __get_random_connection_pair(self):
        """ Chooses a random connection pair (s, t) """
        s = np.random.randint(0, self.__n)
        t = s
        while s == t:
            t = np.random.randint(0, self.__n)
        return s, t

    def __choose_connection_pairs(self):
        """ Assigns list of random pairs to self.__connection_pairs"""
        self.__connection_pairs = set()
        while len(self.__connection_pairs) < self.__n_active_pairs:
            self.__connection_pairs.add(self.__get_random_connection_pair())
        return

    def __gp_model(self):
        """ Creates a named Gurobi model for lps with some options"""
        self.__model = gb.Model(self.get_name().replace("-", "_").replace(" ", "_"))
        self.__model.setParam('LogToConsole', utility.LOGTOCONSOLE_DP)
        self.__model.setParam('TimeLimit', utility.TIME_LIMIT_DP)
        self.__model.setParam('threads', utility.MAX_THREADS_DP)
        return

    def __gp_objective_maximal(self):
        """ Sets the objective of the model: maximize the sum of demands for a given set of src/dst pairs """
        self.__model.setObjective(gb.quicksum(self.__demands[s, t] for s, t in self.__connection_pairs),
                                  gb.GRB.MAXIMIZE)
        return

    def __gp_objective_scale(self):
        """ Sets the objective of the model: maximize the sum of demands for a given set of src/dst pairs """
        self.__model.setObjective(gb.quicksum(self.__demands[s, t] for s, t in self.__connection_pairs),
                                  gb.GRB.MAXIMIZE)
        return

    def __gp__scale_demands(self):
        """ Adds Capacity Constraint to model: Sum of flow on a link must not exceed the links capacity """
        self.__model.addConstrs(
            self.__demands[s, t] == self.__current_unscaled_dm[s, t] * self.__scale for s, t in self.__connection_pairs)
        return

    def __gp_fix_scale(self):
        self.__model.addConstr(self.__scale == 1, "fixed_scale")

    def __gp_vars(self):
        """ Creates all gurobi decision variables """
        self.__demands = self.__model.addVars(self.__connection_pairs, lb=0, ub=gb.GRB.INFINITY,
                                              vtype=gb.GRB.CONTINUOUS)
        self.__scale = self.__model.addVar(lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS)
        self.__flows = self.__model.addVars(self.__connection_pairs, self.__links, lb=0, ub=gb.GRB.INFINITY,
                                            vtype=gb.GRB.CONTINUOUS)
        return

    def __gp_flow_conservation(self):
        """
        Adds Flow Conservation Constraint to model:
            incoming flows must equal outgoing except for source and destinations
        """
        # (5-i) i != p,q
        self.__model.addConstrs(
            self.__flows.sum(s, t, '*', v) - self.__flows.sum(s, t, v, '*') == 0 for v in range(self.__n) for s, t in
            self.__connection_pairs if v != s and v != t)

        # (5-ii) s == i
        self.__model.addConstrs(
            self.__flows.sum(s, t, '*', v) - self.__flows.sum(s, t, v, '*') == -self.__demands[s, t] for
            v in range(self.__n) for s, t in self.__connection_pairs if v == s)

        # (5-iii) t == i
        self.__model.addConstrs(
            self.__flows.sum(s, t, '*', v) - self.__flows.sum(s, t, v, '*') == self.__demands[s, t] for
            v in range(self.__n) for s, t in self.__connection_pairs if v == t)
        return

    def __gp_capacities(self):
        """ Adds Capacity Constraint to model: Sum of flow on a link must not exceed the links capacity """
        self.__model.addConstrs(self.__flows.sum('*', '*', u, v) <= self.__capacities[u, v] for u, v in self.__links)

    def __setup_constraints(self, sample):
        """ setup Constraints for MCF LP"""
        self.__gp_model()
        if self.__method == "MAXIMAL":
            self.__choose_connection_pairs()
            self.__gp_vars()
            self.__gp_fix_scale()
        elif self.__method == "MAXIMAL_CONCURRENT":
            self.__current_unscaled_dm = {(s, t): 0 for s, t in self.__unscaled_dms[sample]}
            for (s, t), d in self.__unscaled_dms[sample].items():
                self.__current_unscaled_dm[s, t] += d
            self.__connection_pairs = list(self.__current_unscaled_dm.keys())
            self.__gp_vars()
            self.__gp__scale_demands()
        elif self.__method == "UNIFORM_MAXIMAL_CONCURRENT":
            self.__choose_connection_pairs()
            self.__current_unscaled_dm = {(s, t): 1 for s, t in self.__connection_pairs}
            self.__gp_vars()
            self.__gp__scale_demands()
        else:
            raise Exception(f"method: {self.__method} not supported")

        self.__gp_capacities()
        self.__gp_flow_conservation()
        self.__gp_objective_maximal()

    def __compute_dm(self, sample):
        """starts LP optimizer and extracts result"""
        self.__setup_constraints(sample)
        self.__model.optimize()
        dm = {(s, t): self.__demands[s, t].X for s, t in self.__demands}
        return dm

    def demand_matrix(self, sample: int) -> dict:
        """ Get a single demand matrix """
        assert 0 <= sample < self.__n_samples, "sample nr out of range"
        np.random.seed(self.__seed + sample)
        if sample not in self.__demand_matrix_sets:
            self.__demand_matrix_sets[sample] = self.__compute_dm(sample)
        return self.__demand_matrix_sets[sample]

    def demand_sequence(self, sample: int) -> list:
        """ Get a single demand sequence """
        np.random.seed(self.__seed + sample)
        if sample not in self.__demand_sequence_sets:
            dm = self.demand_matrix(sample)
            self.__demand_sequence_sets[sample] = list()
            for s, t in dm:
                for _ in range(self.__flows_per_pair):
                    self.__demand_sequence_sets[sample].append((s, t, dm[s, t] / self.__flows_per_pair))
        return self.__demand_sequence_sets[sample]

    def demand_matrices(self) -> list:
        """ Generator object to get all sample demand matrices """
        for sample in range(self.__n_samples):
            try:
                yield self.demand_matrix(sample)
            except:
                continue

    def demand_sequences(self) -> list:
        """ Generator object to get all sample demand sequences """
        for sample in range(self.__n_samples):
            try:
                yield self.demand_sequence(sample)
            except gb.GurobiError as ex:
                raise ex
            except Exception as ex:
                continue

    def __len__(self):
        """ len is defined by the number of samples """
        return self.__n_samples

    def __str__(self):
        self.get_name()

    def get_name(self) -> str:
        return f"LP_MCF_{self.__method.title()}_DP_seed_{self.__seed}"
