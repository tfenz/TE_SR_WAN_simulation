"""
    HeurOSPF proposed in
        Bernard Fortz and Mikkel Thorup. Internet traffic engineering by optimizing OSPF weights.
        In Proc. IEEE INFOCOM, volume 2, pages 519–528. IEEE, 2000. doi:10.1109/INFCOM.2000.
        832225.
"""

import random
import time

import networkit as nk
import numpy as np

from algorithm.generic_sr import GenericSR
from algorithm.segment_routing.sr_utility import SRUtility
from utility import utility


class HeurOSPFWeights(GenericSR):
    BIG_M = 10 ** 9

    def __init__(self, nodes: list, links: list, demands: list, weights: dict = None, waypoints: dict = None,
                 hashtable_size: int = 16, sec_hashtable_size_multiplier: int = 20, max_weight: int = 20,
                 iterations: int = 5000, perturb_it: int = 300, seed: float = 0, time_out: int = None,
                 limit_not_improved=2500, **kwargs):
        super().__init__(nodes, links, demands, weights, waypoints)

        self.__seed = seed
        np.random.seed(self.__seed)

        # topology info
        self.__capacities = self.__extract_capacity_dict(links)  # dict with {(u,v):c, ..}
        self.__links = list(self.__capacities.keys())  # list with [(u,v), ..]
        self.__n = len(nodes)
        self.__max_weight = max_weight  # possible values in the weights vector are in [0, 1,..., max_weight]

        # demand segmentation and aggregate to matrix
        # store all target nodes for Some pairs shortest path algorithm
        self.__waypoints = waypoints
        self.__demands, self.__targets = self.__preprocess_demand_segmentation(waypoints, demands)

        # initial weights
        self.__init_weights = weights

        # neighborhood search
        self.__iterations = iterations
        self.__perturb_it = perturb_it

        # hashtable
        self.__hash_collision_counter = 0
        self.__hash_misses = 0
        self.__l = hashtable_size
        self.__l2 = int(np.log2(len(links) * sec_hashtable_size_multiplier))
        self.__hashtable1 = None
        self.__hashtable2 = None

        # networKit graph and some pairs shortest path (SPSP) algorithm
        self.__g = None
        self.__spsp = None

        # for exit criteria (1) timeout; (2) # iterations of no improvement
        self.__start_time = None
        self.__timeout = time_out if time_out else utility.TIME_LIMIT - 10
        self.__limit_not_improved = limit_not_improved

        self.__init_global_hashtable()
        self.__init_secondary_hashtable()
        self.__init_graph()
        return

    @staticmethod
    def __preprocess_demand_segmentation(segments, demands):
        """ Prepares input (compatibility reasons) """
        targets = set()
        demands_prepared = demands
        demand_matrix = dict()
        if segments is not None:
            demands_prepared = SRUtility.get_segmented_demands(segments, demands)
            demands_prepared = [(s, t, d) for s, t, d in demands_prepared]
        for s, t, d in demands_prepared:
            targets.add(t)
            if (s, t) not in demand_matrix:
                demand_matrix[s, t] = 0
            demand_matrix[s, t] += d
        return demand_matrix, list(targets)

    @staticmethod
    def __extract_capacity_dict(links):
        """ Converts the list of link/capacities into a capacity dict (compatibility reasons)"""
        return {(u, v): c for u, v, c in links}

    @staticmethod
    def __get_link_cost(link_load):
        """ Return cost value of a single link load """
        if link_load >= 2:
            return int(50000 * link_load)
        if link_load >= 11 / 10:
            return int(5000 * link_load)
        if link_load >= 1:
            return int(500 * link_load)
        if link_load >= 9 / 10:
            return int(70 * link_load)
        if link_load >= 2 / 3:
            return int(10 * link_load)
        if link_load >= 1 / 3:
            return int(3 * link_load)
        else:
            return int(1 * link_load)

    def __hash(self, weights: tuple):
        """ Computes hashvalues of a weights vector """
        hash_val = hash(weights)
        h1 = hash_val % 2 ** self.__l
        h2 = hash_val % 2 ** self.__l2
        return h1, h2

    def __init_global_hashtable(self):
        """ Initializes global hash table used to avoid cycling and recomputation of known results """
        self.__hashtable1 = np.zeros((2 ** self.__l), dtype=bool)
        return

    def __init_secondary_hashtable(self):
        """ Initializes secondary hash table to (1) speed up and (2) diversify neighborhood search """
        self.__hashtable2 = np.zeros((2 ** self.__l2), dtype=bool)
        return

    def __get_random_weights(self):
        """ Maps links to randomly chosen weights in the range of [1/4 * max_weight, 3/4 * max_weight] """
        rnd_weights = np.random.randint(
            low=self.__max_weight / 4, high=self.__max_weight * 3 / 4, size=(len(self.__links),))
        random_weights_dict = dict(zip(self.__links, rnd_weights))
        return random_weights_dict

    def __init_graph(self):
        """ Create networKit graph, add weighted edges and create spsp (some pairs shortest path) object """
        self.__g = nk.Graph(weighted=True, directed=True, n=self.__n)
        for u, v in self.__links:
            self.__g.addEdge(u, v, 1)
        self.__spsp = nk.distance.SPSP(self.__g, sources=self.__targets)

    def __update_nkit_graph_weights(self, weights):
        """ Updates weight in networKit graph """
        for u, v, w in self.__g.iterEdgesWeights():
            # Note: the weights are reversed since we need the distance from all sources to a specific target
            if w != weights[v, u]:
                self.__g.setWeight(u, v, weights[v, u])
        return

    def __reset_secondary_hashtable(self):
        """ Sets all values in the hashtable to False; called after each successful iteration """
        self.__hashtable2[self.__hashtable2] = False
        return

    def __get_distances(self):
        """ Recomputes the shortest path for 'some' pairs """
        self.__spsp.run()
        return self.__spsp.getDistances()

    def __perturb(self, weights):
        """ Perturbs current solution to escape local minima """
        new_weights = weights.copy()
        n_samples = max(3, int(len(self.__links) * 0.1))
        inds = np.random.choice(len(self.__links), n_samples)
        rand_links = [self.__links[ind] for ind in inds]
        for u, v in rand_links:
            w_diff = self.__max_weight
            while new_weights[u, v] + w_diff > self.__max_weight or new_weights[u, v] + w_diff < 1:
                w_diff = random.randint(-2, 2)
            new_weights[u, v] += w_diff
        return new_weights

    def __get_neighbor(self, x: int, t_idx: int, distances: dict, weights: dict, loads: dict):
        """ Chooses random neighbor vector w_a' """
        weights = weights.copy()
        # choose theta (load threshold) at random
        theta = np.random.uniform(low=0.25, high=1)

        # retrieve neighbors of x
        neighbors = list(self.__g.iterNeighbors(x))

        # retrieve min w(Pi)
        min_w_pi = self.BIG_M
        for x_i in neighbors:
            distance_x_i = distances[t_idx][x_i]
            min_w_pi = min(min_w_pi, distance_x_i)

        # filter overloaded adjacent arcs
        candidates = list()
        min_rhs = self.__max_weight - 1
        for x_i in neighbors:
            if distances[t_idx][x_i] - min_w_pi > self.__max_weight:
                min_rhs = min(min_rhs, distances[t_idx][x_i] + weights[x, x_i])
                continue
            candidates.append(x_i)

        # compute w_star
        subset_b = candidates.copy()
        w_star = self.BIG_M
        while w_star > min_rhs:
            for x_i in subset_b:
                if 1 + distances[t_idx][x_i] > min_rhs:
                    subset_b.remove(x_i)

            if len(subset_b) == 0:
                return weights
            w_star = max(1 + distances[t_idx][x_i] for x_i in subset_b)

        # compute new neighbor weight vector
        for x_i in subset_b:
            new_weight = w_star - distances[t_idx][x_i]

            if loads[x, x_i] <= theta:
                weights[x, x_i] = new_weight
            else:
                # if link is overloaded link weight can only be increased
                weights[x, x_i] = max(weights[x, x_i], new_weight)

        return weights

    def __add_loads_for(self, t_idx, weights, demands, acc_flows, distances):
        """ Computes flow path from all sources to node t and returns the updated acc_flows dict"""
        current_flows = np.zeros((self.__n, self.__n), np.float)
        t = self.__targets[t_idx]
        A_out = {y: list() for y in range(self.__n)}
        A_in = {y: list() for y in range(self.__n)}
        reverse_indices = range(self.__n - 1, -1, -1)
        for x, y in weights:
            if weights[(x, y)] == distances[t_idx][x] - distances[t_idx][y]:
                A_out[x].append(y)
                A_in[y].append(x)

        y_map = dict(zip(reverse_indices, np.array(distances[t_idx]).argsort()))
        for y_idx in range(self.__n - 1):
            y = y_map[y_idx]
            d_yt = demands[y, t] if (y, t) in demands else 0
            acc_demand_to_t = d_yt + np.sum(current_flows[y])
            if acc_demand_to_t <= 0:
                continue
            l = acc_demand_to_t / len(A_out[y])
            for z in A_out[y]:
                current_flows[z][y] = l
                acc_flows[y, z] += l
        return acc_flows

    def __evaluate_cost(self, weights):
        """
        evaluates cost of weight setting
        :return: max link utilization, distances, link loads
        """
        acc_flows = {(i, j): 0 for i, j in self.__links}
        self.__update_nkit_graph_weights(weights)
        distances = self.__get_distances()
        for t_idx in range(len(self.__targets)):
            acc_flows = self.__add_loads_for(t_idx, weights, self.__demands, acc_flows, distances)
        loads = dict()
        cost = 0
        for u, v in self.__links:
            loads[u, v] = acc_flows[u, v] / self.__capacities[u, v]
            cost += self.__get_link_cost(loads[u, v])

        return cost, distances, loads

    def __explore_neighborhood(self, sample_size: int, c_weights, c_distances, c_loads):
        """ for a given weights vector find the best neighbor weights vector"""
        best_weights, best_cost, best_loads, best_distances = c_weights, self.BIG_M, c_loads, c_distances
        h1 = None
        for _ in range(sample_size):
            # choose src and destination
            t_idx = np.random.randint(len(self.__targets))
            x = self.__targets[t_idx]
            while x == self.__targets[t_idx]:
                x = np.random.randint(self.__n)

            # retrieve neighbor weights vector
            n_weights = self.__get_neighbor(x, t_idx, c_distances, c_weights, c_loads)

            # hash solution
            h1, h2 = self.__hash(tuple(n_weights.values()))
            if self.__hashtable1[h1]:
                self.__hash_collision_counter += 1
                continue
            if self.__hashtable2[h2]:
                self.__hash_collision_counter += 1
                continue
            self.__hash_misses += 1
            self.__hashtable2[h2] = True

            # evaluate cost and compare
            n_cost, n_distances, n_loads = self.__evaluate_cost(n_weights)
            if best_cost >= n_cost:
                best_weights, best_cost, best_loads, best_distances = n_weights, n_cost, n_loads, n_distances

        self.__hashtable1[h1] = True
        return best_weights, best_cost, best_loads, best_distances

    def __ospf_heuristic(self):
        """ main procedure """
        # evaluate initial weights
        weights = self.__init_weights if self.__init_weights else self.__get_random_weights()
        cost, distances, loads = self.__evaluate_cost(weights)
        # bc_ := best cost
        # bu_ := best (= lowest) max link utilization
        bc_cost = bu_cost = cost
        bc_util = bu_util = self.BIG_M
        bc_weights = bu_weights = weights
        bc_loads = bu_loads = loads

        # pr_cost/pr_max_util stores results from last neighborhood search iteration
        pr_cost = pr_util = self.BIG_M

        # initially 20% of the neighborhood size gets evaluated
        neighborhood_size = len(self.__targets) * (self.__n - 1)
        sample_factor = 0.2

        count_not_better_as_pr = 0  # counts worse than previous
        count_not_improved_best = 0  # counts worse than best

        it = 0
        exit_reason = "max iterations reached"
        for it in range(self.__iterations):
            # explore neighborhood
            sample_size = max(int(neighborhood_size * sample_factor), 5)  # max(..,5) is for too small topologies
            weights, cost, loads, distances = self.__explore_neighborhood(sample_size, weights, distances, loads)
            util = max(loads.values())

            # exit criteria (1) timeout
            if self.__timeout < time.time() - self.__start_time:
                exit_reason = "time out"
                break

            # exit criteria (2) not improved best for a very long time
            if not (bc_cost > cost or bu_util > util):
                count_not_improved_best += 1
                if count_not_improved_best > self.__limit_not_improved:
                    exit_reason = f"LIMIT NOT IMPROVED exceeded {self.__limit_not_improved}"
                    break
            else:
                count_not_improved_best = 0

            # keep best solution data
            if bc_cost >= cost and bu_util >= util:
                bc_weights, bc_cost, bc_loads, bc_util, bc_distances = weights, cost, loads, util, distances
                bu_weights, bu_cost, bu_loads, bu_util, bu_distances = weights, cost, loads, util, distances
            elif bc_cost >= cost:
                bc_weights, bc_cost, bc_loads, bc_util, bc_distances = weights, cost, loads, util, distances
            elif bu_util >= util:
                bu_weights, bu_cost, bu_loads, bu_util, bu_distances = weights, cost, loads, util, distances
            # better than previous solution?
            if pr_cost > cost or pr_util > util:
                sample_factor = max(0.01, sample_factor / 3)
                count_not_better_as_pr = 0
                self.__reset_secondary_hashtable()
            else:
                sample_factor = min(1, sample_factor * 10)
                count_not_better_as_pr += 1
                if count_not_better_as_pr >= self.__perturb_it:
                    weights = self.__perturb(weights)
                    count_not_better_as_pr = 0
                    self.__reset_secondary_hashtable()

            pr_cost, pr_util = cost, util
        return bc_weights, bc_cost, bc_loads, bc_util, bu_weights, bu_cost, bu_loads, bu_util, it, exit_reason

    def solve(self) -> dict:
        """ compute solution """

        self.__start_time = t_start = time.time()  # sys wide time
        pt_start = time.process_time()  # count process time (e.g. sleep excluded and count per core)
        bc_weights, bc_cost, bc_loads, bc_util, bu_weights, bu_cost, bu_loads, bu_util, number_iterations, exit_reason = self.__ospf_heuristic()
        pt_duration = time.process_time() - pt_start
        t_duration = time.time() - t_start

        solution = dict()
        # best max utilization result
        solution["objective"] = bu_util
        solution["execution_time"] = t_duration
        solution["process_time"] = pt_duration
        solution["waypoints"] = self.__waypoints
        solution["weights"] = bu_weights
        solution["loads"] = bu_loads
        solution["cost"] = bu_cost

        # bc := best cost result
        solution["bc_objective"] = bc_util
        solution["bc_weights"] = bc_weights
        solution["bc_cost"] = bc_cost
        solution["bc_loads"] = bc_loads

        solution["used_iterations"] = number_iterations
        solution["exit_reason"] = exit_reason

        # parameters
        solution["max_iterations"] = self.__iterations
        solution["max_weight"] = self.__max_weight
        solution["perturb_it"] = self.__perturb_it
        solution["seed"] = self.__seed
        solution["hash_table_l1"] = self.__l
        solution["hash_table_l2"] = self.__l2
        return solution

    def get_name(self):
        """ returns name of algorithm """
        return f"heur_ospf_weights"
