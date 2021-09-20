from algorithm.generic_sr import GenericSR


class SequentialCombination(GenericSR):
    def __init__(self, nodes: list, links: list, demands: list, weights: dict = None, waypoints: dict = None,
                 first_algorithm: str = "", second_algorithm: str = "", seed=0, **kwargs):
        super().__init__(nodes, links, demands, weights, waypoints)
        assert first_algorithm and first_algorithm != "", "First algorithm must be defined"
        assert second_algorithm and second_algorithm != "", "Second algorithm must be defined"

        self.__nodes = nodes  # [i, ..., n-1]
        self.__links = links  # [(i, j, capacity), ...]
        self.__demands = demands  # [(src, dst, demand), ...]
        self.__first_algorithm = first_algorithm  # name of first algorithm (weights and/or waypoints are input for the second algorithm)
        self.__second_algorithm = second_algorithm  # name of second algorithm
        self.__seed = seed

    def solve(self) -> dict:
        """
        sequential combination of two arbitrary sr algorithms to compute
        first optimal weight setting and then waypoints or vice versa
        """

        solution = dict()

        from algorithm.sr_factory import get_algorithm
        # route on shortest paths
        first = get_algorithm(algorithm_name=self.__first_algorithm, demands=self.__demands, nodes=self.__nodes,
                              links=self.__links)
        solution_first = first.solve()
        solution.update({f"{self.__first_algorithm}_{k}": v for k, v in solution_first.items()})
        weights = solution_first['weights']
        waypoints = solution_first['waypoints']

        second = get_algorithm(algorithm_name=self.__second_algorithm, demands=self.__demands, nodes=self.__nodes,
                               links=self.__links, weights=weights, waypoints=waypoints)
        solution_second = second.solve()
        solution.update({f"{self.__second_algorithm}_{k}": v for k, v in solution_second.items()})

        solution["first_algorithm"] = self.__first_algorithm
        solution["second_algorithm"] = self.__second_algorithm

        solution["execution_time"] = solution_first["execution_time"] + solution_second["execution_time"]
        solution["process_time"] = solution_first["process_time"] + solution_second["process_time"]
        solution["objective"] = solution_second["objective"]
        solution["waypoints"] = solution_second["waypoints"]
        solution["weights"] = solution_second["weights"]
        solution["loads"] = solution_second["loads"]
        return solution

    def get_name(self):
        """ returns name of algorithm """
        return f"sequential_combination"
