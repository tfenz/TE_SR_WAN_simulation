from abc import abstractmethod


class GenericSR:
    def __init__(self, nodes: list, links: list, demands: list, weights: dict, waypoints: dict, **kwargs):
        """
        generic baseclass for Segment Routing (SR) algorithms
        :param nodes: list of node indices
        :param links: list of links with: [(u, v, capacity), ...]
        :param demands: list of demands with: [(src, dst, demand), ...]
        :param weights: (optional) weights as dict with: {(i,j):weight, ...}
        :param waypoints: (optional) waypoints as dict with: {idx:[(p,q), ...], ...}
        """
        assert type(nodes) is list, f"Error {self.get_name()}: nodes must be a list with [i, ...]"
        assert type(links) is list, f"Error {self.get_name()}: links must be a list with [(i, j, capacity), ...]"
        assert type(demands) is list, f"Error {self.get_name()}: demands must be a list with [(src, dst, demand), ...]"
        assert weights is None or type(
            weights) is dict, f"Error {self.get_name()}: weights must be dict with {{(i,j):weight, ...}}"
        assert waypoints is None or type(
            waypoints) is dict, f"Error {self.get_name()}: waypoints must be dict with {{idx:[(p,q), ...], ...}}"

    @abstractmethod
    def solve(self) -> dict:
        """
        :return solution = {
            "objective": float,
            "execution_time": float,
            "waypoints": dict,
            "weights": dict,
            "utilization": dict,
            # can contain algorithm specific values
        }
        """
        raise Exception("method not implemented")

    @abstractmethod
    def get_name(self) -> str:
        raise Exception("method not implemented")
