""" Factory for segment routing algorithms"""

from algorithm.generic_sr import GenericSR
from algorithm.segment_routing.demand_first_waypoints import DemandsFirstWaypoints
from algorithm.segment_routing.heur_ospf_weights import HeurOSPFWeights
from algorithm.segment_routing.inverse_capacity import InverseCapacity
from algorithm.segment_routing.segment_ilp import SegmentILP
from algorithm.segment_routing.sequential_combination import SequentialCombination
from algorithm.segment_routing.uniform_weights import UniformWeights


def get_algorithm(algorithm_name: str, nodes: list, links: list, demands: list, weights=None, waypoints=None,
                  seed: float = 42, ilp_method: str = None, time_out: int = None, sf: int = 100) -> GenericSR:
    algorithm_name = algorithm_name.lower()
    if algorithm_name == "demand_first_waypoints":
        algorithm = DemandsFirstWaypoints(nodes, links, demands, weights, waypoints)
    elif algorithm_name == "heur_ospf_weights":
        algorithm = HeurOSPFWeights(nodes, links, demands, weights, waypoints, seed=seed, time_out=time_out)
    elif algorithm_name == "inverse_capacity":
        algorithm = InverseCapacity(nodes, links, demands, weights, waypoints, seed=seed)
    elif algorithm_name == "segment_ilp":
        algorithm = SegmentILP(nodes, links, demands, weights, waypoints, waypoint_count=1, method=ilp_method,
                               splitting_factor=sf, time_out=time_out)
    elif algorithm_name == "sequential_combination":
        algorithm = SequentialCombination(nodes, links, demands, weights, waypoints, seed=seed, time_out=time_out,
                                          first_algorithm="heur_ospf_weights", second_algorithm="demand_first_waypoints")
    elif algorithm_name == "uniform_weights":
        algorithm = UniformWeights(nodes, links, demands, weights, waypoints, seed=seed)
    else:
        err_msg = f"algorithm not found: {algorithm_name}"
        raise Exception(err_msg)
    return algorithm
