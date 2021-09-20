""" used for console output highlighting only """
import os

# for console output highlighting
HIGHLIGHT = '\33[33m\33[1m'
FAIL = '\033[91m'
CEND = '\033[0m'

# Gurobi consts for SR Routing ILP
# TIME_LIMIT = 60 * 60 * 10  # in sec
TIME_LIMIT = 60 * 60 * 4  # in sec
MAX_THREADS = 10
LOGTOCONSOLE = 0  # 0:= no logging to console
NON_CONVEX = 2

# Gurobi consts for demand provider MCF LP
TIME_LIMIT_DP = 60 * 5  # in sec
MAX_THREADS_DP = 4
LOGTOCONSOLE_DP = 0
NON_CONVEX_DP = 0

# DATA PATHS
BASE_PATH_SNDLIB_DEMANDS = os.path.abspath("../data/demands/sndlib/")
BASE_PATH_SNDLIB_TOPOLOGY = os.path.abspath("../data/topologies/sndlib/")
BASE_PATH_ZOO_TOPOLOGY = os.path.abspath("../data/topologies/topology_zoo/")


def create_dirs(path: str):
    """ make dirs """
    try:
        out_dir = os.path.abspath(path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_dir = os.path.abspath(out_dir)
    except Exception as ex:
        print(f"Fatal Error: create_out_dirs(...) throws: {str(ex)} ")
        raise ex
    return out_dir


def get_base_name(full_name):
    """extract base name without extension from filename"""
    base_name = os.path.basename(full_name)
    name, ext = os.path.splitext(base_name)
    return name


def error_solution():
    return {
        "objective": -1,
        "execution_time": -1,
        "waypoints": {},
        "weights": {},
        "utilization": {}
    }


def get_fpp(links, divisor: int = 4, min_fpp: int = 3):
    """ flows per pair is dependent on the size of the links"""
    flows_per_pair = max(int(len(links) / divisor), min_fpp)
    return flows_per_pair


def get_setup_dict(algorithm, demands, demand_provider_name, links, method, n, sample_idx, test_idx, topology_name,
                   topology_provider_name, active_pairs_fraction, mcf_method, seed):
    """ stores all setup information of a single test """
    setup = dict()
    setup["test_idx"] = test_idx
    setup["topology_provider"] = topology_provider_name
    setup["topology_name"] = topology_name
    setup["#nodes"] = n
    setup["#links"] = len(links)
    setup["provider"] = demand_provider_name
    setup["#demands"] = len(demands)
    setup["active_pairs_fraction"] = active_pairs_fraction
    setup["flows_per_pair"] = get_fpp(links)
    setup["mcf_method"] = mcf_method
    setup["seed"] = seed
    setup["sample_idx"] = sample_idx
    setup["ilp_method"] = method
    setup["algorithm"] = algorithm
    return setup
