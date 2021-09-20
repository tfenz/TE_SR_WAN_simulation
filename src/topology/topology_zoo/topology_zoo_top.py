import os

import networkx as nx

from topology.generic_topology_provider import GenericTopologyProvider
from topology.topology_zoo.file_mapping import file_map
from utility import utility


class TopologyZoo(GenericTopologyProvider):
    @staticmethod
    def get_topology_names():
        """ returns names of all supported topologies """
        return list(file_map.keys())

    def get_topology(self, topology_name: str, default_capacity: float = 1, **kwargs) -> (list, int):
        topology_name = utility.get_base_name(topology_name).lower()
        assert topology_name in file_map, "topology not supported. \nchoose from:\n\t" + ', '.join(
            list(file_map.keys()))

        topology_file_name = os.path.abspath(os.path.join(utility.BASE_PATH_ZOO_TOPOLOGY, file_map[topology_name]))

        if not os.path.exists(topology_file_name):
            msg = f"topology file not found: {topology_file_name}"
            raise Exception(msg)
        nx_graph = nx.read_graphml(topology_file_name, node_type=int)

        n = len(nx_graph)

        links = list()
        links_map = dict()
        for edge in nx_graph.edges(data=True):
            i = edge[0]
            j = edge[1]
            c = default_capacity
            if "LinkSpeedRaw" in edge[2]:
                c = edge[2]["LinkSpeedRaw"]
            links_map[i, j] = float(c)
            links_map[j, i] = float(c)
            links = [(i, j, links_map[i, j]) for i, j in links_map]
        return links, n
