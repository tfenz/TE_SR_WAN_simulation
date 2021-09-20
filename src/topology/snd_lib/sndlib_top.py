import os
from xml.dom import minidom

from topology.generic_topology_provider import GenericTopologyProvider
from topology.snd_lib.file_mapping import file_map
from utility import utility


class SndLibTop(GenericTopologyProvider):
    @staticmethod
    def get_topology_names():
        """ returns names of all supported topologies """
        return list(file_map.keys())

    @staticmethod
    def __read_network_xml(topology_file_name) -> (list, int):
        node_map = dict()
        full_file_name = os.path.abspath(topology_file_name)
        abilene_xml = minidom.parse(full_file_name)
        node_list = abilene_xml.getElementsByTagName('node')
        edge_list = abilene_xml.getElementsByTagName('link')

        index = 0
        for node in node_list:
            name = node.getAttribute('id')
            node_map[name] = index
            index += 1

        links_map = dict()
        for edge in edge_list:
            src_name = edge.getElementsByTagName('source')[0].firstChild.data
            dst_name = edge.getElementsByTagName('target')[0].firstChild.data
            capacity = edge.getElementsByTagName('capacity')[0].firstChild.data
            i = node_map[src_name]
            j = node_map[dst_name]
            links_map[i, j] = float(capacity)
            links_map[j, i] = float(capacity)
        links = [(i, j, links_map[i, j]) for i, j in links_map]
        n = len(node_map)
        return links, n

    def get_topology(self, topology_name: str, default_capacity: float = 1, **kwargs) -> (list, int):
        topology_name = utility.get_base_name(topology_name).lower()
        assert topology_name in file_map, "topology not supported. \nchoose from:\n\t" + ', '.join(
            list(file_map.keys()))
        topology_file_name = os.path.join(utility.BASE_PATH_SNDLIB_TOPOLOGY, file_map[topology_name])
        links, n = self.__read_network_xml(topology_file_name)
        return links, n
