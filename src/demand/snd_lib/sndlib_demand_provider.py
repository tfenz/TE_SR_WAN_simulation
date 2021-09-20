""" Reads demand matrix files stored in the form defined by snd_lib (http://sndlib.zib.de/html/docu/io-formats/xml/index.html) """

import os
from xml.dom import minidom

import numpy as np

from demand.generic_demand_provider import GenericDemandProvider
from demand.snd_lib.directory_mapping import directory_map
from utility import utility


class SndLibTM(GenericDemandProvider):
    def __init__(self, topology_name: str, number_samples: int, fixed_total: float = None, flows_per_pair: int = 3,
                 **kwargs):
        topology_name = utility.get_base_name(topology_name).lower()
        assert topology_name in directory_map, f"topology not supported. \nchoose from:\n\t" + '\n\t'.join(
            list(directory_map.keys()))

        self.__flows_per_pair = flows_per_pair
        self.__topology_name = topology_name
        self.__n_samples = number_samples
        self.__fixed_total = fixed_total

        demand_dir = os.path.abspath(os.path.join(utility.BASE_PATH_SNDLIB_DEMANDS, directory_map[topology_name]))
        all_files = [os.path.abspath(os.path.join(demand_dir, f)) for f in os.listdir(demand_dir) if
                     os.path.isfile(os.path.join(demand_dir, f))]
        self.__files = list()

        for i in range(number_samples):
            f_index = i * (len(all_files) // number_samples)
            self.__files.append(all_files[f_index])

        # set of results
        self.__demand_matrix_sets = dict()
        self.__demand_sequence_sets = dict()
        return

    def __get_scaled_tm(self, tm):
        """ Scales demands so that the sum of all demands in the TM equals self.__fixed_total"""
        total_demand = np.sum(np.array(list(tm.values())))
        if not total_demand:
            raise Exception(f"unexpected error while scaling the tm")
        dm = self.__fixed_total / total_demand if self.__fixed_total else 1
        tm.update((pair, demand * dm) for pair, demand in tm.items())
        return tm

    def __read_demand_xml(self, idx: int) -> dict:
        abilene_xml = minidom.parse(str(self.__files[idx]))

        node_map = dict()
        node_list = abilene_xml.getElementsByTagName('node')

        index = 0
        for node in node_list:
            name = node.getAttribute('id')
            node_map[name] = index
            index += 1

        demand_items = abilene_xml.getElementsByTagName('demand')
        demand_matrix = dict()
        for demand_item in demand_items:
            src_name = demand_item.getElementsByTagName('source')[0].firstChild.data
            src = node_map[src_name]
            dst_name = demand_item.getElementsByTagName('target')[0].firstChild.data
            dst = node_map[dst_name]
            if src == dst:
                continue
            value = float(demand_item.getElementsByTagName('demandValue')[0].firstChild.data)
            if (src, dst) not in demand_matrix:
                demand_matrix[(src, dst)] = 0
            demand_matrix[(src, dst)] += value
        return self.__get_scaled_tm(demand_matrix)

    def demand_matrix(self, sample: int) -> dict:
        """ Get a single demand matrix """
        assert 0 <= sample < self.__n_samples, "sample nr out of range"
        if sample not in self.__demand_matrix_sets:
            self.__demand_matrix_sets[sample] = self.__read_demand_xml(sample)
        return self.__demand_matrix_sets[sample]

    def demand_sequence(self, sample: int) -> list:
        """ Get a single demand sequence """
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
            except:
                continue

    def __len__(self):
        """ len is defined by the number of samples """
        return self.__n_samples

    def __str__(self):
        self.get_name()

    def get_name(self) -> str:
        return f"SndLib_{self.__topology_name}"
