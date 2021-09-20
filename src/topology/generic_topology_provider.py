from abc import abstractmethod


class GenericTopologyProvider:
    @abstractmethod
    def get_topology(self, topology: str, n: int = None, default_capacity: float = 1, **kwargs) -> (list, int):
        """ returns links: list with [(i, j, capacity), ...], n: #nodes as int """
        raise Exception("Abstract topology provider - use a concrete class")

    @staticmethod
    @abstractmethod
    def get_topology_names():
        """ returns names of all supported topologies """
        raise Exception("Abstract topology provider - use a concrete class")
