from abc import abstractmethod


class GenericDemandProvider:
    @abstractmethod
    def demand_matrix(self, sample: int) -> dict:
        raise Exception("Abstract traffic matrix factory - use a concrete class")

    @abstractmethod
    def demand_sequence(self, sample: int) -> list:
        raise Exception("Abstract traffic matrix factory - use a concrete class")

    @abstractmethod
    def demand_matrices(self) -> dict:
        raise Exception("Abstract traffic matrix factory - use a concrete class")

    @abstractmethod
    def demand_sequences(self) -> list:
        raise Exception("Abstract traffic matrix factory - use a concrete class")

    @abstractmethod
    def get_name(self) -> str:
        raise Exception("Abstract traffic matrix factory - use a concrete class")
