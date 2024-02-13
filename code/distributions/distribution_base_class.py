import networkx as nx
import numpy as np

from abc import ABC, abstractmethod
from typing import FrozenSet, Tuple, Dict, List, Union

class DistributionBaseClass(ABC):
    def __init__(
            self,
            seed: int,
            nx_dag: nx.Graph,
            intended_target: FrozenSet[int],
            params: List[Union[int, float]]
        ) -> None:
        # For now, assume that intended target is a subset of vertices
        # If len(intended_target) == 1, then we have atomic interventions
        for v in intended_target:
            assert v in nx_dag.nodes

        self.n = nx_dag.number_of_nodes()
        self.random_state = np.random.RandomState(seed)
        self.num_edges = nx_dag.number_of_edges()
        self.target = intended_target
        self.params = params
        self.expected_cut_probabilities = dict()
        self._setup(nx_dag)

    def get_expected_cut_probabilities(
            self
        ) -> Dict[FrozenSet[Tuple[int, int]], float]:
        assert len(self.expected_cut_probabilities) == self.num_edges
        return self.expected_cut_probabilities

    @abstractmethod
    def _setup(self, nx_dag: nx.Graph) -> None:
        pass

    @abstractmethod
    def sample(self) -> FrozenSet[int]:
        pass

