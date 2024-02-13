import networkx as nx

from abc import ABC, abstractmethod
from typing import FrozenSet, Tuple, Dict, List

from .distribution_base_class import DistributionBaseClass

class OnTargetDistribution(DistributionBaseClass):
    def _setup(self, nx_dag: nx.Graph) -> None:
        assert len(self.target) == 1
        assert len(self.params) == 1
        assert self.params[0] == None

        for u,v in nx_dag.edges:
            # Check if edge is being cut by the intended target
            if ((u in self.target and v not in self.target) or
                (v in self.target and u not in self.target)):
                self.expected_cut_probabilities[frozenset([u,v])] = 1
            else:
                self.expected_cut_probabilities[frozenset([u,v])] = 0

    def sample(self) -> FrozenSet[int]:
        return self.target

