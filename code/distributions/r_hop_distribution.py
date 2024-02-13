import networkx as nx
import numpy as np

from abc import ABC, abstractmethod
from typing import FrozenSet, Tuple, Dict, List

from .distribution_base_class import DistributionBaseClass

class RHopDistribution(DistributionBaseClass):
    def _setup(self, nx_dag: nx.Graph) -> None:
        assert len(self.target) == 1
        assert len(self.params) == 1
        center = set(self.target).pop()
        r = self.params[0]

        r_hop_subgraph = nx.ego_graph(nx_dag, center, radius=r)
        self.r_hop_neighbors = list(r_hop_subgraph.nodes())

        denom = len(self.r_hop_neighbors)
        for u,v in nx_dag.edges:
            self.expected_cut_probabilities[frozenset([u,v])] =\
                len(set([u,v]).intersection(self.r_hop_neighbors))/denom

    def sample(self) -> FrozenSet[int]:
        return frozenset([self.random_state.choice(self.r_hop_neighbors)])

