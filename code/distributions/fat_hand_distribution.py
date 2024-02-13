import networkx as nx
import numpy as np

from abc import ABC, abstractmethod
from typing import FrozenSet, Tuple, Dict, List

from .distribution_base_class import DistributionBaseClass

class FatHandDistribution(DistributionBaseClass):
    def _setup(self, nx_dag: nx.Graph) -> None:
        assert len(self.target) == 1
        assert len(self.params) == 1
        center = set(self.target).pop()
        p = self.params[0]

        self.neighbors = list(nx_dag.neighbors(center))
        closed_nbrhood = self.neighbors + [center]
        for u,v in nx_dag.edges:
            if len(self.neighbors) == 0:
                if u == center or v == center:
                    self.expected_cut_probabilities[frozenset([u,v])] = 1
                else:
                    self.expected_cut_probabilities[frozenset([u,v])] = 0
            else:
                if u not in closed_nbrhood and v not in closed_nbrhood:
                    self.expected_cut_probabilities[frozenset([u,v])] = 0
                elif u in closed_nbrhood and v not in closed_nbrhood:
                    if u == center:
                        self.expected_cut_probabilities[frozenset([u,v])] = 1
                    else:
                        assert u in closed_nbrhood
                        self.expected_cut_probabilities[frozenset([u,v])] =\
                            p / len(self.neighbors)
                elif v in closed_nbrhood and u not in closed_nbrhood:
                    if v == center:
                        self.expected_cut_probabilities[frozenset([u,v])] = 1
                    else:
                        assert v in closed_nbrhood
                        self.expected_cut_probabilities[frozenset([u,v])] =\
                            p / len(self.neighbors)
                else:
                    assert u in closed_nbrhood and v in closed_nbrhood
                    if u == center:
                        assert v in self.neighbors
                        self.expected_cut_probabilities[frozenset([u,v])] =\
                            (1 - p) + p * (1 - p / len(self.neighbors))
                    elif v == center:
                        assert u in self.neighbors
                        self.expected_cut_probabilities[frozenset([u,v])] =\
                            (1 - p) + p * (1 - p / len(self.neighbors))
                    else:
                        assert u in self.neighbors and v in self.neighbors
                        self.expected_cut_probabilities[frozenset([u,v])] = 0

    def sample(self) -> FrozenSet[int]:
        p = self.params[0]
        intervened = set(self.target)
        if len(self.neighbors) > 0 and self.random_state.random() <= p:
            x = self.random_state.choice(self.neighbors)
            intervened.add(x)
        return frozenset(intervened)

