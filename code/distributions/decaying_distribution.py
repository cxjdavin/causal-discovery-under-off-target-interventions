import networkx as nx
import numpy as np

from abc import ABC, abstractmethod
from typing import FrozenSet, Tuple, Dict, List

from .distribution_base_class import DistributionBaseClass

class DecayingDistribution(DistributionBaseClass):
    def _setup(self, nx_dag: nx.Graph) -> None:
        assert len(self.target) == 1
        assert len(self.params) == 1
        center = set(self.target).pop()
        alpha = self.params[0]

        w = [0.0 for _ in range(self.n)]
        hop_layers = self._bfs_layers(nx_dag, center)
        for hop, nodes in dict(enumerate(hop_layers)).items():
            for node in nodes:
                w[node] = pow(alpha, hop)
        self.prob = w / np.sum(w)
        assert np.isclose(1, np.sum(self.prob))

        for u,v in nx_dag.edges:
            self.expected_cut_probabilities[frozenset([u,v])] =\
                self.prob[u] + self.prob[v]

    def sample(self) -> FrozenSet[int]:
        return frozenset([self.random_state.choice(self.n, p=self.prob)])

    def _bfs_layers(self, G, sources):
        """
        Returns an iterator of all the layers in breadth-first search traversal.
        
        No idea why API call failed, so just copy directly from
        https://networkx.org/documentation/stable/_modules/networkx/
            algorithms/traversal/breadth_first_search.html#bfs_layers
        """
        if sources in G:
            sources = [sources]
    
        current_layer = list(sources)
        visited = set(sources)
    
        for source in current_layer:
            if source not in G:
                raise nx.NetworkXError(f"The node {source} is not in the graph.")
    
        # this is basically BFS, except that the current layer only stores the
        # nodes at same distance from sources at each iteration
        while current_layer:
            yield current_layer
            next_layer = []
            for node in current_layer:
                for child in G[node]:
                    if child not in visited:
                        visited.add(child)
                        next_layer.append(child)
            current_layer = next_layer

