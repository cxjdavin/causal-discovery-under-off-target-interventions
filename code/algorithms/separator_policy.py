import networkx as nx
from typing import FrozenSet, Dict, Tuple, List, Optional, Set

from causaldag import DAG
from code.distributions import DistributionBaseClass
from code.algorithms import AlgorithmBaseClass
from .util import *

class SeparatorPolicy(AlgorithmBaseClass):
    """
    Separator [CSB23]

    Modified from: https://github.com/cxjdavin/
        verification-and-search-algorithms-for-causal-DAGs/
        blob/main/our_code/separator_policy.py
    """
    def solve(
            self,
            dag: DAG,
            k: int,
            weights: List[float], 
            action_distributions: DistributionBaseClass
        ) -> List[FrozenSet[int]]:
        """
        Adaptively recover a causally sufficient causal DAG from its essential
        graph and output the actual interventions used

        Note: the search algorithm should only look at the essential graph to
        decide its next action

        Args:
            dag (DAG): ground truth causal DAG that is causally sufficient
            k (int): number of actions
            weights (List[int]): weights of each action
            action_distributions (DistributionBaseClass):
                Each action corresponds to a DistributionBaseClass object,
                which we can call .sample() on

        Returns:
            intervened_nodes (List[FrozenSet[int]]): Collection of
                interventions that were actually performed
        """
        self.dag = dag
        self.cpdag = dag.cpdag()
        self.k = k
        self.weights = weights
        self.action_distributions = action_distributions
        self.expected_cutting_probabilities = [
            action_distributions[idx].get_expected_cut_probabilities()
            for idx in range(k)
        ]

        intervened_nodes = []
        while len(self.cpdag.edges) > 0:
            G = self._get_chain_components()
    
            # Collect 1/2-clique separators for each chain comp of size >= 2
            K = []
            H_nodes = []
            for cc_nodes in nx.connected_components(G):
                if len(cc_nodes) >= 2:
                    cc = G.subgraph(cc_nodes).copy()
                    clique_separator_nodes = compute_cc_separator(cc)
                    K.append(frozenset(clique_separator_nodes))

            # T contain all edges incident to any clique K_H in K
            T = set()
            for K_H in K:
                for v in K_H:
                    T.update(set([frozenset(e) for e in G.edges(v)]))

            # Orient all edges in T
            while any(frozenset(e) in self.cpdag.edges for e in T):
                # Get sampling probabilities over action space
                action_probs =\
                    get_action_probabilities(T,
                                             self.k,
                                             self.weights,
                                             self.expected_cutting_probabilities)
                action_idx = self.random_state.choice(self.k, p=action_probs)
                intervention = self.action_distributions[action_idx].sample()
                intervened_nodes.append(intervention)
                self.cpdag = self.cpdag.interventional_cpdag(self.dag, intervention)

        return intervened_nodes

