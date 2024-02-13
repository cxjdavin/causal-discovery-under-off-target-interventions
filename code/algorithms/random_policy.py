import networkx as nx
from typing import FrozenSet, Dict, Tuple, List, Optional, Set

from causaldag import DAG
from code.distributions import DistributionBaseClass
from code.algorithms import AlgorithmBaseClass
from .util import *

class RandomPolicy(AlgorithmBaseClass):
    """
    Naive baseline
    """
    def solve(
            self,
            dag: DAG,
            k: int,
            weights: List[float], 
            action_distributions: DistributionBaseClass
        ) -> List[FrozenSet[int]]:
        """
        Repeatedly pick a random action until the graph is fully oriented

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
            # Pick random action
            action_idx = self.random_state.choice(self.k)
            intervention = self.action_distributions[action_idx].sample()
            intervened_nodes.append(intervention)
            self.cpdag = self.cpdag.interventional_cpdag(self.dag, intervention)

        return intervened_nodes

