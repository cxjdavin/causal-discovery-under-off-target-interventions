import networkx as nx
from typing import FrozenSet, Dict, Tuple, List, Optional, Set

from causaldag import DAG
from code.distributions import DistributionBaseClass
from code.algorithms import AlgorithmBaseClass
from .util import *

class OneShotPolicy(AlgorithmBaseClass):
    """
    Compute probability over actions by solving VLP optimally on all unoriented
    edges. CANNOT update its knowledge based on arc orientations that are
    subsequently revealed.
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
        graph and output the actual interventions used WHILE TRYING TO EMULATE
        NON-ADAPTIVE INTERVENTIONS

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

        if len(self.cpdag.edges) > 0:
            # Optimally solve VLP once at the start
            action_probs =\
                get_action_probabilities(self.cpdag.edges,
                                         self.k,
                                         self.weights,
                                         self.expected_cutting_probabilities)

        intervened_nodes = []
        while len(self.cpdag.edges) > 0:
            # Sample actions using non-adaptive action probabilities
            action_idx = self.random_state.choice(self.k, p=action_probs)
            intervention = self.action_distributions[action_idx].sample()
            intervened_nodes.append(intervention)
            self.cpdag = self.cpdag.interventional_cpdag(self.dag, intervention)

        return intervened_nodes

