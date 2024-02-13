import networkx as nx
from typing import FrozenSet, Dict, Tuple, List, Optional, Set

from causaldag import DAG
from code.distributions import DistributionBaseClass
from code.algorithms import AlgorithmBaseClass
from .util import *

class ColoringPolicy(AlgorithmBaseClass):
    """
    Coloring [SKD+15]

    Modified from: https://github.com/csquires/dct-policy/
        blob/master/baseline_policies/coloring_policy.py

    Note: Need to ignore degree 0 nodes from scoring consideration
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
            node = self._pick_coloring_policy_node(G)

            # T contain all edges incident to chosen node
            T = set([frozenset(e) for e in G.edges(node)])

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

    def _pick_coloring_policy_node(self, graph: nx.Graph):
        coloring = nx.greedy_color(graph)
        node_scores = {node: self._score_node(graph, coloring, node)
            for node in graph.nodes}
        return max(node_scores.keys(), key=lambda k: node_scores[k])

    def _score_node(self, graph: nx.Graph, coloring: dict, v):
        if graph.degree(v) == 0:
            return 0

        colors = set(coloring.values())
        forests = [self._induced_forest(graph, coloring, coloring[v], color)
            for color in colors if color != coloring[v]]
        trees_containing_v =\
            [forest.subgraph(nx.node_connected_component(forest, v))
            for forest in forests]
        wc_subtrees = [self._worst_case_subtree(tree, v)
            for tree in trees_containing_v]
        wc_edges_learned = [
            tree.number_of_nodes() - len(wc_subtree)
            for tree, wc_subtree in zip(trees_containing_v, wc_subtrees)
        ]
        return sum(wc_edges_learned)

    def _induced_forest(self, graph: nx.Graph, coloring: dict, color1, color2):
        forest = nx.Graph()
        forest.add_nodes_from(graph.nodes)
        forest.add_edges_from({(i, j) for i, j in graph.edges
            if {coloring[i], coloring[j]} == {color1, color2}})
        return forest
    
    def _worst_case_subtree(self, tree: nx.Graph, root) -> set:
        if tree.number_of_nodes() == 1:
            return set()
        tree_ = tree.copy()
        tree_.remove_node(root)
        subtrees = nx.connected_components(tree_)
        return max(subtrees, key=lambda t: len(t))

