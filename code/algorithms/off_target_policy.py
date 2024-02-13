import networkx as nx

from typing import FrozenSet, Dict, Tuple, List, Optional, Set

from causaldag import DAG
from code.distributions import DistributionBaseClass
from code.algorithms import AlgorithmBaseClass
from .util import *

class OffTargetPolicy(AlgorithmBaseClass):
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
            # Extract chain components
            G = self._get_chain_components()
    
            # Collect 1/2-clique separators for each chain comp of size >= 2
            K = []
            H_nodes = []
            for cc_nodes in nx.connected_components(G):
                if len(cc_nodes) >= 2:
                    cc = G.subgraph(cc_nodes).copy()
                    clique_separator_nodes = compute_cc_separator(cc)
                    K.append(frozenset(clique_separator_nodes))
                    H_nodes.append(frozenset(cc_nodes))

            # Perform partitioning and collect intervened nodes
            intervened_nodes += self._perform_partitioning(K, H_nodes)
        return intervened_nodes

    def _compute_unoriented_covered_edges(
            self,
            preordering: Optional[List[int]] = []
        ) -> Set[FrozenSet[Tuple[int,int]]]:
        """
        Retrieves all unoriented covered edges with respect to oriented arcs
        and given preordering

        Args:
            preordering (Optional[List[int]], optional): Prefix of topological
                ordering we want to enforce
        """
        # Extract chain components
        G = self._get_chain_components()

        # Setup ordering to orient unoriented edges
        dfs_order = preordering

        # Remove all edges incident to preordering vertices
        for v in preordering:
            edges_incident_to_v = list(G.edges(v))
            G.remove_edges_from(edges_incident_to_v)

        # Pick arbitrary topological ordering on remaining things
        for v in nx.dfs_preorder_nodes(G):
            if v not in dfs_order:
                dfs_order.append(v)
        assert set(dfs_order) == set(G.nodes)
        assert len(dfs_order) == len(G.nodes)

        # Setup mapping from vertex to ordering
        vertex_to_ordering = dict()
        for i in range(len(dfs_order)):
            vertex_to_ordering[dfs_order[i]] = i

        # Combine with existing arcs to form a DAG G' from [G^*]
        G_prime = nx.DiGraph()
        G_prime.add_nodes_from(G.nodes)
        for u, v in self.cpdag.edges:
            if vertex_to_ordering[u] < vertex_to_ordering[v]:
                G_prime.add_edge(u,v)
            else:
                G_prime.add_edge(v,u)
        for u, v in self.cpdag.arcs:
            G_prime.add_edge(u,v)
        assert nx.is_directed_acyclic_graph(G_prime)

        # Extract unoriented covered edges in G'
        covered_edges = DAG(arcs=G_prime.edges).reversible_arcs()
        unoriented_covered = set()
        for e in covered_edges:
            if e not in self.cpdag.arcs:
                unoriented_covered.add(frozenset(e))

        return unoriented_covered

    def _extract_large_chain_components(
            self,
            K: List[FrozenSet[int]],
            H_nodes: List[FrozenSet[int]]
        ) -> Dict[int,int]:
        """
        Extract any remaining large chain components after orienting internal
        edges of clique separators in K

        Args:
            K (List[FrozenSet[int]]): Collection of 1/2-clique separators
            H_nodes (List[FrozenSet[int]]): Nodes in corresponding chain comps

        Returns:
            large_components (Dict[int,int]):
                Each large component (key, val) pair is as follows:
                (key) index H_idx of original chain component within H_nodes
                (val) unique vertex u_H from clique separator K_H = K[H_idx]
        """
        assert len(K) == len(H_nodes)
        large_components = dict()
        G = self._get_chain_components()
        for cc_nodes in nx.connected_components(G):
            if len(cc_nodes) >= 2:
                # Identify the original chain component this used to belong to
                H_idx = None
                for i in range(len(H_nodes)):
                    if cc_nodes.intersection(set(H_nodes[i])):
                        assert H_idx is None
                        H_idx = i
                assert H_idx is not None    

                if len(cc_nodes) > len(H_nodes[H_idx]) / 2:
                    # Identify the unique u_H within L_H
                    shared = K[H_idx].intersection(cc_nodes)
                    assert len(shared) == 1
                    u_H = set(shared).pop()
                    large_components[H_idx] = u_H
        return large_components

    def _perform_partitioning(
            self,
            K: List[FrozenSet[int]],
            H_nodes: List[FrozenSet[int]],
        ) -> List[FrozenSet[int]]:
        """
        Performs Partitioning step in off target search algorithm

        Args:
            K (List[FrozenSet[int]]): Collection of 1/2-clique separators
            H_nodes (List[FrozenSet[int]]): Nodes in corresponding chain comps

        Returns:
            intervened_nodes (List[FrozenSet[int]]): Collection of
                interventions that were actually performed
        """
        assert len(K) == len(H_nodes)
        skel = self._get_skeleton()
        intervened_nodes = []

        # all_K_edges is all edges within all 1/2-clique separators in K
        all_K_edges = set()
        for K_H in K:
            # K_H_edges is all edges within 1/2-clique separator K_H
            K_H_edges =\
                set([frozenset([u,v]) for u, v in self.cpdag.edges
                if u in K_H and v in K_H])
            assert len(K_H_edges) == len(K_H) * (len(K_H) - 1) // 2
            all_K_edges.update(K_H_edges)

        # Orient edgeset containing all the internal edges of K_Hs in K
        # Consider the clique separators as prefix in ordering
        preordering = [v for K_H in K for v in K_H]
        while any(frozenset(e) in self.cpdag.edges for e in all_K_edges):
            unoriented_covered =\
                self._compute_unoriented_covered_edges(preordering)
            edges_to_cut = unoriented_covered.intersection(all_K_edges)

            # Get sampling probabilities over action space
            action_probs =\
                get_action_probabilities(edges_to_cut,
                                         self.k,
                                         self.weights,
                                         self.expected_cutting_probabilities)
            action_idx = self.random_state.choice(self.k, p=action_probs)
            intervention = self.action_distributions[action_idx].sample()
            intervened_nodes.append(intervention)
            self.cpdag = self.cpdag.interventional_cpdag(self.dag, intervention)

        # Handle large chain components, if any
        while True:
            large_components = self._extract_large_chain_components(K, H_nodes)

            # If no large chain components, quit
            if len(large_components) == 0:
                break

            # Gather 1/2-clique separators Z_{H'}s of H', where H' is a chain
            # component of L_H[V(L_H) \cap N(u_H)] for each large component L_H
            Z = []
            for H_idx, u_H in large_components.items():
                L_H = skel.subgraph(H_nodes[H_idx])
                induced_nodes = self.cpdag.undirected_neighbors_of(u_H)\
                    .intersection(list(H_nodes[H_idx]))
                assert len(induced_nodes.intersection([u for K_H in K
                    for u in K_H])) == 0
                induced_subgraph = skel.subgraph(induced_nodes).copy()
                for H_prime_nodes in nx.connected_components(induced_subgraph):
                    cc = induced_subgraph.subgraph(H_prime_nodes).copy()
                    Z_H_prime = compute_cc_separator(cc)
                    Z.append((H_idx, Z_H_prime))
                    for w in Z_H_prime:
                        assert frozenset([u_H, w]) in self.cpdag.edges

            # Extract all edges within all 1/2-clique separators in Z
            edges_to_cut = set()
            for H_idx, Z_H_prime in Z:
                Z_H_prime_edges =\
                    set([frozenset([u, v]) for u, v in self.cpdag.edges
                    if u in Z_H_prime and v in Z_H_prime])
                assert len(Z_H_prime_edges) ==\
                    len(Z_H_prime) * (len(Z_H_prime) - 1) // 2
                if len(Z_H_prime_edges) > 0:
                    edges_to_cut.update(Z_H_prime_edges)

            # Orient edgeset containing all internal edges of Z_{H'}s in Z
            while any(frozenset(e) in self.cpdag.edges for e in edges_to_cut):
                preordering = [u_H for u_H in large_components.values()] +\
                    [v for _, Z_H_prime in Z for v in Z_H_prime]

                unoriented_covered =\
                    self._compute_unoriented_covered_edges(preordering)
                edges_to_cut = unoriented_covered\
                    .intersection([frozenset(e) for e in edges_to_cut])

                # Get sampling probabilities over action space
                action_probs =\
                    get_action_probabilities(edges_to_cut,
                                             self.k,
                                             self.weights,
                                             self.expected_cutting_probabilities)
                action_idx = self.random_state.choice(self.k, p=action_probs)
                intervention = self.action_distributions[action_idx].sample()
                intervened_nodes.append(intervention)
                self.cpdag = self.cpdag.interventional_cpdag(self.dag, intervention)

            # Extract u_H ~ Z_H_prime_source edges while tracking index in Z
            u_Z_edges = []
            for Z_idx in range(len(Z)):
                H_idx, Z_H_prime = Z[Z_idx]
                candidates = set(Z_H_prime)
                for u, v in self.cpdag.arcs:
                    if u in Z_H_prime and v in Z_H_prime:
                        candidates.discard(v)
                assert len(candidates) == 1
                Z_H_prime_source = candidates.pop()
                u_H = large_components[H_idx]
                u_Z_edges.append((Z_idx, (u_H, Z_H_prime_source)))

            # Cut all u_H ~ Z_H_prime_source edges, if not already cut
            while any(frozenset(e) in self.cpdag.edges for _, e in u_Z_edges):
                preordering = [u_H for u_H in large_components.values()] +\
                    [e[1] for _, e in u_Z_edges]

                unoriented_covered =\
                    self._compute_unoriented_covered_edges(preordering)
                edges_to_cut = unoriented_covered\
                    .intersection([frozenset(e) for _, e in u_Z_edges])

                assert len(edges_to_cut) > 0

                # Get sampling probabilities over action space
                action_probs =\
                    get_action_probabilities(edges_to_cut,
                                             self.k,
                                             self.weights,
                                             self.expected_cutting_probabilities)
                action_idx = self.random_state.choice(self.k, p=action_probs)
                intervention = self.action_distributions[action_idx].sample()
                intervened_nodes.append(intervention)
                self.cpdag = self.cpdag.interventional_cpdag(self.dag, intervention)

        return intervened_nodes

