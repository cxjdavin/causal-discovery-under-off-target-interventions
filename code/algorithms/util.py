import cvxpy as cp
import networkx as nx
import numpy as np

from collections import defaultdict
from typing import FrozenSet, Dict, Tuple, List, Union, Set

from causaldag import DAG
from code.distributions import DistributionBaseClass

import sys
sys.path.insert(0, "./code/PADS")
import LexBFS

def solve_VLP(
        T: Set[FrozenSet[Tuple[int,int]]],
        k: int,
        weights: List[float],
        expected_cutting_probabilities:
            List[Dict[FrozenSet[Tuple[int, int]], float]]
    ) -> Tuple[float, List[float]]:
    """
    Form and solve verification LP (VLP in paper):

        minimize sum_{i=1}^k w_i * x_i
      subject to sum_{i=1}^k E[c_i(e)] * x_i >= 1, for all edges e in T
                                         x_i >= 0, for all i in {1, 2, ..., k}

    Args:
        T (set): edges to cut, each edge is a FrozenSet of pairs of integers
        k (int): number of actions
        weights (List[int]): weights of each action
        expected_cutting_probabilities
            (List[Dict[FrozenSet[Tuple[int, int]], float]]):
            expected probabilities of each action cutting an edge

    Returns:
        obj_value (float): optimal objective value from output of LP
        X_values (List[float]): optimal x^* vector from output of LP
    """
    X = cp.Variable(k)
    constraints = []
    for e in T:
        u,v = e
        mu_e = [expected_cutting_probabilities[idx][frozenset([u,v])]\
            for idx in range(k)]
        constraints += [mu_e @ X >= 1]
    constraints += [X >= 0]
    objective = cp.Minimize(weights @ X)
    VLP = cp.Problem(objective, constraints)
    VLP.solve(abstol=0.01, reltol=0.01)

    # Default solver seems to be ECOS
    # It has options 'abstol' (default: 1e-8) and 'reltol' (default: 1e-8)
    """
    print("Solver {0} ran for {1} time with output status {2}".format(
        VLP.solver_stats.solver_name,
        VLP.solver_stats.solve_time,
        VLP.status
    ))
    """

    assert VLP.status == "optimal"
    obj_value = VLP.value
    X_values = [X.value[i] for i in range(k)]
    return obj_value, X_values

def validate_results(dag: DAG, intervened_nodes: List[FrozenSet[int]]) -> bool:
    """
    Validate that the intervened nodes really fully orient the DAG

    Args:
        dag (DAG): ground truth causal DAG that is causally sufficient
        intervened_nodes (List[FrozenSet[int]]): Collection of interventions
            that were actually performed
    """
    cpdag = dag.cpdag()
    for intervention in intervened_nodes:
        cpdag = cpdag.interventional_cpdag(dag, intervention)
    return cpdag.num_arcs == dag.num_arcs

def generate_distribution(
        seed: int,
        nx_dag: nx.Graph,
        distribution_type: DistributionBaseClass,
        params: List[Union[int, float]]
    ) -> Tuple[
        Dict[FrozenSet[int], DistributionBaseClass],
        Dict[Tuple[int, FrozenSet[Tuple[int, int]]], float]
    ]:
    """
    Generate action distributions and expected cutting probabilities based on
    distribution class of interest
    
    Args:
        seed (int): For reproducibility
        nx_dag (nx.Graph): Underlying causal graph skeleton
        distribution_type (DistributionBaseClass): What kind of interventions
            are we simulating?
        params (List[Union[int, float]]): Parameters associated to simulated
            intervention

    Returns:
        action_distributions (list): Each action corresponds to a
            DistributionBaseClass object, which we can call .sample() on
        expected_cutting_probabilities (dict): expected probabilities of an
            action cutting an edge
    """
    action_distributions = dict()
    expected_cutting_probabilities = dict()
    for v in range(len(nx_dag.nodes)):
        intended_target = frozenset([v])
        dist = distribution_type(nx_dag, intended_target, params)
        action_distributions[intended_target] = dist
        for key, val in dist.get_expected_cut_probabilities.items():
            expected_cutting_probabilities[intended_target, key] = val
    return action_distributions, expected_cutting_probabilities

def get_action_probabilities(
        edges_to_cut: Set[FrozenSet[Tuple[int,int]]],
        k: int,
        weights: List[float],
        expected_cutting_probabilities:
            List[Dict[FrozenSet[Tuple[int, int]], float]]
    ) -> List[float]:
    """
    Solve VLP and interpret X_values as a probability over actions

    Args:
        k (int): number of actions
        weights (List[int]): weights of each action
        expected_cutting_probabilities
            (List[Dict[FrozenSet[Tuple[int, int]], float]]):
            expected probabilities of each action cutting an edge

    Returns:
        action_probs (List[float]): Probability vector over actions
    """
    _, X_values =\
        solve_VLP(edges_to_cut, k, weights, expected_cutting_probabilities)
    action_probs = np.clip(X_values, a_min=0, a_max=None)
    action_probs /= np.sum(action_probs)
    assert np.isclose(1, np.sum(action_probs))
    return action_probs 

def compute_clique_graph_separator(
        adj_list: Dict[int, Tuple[int, int]]
    ) -> List[int]:
    """
    Compute 1/2-clique graph separator of a connected chordal graph on n nodes

    Args:
        adj_list (Dict[int, Tuple[int, int]]): Adjacency list

    Returns:
        C (List[int]): Clique separator
    """
    G = nx.Graph(adj_list)
    assert nx.is_connected(G)
    assert nx.is_chordal(G)
    n = len(G.nodes)

    # Compute perfect elimination ordering via LexBFS from PADS
    # Source: https://www.ics.uci.edu/~eppstein/PADS/ABOUT-PADS.txt
    lexbfs_output = list(LexBFS.LexBFS(adj_list))

    # Reverse computed ordering to get actual perfect elimination ordering
    peo = lexbfs_output[::-1]
    
    # Build maps between node index and peo index
    actual_to_peo = dict()
    peo_to_actual = dict()
    for i in range(n):
        peo_to_actual[i] = peo[i]
        actual_to_peo[peo[i]] = i

    # FAST CHORDAL SEPARATOR algorithm of [GRE84]
    # Reference: [GRE84] A Separator Theorem for Chordal Graphs
    w = [1] * n
    total_weight = sum(w)

    peo_i = 0
    while w[peo_i] <= total_weight/2:
        # w[i] is weight of connected component {v_0, ..., v_i} containing v_i
        # v_k is the lowest numbered neighbor of v_i with k > i
        k = None
        for j in adj_list[peo_to_actual[peo_i]]:
            if (actual_to_peo[j] > peo_i and
                (k is None or actual_to_peo[j] < actual_to_peo[k])):
                k = j
        if k is not None:
            w[actual_to_peo[k]] += w[peo_i]
        peo_i += 1

    # i is the minimum such that some component of {v_0, ..., v_i}
    # weighs more than total_weight / 2
    # C is the v_i plus all of v_{i+1}, ..., v_n that are adjacent to v_i
    C = [peo_to_actual[peo_i]]
    for j in adj_list[peo_to_actual[peo_i]]:
        if actual_to_peo[j] > peo_i:
            C.append(j)
    return C

def compute_cc_separator(cc: nx.Graph) -> List[int]:
    """
    Compute clique separator for chain component
    Remark: Need to do some index remappings as nodes in cc may not be 0-index

    Args:
        cc (nx.Graph): Chain component of interest

    Returns:
        clique_separator_nodes (List[int]): Nodes of a clique separator for cc
    """
    assert not nx.is_directed(cc)
    assert nx.is_connected(cc)

    # Map indices of subgraph into 0..n-1
    map_idx = dict()
    unmap_idx = dict()
    for v in cc.nodes():
        map_idx[v] = len(map_idx)
        unmap_idx[map_idx[v]] = v

    # Extract mapped adj_list
    adj_list = dict()
    for v, nbr_dict in cc.adjacency():
        adj_list[map_idx[v]] = [map_idx[x] for x in list(nbr_dict.keys())]

    # Compute clique separator for this connected component
    clique_separator_nodes =\
        [unmap_idx[v] for v in compute_clique_graph_separator(adj_list)]
    return clique_separator_nodes

