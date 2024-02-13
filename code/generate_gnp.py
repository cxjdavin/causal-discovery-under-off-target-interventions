import json
import networkx as nx
import numpy as np
import os
import pickle
import sys

from typing import FrozenSet, Tuple, Dict, List, Union

random_state = None

def create_GNP_TREE(n: int, p: float) -> nx.DiGraph:
    """
    Create instance for given graph and action distribution type

    Args:
        n (int): Number of nodes to generate GNP_TREE instance on
        p (float): Probability used to create G(n,p) graph

    Returns:
        nx_dag (nx.DiGraph): Generated Networkx DAG
    """
    gnp = nx.gnp_random_graph(n, p, seed=random_state.randint(1e9))
    tree = nx.random_tree(n, seed=random_state.randint(1e9))

    # Form graph by taking both edgesets and orient in acyclic fashion
    nx_dag = nx.DiGraph()
    nx_dag.add_nodes_from(gnp.nodes)
    nx_dag.add_edges_from([(u,v) if u < v else (v,u) for (u,v) in gnp.edges])
    nx_dag.add_edges_from([(u,v) if u < v else (v,u) for (u,v) in tree.edges])
    
    # Remove v-structures by adding arcs. Iterate from the back.
    for w in range(n-1, -1, -1):
        for v in range(w-1, -1, -1):
            for u in range(v-1, -1, -1):
                if (u,w) in nx_dag.edges\
                    and (v,w) in nx_dag.edges\
                    and (u,v) not in nx_dag.edges:
                    nx_dag.add_edge(u,v)

    return nx_dag

def generate_gnp_tree(json_filename: str, global_rng: int = 314) -> None:
    """
    Run experiment based on given JSON config and experimental settings
    Will print out result in console, then plot and save figures

    Args:
        json_filename (str): Configuration given in JSON format.
            Contains directory names and algorithms to run
        global_random_seed (int): Random seeed used for reproducibility.
            Defaults to 314.

    Returns:
        None
    """
    print("Generating GNP_TREE graphs...")
    global random_state
    random_state = np.random.RandomState(global_rng)

    # Parse config data
    config_data = None
    with open(json_filename) as config_file:
        config_data = json.load(config_file)
    assert config_data is not None

    # Extract JSON settings
    nx_dirname = config_data["dirnames"]["nx"]
    os.makedirs(nx_dirname, exist_ok=True)
    generation_count = config_data["generation_count"]
    generation_sizes = config_data["generation_sizes"]
    generation_probs = config_data["generation_probs"]

    # Create instances
    for n in generation_sizes:
        for p in generation_probs:
            for c in range(1, generation_count+1):
                nx_filename = "{0}/n={1}_p={2}_c={3}.nx".format(nx_dirname, n, p, c)
                nx_dag = create_GNP_TREE(n, p)
                nx.write_adjlist(nx_dag, nx_filename)

