import json
import networkx as nx
import numpy as np
import os
import pickle
import sys

from typing import FrozenSet, Tuple, Dict, List, Union
from causaldag import DAG

from code.distributions import OnTargetDistribution
from code.distributions import RHopDistribution
from code.distributions import DecayingDistribution
from code.distributions import FatHandDistribution

random_state = None

dist_dict = {
    "on_target": OnTargetDistribution,
    "r_hop": RHopDistribution,
    "decaying": DecayingDistribution,
    "fat_hand": FatHandDistribution
}

def create_instance(
        num_times: int,
        nx_name: str,
        dist_name: str,
        params: List[Union[int, float]],
        instance_pickle: str
    ) -> None:
    """
    Create instance for given graph and action distribution type

    Args:
        num_times (int): Number of times to run on this graph input
        nx_name (str): Name of networkx graph file
        dist_name (str): Name of action distribution
        params (List[Union[int, float]]): Parameters for action distribution
        instance_pickle (str): Name of instance pickle

    Returns:
        None
    """
    # Import networkx graph
    nx_dag = nx.read_adjlist(nx_name, create_using=nx.DiGraph)
    nx_dag = nx.relabel_nodes(nx_dag, lambda x: int(x))
    n = nx_dag.number_of_nodes()
    
    # Setup action weights, each action corresponds to a vertex
    k = n
    weights = [1] * k
    
    # Generate distribution objects
    action_distr_objs = []
    for _ in range(num_times):
        action_distr_objs.append([
            dist_dict[dist_name](random_state.randint(int(1e9)),
                                 nx_dag,
                                 frozenset([v]),
                                 params)
            for v in range(n)
        ])
    
    # Save to file
    dag = DAG.from_nx(nx_dag)
    instance = (dag, k, weights, action_distr_objs)
    pickle.dump(instance, open(instance_pickle, 'wb'))

def setup_instances(
        json_filename: str,
        num_times: int,
        dist_name: str,
        param: str,
        global_rng: int = 314
    ) -> None:
    """
    Run experiment based on given JSON config and experimental settings
    Will print out result in console, then plot and save figures

    Args:
        json_filename (str): Configuration given in JSON format.
            Contains directory names and algorithms to run
        num_times (int): Number of runs per experimental graph.
        dist_name (str): Type of experiment. One of the following:
            "on_target", "r_hop", "decaying", "fat_hand"
        param (str): Setting for experiment. May be None.
        global_random_seed (int): Random seeed used for reproducibility.
            Defaults to 314.

    Returns:
        None
    """
    print("Creating instances...")
    global random_state
    random_state = np.random.RandomState(global_rng)

    # Parse config data
    config_data = None
    with open(json_filename) as config_file:
        config_data = json.load(config_file)
    assert config_data is not None

    # Setup directories
    nx_dirname = config_data["dirnames"]["nx"]
    instances_dirname = config_data["dirnames"]["instances"]
    results_dirname = config_data["dirnames"]["results"]
    figures_dirname = config_data["dirnames"]["figures"]
    assert os.path.exists(nx_dirname)
    os.makedirs(instances_dirname, exist_ok=True)
    os.makedirs(results_dirname, exist_ok=True)
    os.makedirs(figures_dirname, exist_ok=True)

    # Setup experiment configurations
    assert dist_name in ["on_target", "r_hop", "decaying", "fat_hand"]
    if dist_name == "on_target":
        assert param is None
    elif dist_name == "r_hop":
        assert param is not None
        param = int(param)
    else:
        assert param is not None
        param = float(param)
    exp_setting = [dist_name, [param]]

    # Create instances
    for nx_file in os.listdir(nx_dirname):
        nx_name, ext = os.path.splitext(nx_file)
        instance_tuple = [num_times, exp_setting, nx_name]
        instance_pickle = "{0}/{1}.pickle"\
            .format(instances_dirname, str(instance_tuple))
        nx_name = "{0}/{1}.nx".format(nx_dirname, nx_name)
        if not os.path.exists(instance_pickle):
            create_instance(num_times,
                            nx_name,
                            dist_name,
                            [param],
                            instance_pickle)

