import json
import networkx as nx
import numpy as np
import os
import pickle
import sys

from tqdm import tqdm
from p_tqdm import p_map
from typing import FrozenSet, Tuple, Dict, List, Union

from code.distributions import OnTargetDistribution
from code.distributions import RHopDistribution
from code.distributions import DecayingDistribution
from code.distributions import FatHandDistribution

from code.algorithms.util import *
from code.algorithms import OffTargetPolicy
from code.algorithms import ColoringPolicy
from code.algorithms import SeparatorPolicy
from code.algorithms import OneShotPolicy
from code.algorithms import RandomPolicy

multithread = False

dist_dict = {
    "on_target": OnTargetDistribution,
    "r_hop": RHopDistribution,
    "decaying": DecayingDistribution,
    "fat_hand": FatHandDistribution
}

alg_dict = {
    "off_target": OffTargetPolicy,
    "coloring": ColoringPolicy,
    "separator": SeparatorPolicy,
    "one_shot": OneShotPolicy,
    "random": RandomPolicy,
}

def solve_instance(
        alg_name: str,
        global_rng: int,
        num_times: int,
        instance: Tuple[DAG, int, List[int], List[DistributionBaseClass]],
        result_pickle: str
    ) -> None:
    """
    Solve instance given algorithm

    Args:
        alg_name (str): Name of algorithm to run
        global_rng (int): For reproducibility
        num_times (int): Number of times to run experiment
        instance (Tuple[DAG, int, List[int], List[DistributionBaseClass]):
            Each instance is a tuple of 4 items:
            (1) DAG
            (2) Number of actions k
            (3) Weights of each action
            (4) Action distribution for each action
        result_pickle (str): Name of result pickle

    Returns:
        None
    """
    dag, k, weights, action_distr_objs = instance

    if alg_name == "VLP_opt":
        # Gather covered edges
        covered_edges = set([frozenset(e) for e in dag.reversible_arcs()])
    
        # Arbitrarily use one of the copies since all copies have the same
        # expected cutting probabilities and we will not be using .sample()
        opt, _ = solve_VLP(
            covered_edges,
            k,
            weights,
            [x.get_expected_cut_probabilities() for x in action_distr_objs[0]]
        )

        # Round up to 2 decimals and store
        opt = np.clip(np.ceil(opt * 100) / 100, a_min = 0, a_max = None)
        all_intervened_nodes = [opt for _ in range(num_times)]
    else:
        if multithread:
            all_intervened_nodes =\
                p_map(solve_one_time,
                      [alg_name] * num_times,
                      [global_rng] * num_times,
                      [dag] * num_times,
                      [k] * num_times,
                      [weights] * num_times,
                      action_distr_objs,
                      desc="Multi thread solving with {0}".format(alg_name))
        else:
            all_intervened_nodes = []
            for idx in tqdm(range(num_times),
                    desc="Single thread solving with {0}".format(alg_name)):
                alg_obj = alg_dict[alg_name](global_rng)
                action_distr_obj = action_distr_objs[idx]
                intervened_nodes = solve_one_time(alg_name,
                                                  global_rng,
                                                  dag,
                                                  k,
                                                  weights,
                                                  action_distr_obj)
                all_intervened_nodes.append(intervened_nodes)

    # Save to file
    pickle.dump(all_intervened_nodes, open(result_pickle, 'wb'))

def solve_one_time(
        alg_name: str,
        global_rng: int,
        dag: DAG,
        k: int,
        weights: List[int],
        action_distr_obj: List[DistributionBaseClass]
    ) -> List[FrozenSet[int]]:
    """
    Solve one copy of the instance (out of num_times)

    Args:
        alg_name (str): Name of algorithm to run
        global_rng (int): For reproducibility
        dag: DAG object
        k: Number of actions
        weights: Weights of each action
        action_distr_obj: Action distribution for each action

    Returns:
        intervened_nodes (List[FrozenSet[int]]): Collection of
            interventions that were actually performed
    """
    alg_obj = alg_dict[alg_name](global_rng)
    intervened_nodes = alg_obj.solve(dag, k, weights, action_distr_obj)
    assert validate_results(dag, intervened_nodes)
    return intervened_nodes

def solve_single_instance_with_config(
        json_filename: str,
        nx_name: str,
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
        nx_name (str): Name of instance to run
        num_times (int): Number of runs per experimental graph.
        dist_name (str): Type of experiment. One of the following:
            "on_target", "r_hop", "decaying", "fat_hand"
        param (str): Setting for experiment. May be None.
        global_random_seed (int): Random seeed used for reproducibility.
            Defaults to 314.

    Returns:
        None
    """
    print("---------")
    print("Solving {0}...".format(nx_name))

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
    global multithread
    multithread = config_data["multithread"]
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
    alg_list = config_data["algorithm_list"]
    instance_tuple = [num_times, exp_setting, nx_name]

    # Collect results
    results = defaultdict(list)
    num_times, exp_setting, nx_name = instance_tuple
    instance_str = str(instance_tuple)
    instance_pickle = "{0}/{1}.pickle"\
        .format(instances_dirname, str(instance_tuple))

    # Get results for this instance
    # If results exist, just extract them out
    # Otherwise, solve and store them
    instance = None
    for alg_name in alg_list:
        result_str = "{0}_{1}".format(instance_str, alg_name)
        result_pickle = "{0}/{1}.pickle"\
            .format(results_dirname, result_str)
        if not os.path.exists(result_pickle):
            # Load instance only once, if necessary
            if instance is None:
                assert os.path.exists(instance_pickle)
                instance = pickle.load(open(instance_pickle, 'rb'))
            all_intervened_nodes =\
                solve_instance(alg_name,
                               global_rng,
                               num_times,
                               instance,
                               result_pickle)
       
    # Print results for this instance
    print("nx: {0}, {1} times, action distribution: {2}"\
        .format(nx_name, num_times, exp_setting))
    for alg_name in alg_list:
        result_str = "{0}_{1}".format(instance_str, alg_name)
        result_pickle = "{0}/{1}.pickle"\
            .format(results_dirname, result_str)
        assert os.path.exists(result_pickle)
        pickled_results = pickle.load(open(result_pickle, 'rb'))
        if alg_name == "VLP_opt":
            print("VLP_opt: {0}".format(pickled_results))
        else:
            # Compute mean and std, then round up to 2 decimals
            counts = [len(x) for x in pickled_results]
            mean = np.ceil(np.mean(counts) * 100) / 100
            std = np.ceil(np.std(counts) * 100) / 100
            #print("{0}: {1} (± {2})".format(alg_name, mean, std))
            print("{0}: {1} -> {2} (± {3})".format(alg_name,
                                                   counts,
                                                   mean,
                                                   std))

if __name__ == "__main__":
    json_filename = sys.argv[1]
    nx_name = sys.argv[2]
    num_times = int(sys.argv[3])
    dist_name = sys.argv[4]
    param = None
    if len(sys.argv) > 5:
        param = sys.argv[5]

    solve_single_instance_with_config(json_filename,
                                      nx_name,
                                      num_times,
                                      dist_name,
                                      param)

