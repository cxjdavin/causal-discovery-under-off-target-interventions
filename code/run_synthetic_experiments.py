import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pickle
import subprocess
import sys

from typing import FrozenSet, Tuple, Dict, List, Union

from code.generate_gnp import *
from code.setup_instances import *
from code.solve_single_experiment import *

def plot_synthetic_results(
        generation_count: int,
        generation_sizes: List[int],
        generation_probs: List[float],
        figures_dirname: str,
        all_instance_tuples:\
            List[Tuple[int, Tuple[str, List[Union[int, float]]], List[str], str]],
        alg_list: List[str],
        results: Dict[str, List[int]]
    ) -> None:
    """
    generation_count (int): Number of G(n,p) graphs for a given {n,p}
    generation_sizes (List[int]): Number of nodes used for G(n,p)
    generation_probs (List[float]): Probabilities used for G(n,p)
    figures_dirname (str): Directory to save figures into
    all_instance_tuples
        (List[Tuple[int, Tuple[str, List[Union[int, float]]], List[str], str]]):
        A collection of instance settings, each of which is a tuple of 4 items:
        (1) num_times (int): Number of times to run experiment
        (2) exp_setting (Tuple[str, List[Union[int, float]]]): experimental
            setup including the name of the distribution and the parameters
        (4) nx_name (str): Name of networkx graph file
    alg_list List[str]: List of algorithms run
    results (Dict[str, List[int]]): Collection of results
    """
    nt = None
    metrics = alg_list
    dist_names_and_params = set()
    aggregated_results = defaultdict(list)
    for instance_tuple in all_instance_tuples:
        num_times, exp_setting, nx_name = instance_tuple
        if nt is None:
            nt = num_times
        else:
            assert nt == num_times

        dist_name, params = exp_setting
        dist_names_and_params.add((dist_name, tuple(params)))
        nstr, pstr, _ = nx_name.split("_")
        n = nstr.split("=")[1]
        p = pstr.split("=")[1]

        instance_str = str(instance_tuple)
        for alg_name in alg_list:
            result_str = "{0}_{1}".format(instance_str, alg_name)
            key = (dist_name, tuple(params), n, p, alg_name)
            aggregated_results[key].append(results[result_str])

    for dist_name, params in dist_names_and_params:
        front = None
        if dist_name == "on_target":
            front = "On target distribution"
        elif dist_name == "r_hop":
            front = "{0}-hop distribution".format(params[0])
        elif dist_name == "decaying":
            front = "Decaying distribution, alpha = {0}".format(params[0])
        elif dist_name == "fat_hand":
            front = "Fat hand distribution, p = {0}".format(params[0])
        else:
            assert False
        assert front is not None

        for p in generation_probs:
            back = "on synthetic GNP_TREE with p = {0} (Avg. over {1} graphs "\
                "per size and {2} runs each)".format(p, generation_count, nt)
            plot_title = "{0} {1}".format(front, back)

            mean_vals = defaultdict(list)
            std_vals = defaultdict(list)
            for alg_name in alg_list:
                for n in generation_sizes:
                    # Compute mean and std, then round to 2 decimals
                    key = (dist_name, params, str(n), str(p), alg_name)
                    mean = np.ceil(np.mean(aggregated_results[key]) * 100)/100
                    std = np.ceil(np.std(aggregated_results[key]) * 100)/100
                    mean_vals[alg_name].append(mean)
                    std_vals[alg_name].append(std)

            # Plot
            fig, ax = plt.subplots(figsize=(16,12))
            for idx, metric in enumerate(metrics):
                ax.errorbar(generation_sizes,
                            mean_vals[metric],
                            yerr=std_vals[metric],
                            label=metric,
                            capsize=5)
            plt.xticks(generation_sizes, generation_sizes)
            ax.set_ylabel("Number of interventions")
            ax.set_xlabel("Number of nodes")
            ax.set_title(plot_title)
            ax.legend()

            # Save plot
            plt.tight_layout()
            fname = "{0}_{1}_{2}_{3}".format(nt, p, exp_setting, dist_name)
            plt.savefig("{0}/synthetic_{1}.png".format(figures_dirname, fname),
                        format="png",
                        dpi=300,
                        bbox_inches="tight")

            # Plot (without random)
            fig, ax = plt.subplots(figsize=(16,12))
            for idx, metric in enumerate(metrics):
                if metric == "random":
                    continue
                ax.errorbar(generation_sizes,
                            mean_vals[metric],
                            yerr=std_vals[metric],
                            label=metric,
                            capsize=5)
            plt.xticks(generation_sizes, generation_sizes)
            ax.set_ylabel("Number of interventions")
            ax.set_xlabel("Number of nodes")
            ax.set_title(plot_title)
            ax.legend()

            # Save plot
            plt.tight_layout()
            fname = "{0}_{1}_{2}_{3}".format(nt, p, exp_setting, dist_name)
            plt.savefig("{0}/synthetic_no_random_{1}.png".format(figures_dirname, fname),
                        format="png",
                        dpi=300,
                        bbox_inches="tight")

def run_with_config(
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
    # Generate synthetic instances
    generate_gnp_tree(json_filename, global_rng)

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
    alg_list = config_data["algorithm_list"]
    assert "VLP_opt" in alg_list

    # Setup instances
    setup_instances(json_filename, num_times, dist_name, param, global_rng)

    # Solve instances one by one
    all_instance_tuples = []
    for nx_file in os.listdir(nx_dirname):
        nx_name, ext = os.path.splitext(nx_file)
        instance_tuple = [num_times, exp_setting, nx_name]
        all_instance_tuples.append(instance_tuple)
        solve_single_instance_with_config(json_filename,
                                          nx_name,
                                          num_times,
                                          dist_name,
                                          param,
                                          global_rng)

    # Collect results
    results = dict()
    for instance_tuple in all_instance_tuples:
        num_times, exp_setting, nx_name = instance_tuple
        instance_str = str(instance_tuple)
        instance_pickle = "{0}/{1}.pickle"\
            .format(instances_dirname, str(instance_tuple))
        for alg_name in alg_list:
            result_str = "{0}_{1}".format(instance_str, alg_name)
            result_pickle = "{0}/{1}.pickle"\
                .format(results_dirname, result_str)
            assert os.path.exists(result_pickle)
            pickled_results = pickle.load(open(result_pickle, 'rb'))
            if alg_name == "VLP_opt":
                results[result_str] = pickled_results
            else:
                results[result_str] = [len(x) for x in pickled_results]

    # Extract JSON settings
    generation_count = config_data["generation_count"]
    generation_sizes = config_data["generation_sizes"]
    generation_probs = config_data["generation_probs"]

    # Plot results across all instances
    plot_synthetic_results(generation_count,
                           generation_sizes,
                           generation_probs,
                           figures_dirname,
                           all_instance_tuples,
                           alg_list,
                           results)

if __name__ == "__main__":
    json_filename = sys.argv[1]
    num_times = int(sys.argv[2])
    dist_name = sys.argv[3]
    param = None
    if len(sys.argv) > 4:
        param = sys.argv[4]

    run_with_config(json_filename, num_times, dist_name, param)

