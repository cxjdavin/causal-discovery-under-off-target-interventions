import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pickle
import subprocess
import sys

from typing import FrozenSet, Tuple, Dict, List, Union

from code.setup_instances import *
from code.solve_single_experiment import *

def plot_results(
        figures_dirname: str,
        all_instance_tuples:\
            List[Tuple[int, Tuple[str, List[Union[int, float]]], List[str], str]],
        alg_list: List[str],
        results: Dict[str, List[int]]
    ) -> None:
    """
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
    # Extract results into format for plotting
    nx_labels = []
    metrics = []
    keys = []
    mean_vals = defaultdict(list)
    std_vals = defaultdict(list)
    plot_title = None

    for instance_tuple in all_instance_tuples:
        num_times, exp_setting, nx_name = instance_tuple
        metrics = alg_list

        if plot_title is None:
            dist_name, params = exp_setting
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
            plot_title = "{0} (Avg. over {1} runs)".format(front, num_times)

        instance_str = str(instance_tuple)
        if results["{0}_VLP_opt".format(instance_str)][0] == 0:
            # Ignore graphs that are already fully oriented
            print("{0} is already fully oriented".format(nx_name))
        else:
            nx_labels.append(nx_name)
            keys.append(instance_str)
            for alg_name in alg_list:
                # Compute mean and std, then round up to 2 decimals
                result_str = "{0}_{1}".format(instance_str, alg_name)
                mean = np.ceil(np.mean(results[result_str]) * 100) / 100
                std = np.ceil(np.std(results[result_str]) * 100) / 100
                mean_vals[instance_str].append(mean)
                std_vals[instance_str].append(std)

    # Plot
    assert plot_title is not None
    fig, ax = plt.subplots(figsize=(16,12))
    bar_width = 0.15
    bar_offsets = np.arange(len(nx_labels))

    for idx, metric in enumerate(metrics):
        ax.bar(bar_offsets + idx * bar_width,
               [mean_vals[k][idx] for k in keys],
               yerr=[std_vals[k][idx] for k in keys],
               width=bar_width,
               align="center",
               label=metric,
               capsize=5)
    ax.set_xticks(bar_offsets + bar_width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(nx_labels)
    ax.set_ylabel("Number of interventions")
    ax.set_xlabel("Benchmark graphs")
    ax.set_title(plot_title)
    ax.legend()

    # Save plot
    plt.tight_layout()
    fname = "{0}_{1}_{2}".format(num_times, exp_setting, dist_name)
    plt.savefig("{0}/bnlearn_{1}.png".format(figures_dirname, fname),
                format="png",
                dpi=300,
                bbox_inches="tight")

    # Plot (without random)
    assert plot_title is not None
    fig, ax = plt.subplots(figsize=(16,12))
    bar_width = 0.15
    bar_offsets = np.arange(len(nx_labels))

    for idx, metric in enumerate(metrics):
        if metric == "random":
            continue
        ax.bar(bar_offsets + idx * bar_width,
               [mean_vals[k][idx] for k in keys],
               yerr=[std_vals[k][idx] for k in keys],
               width=bar_width,
               align="center",
               label=metric,
               capsize=5)
    ax.set_xticks(bar_offsets + bar_width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(nx_labels)
    ax.set_ylabel("Number of interventions")
    ax.set_xlabel("Benchmark graphs")
    ax.set_title(plot_title)
    ax.legend()

    # Save plot
    plt.tight_layout()
    fname = "{0}_{1}_{2}".format(num_times, exp_setting, dist_name)
    plt.savefig("{0}/bnlearn_no_random_{1}.png".format(figures_dirname, fname),
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

    # Plot results across all instances
    plot_results(figures_dirname,
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

