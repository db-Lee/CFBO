import argparse
import os
import random
import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style('white')

import torch

from benchmarks.lcbench import LCBench
from benchmarks.taskset import TaskSet
from benchmarks.pd1 import PD1
from benchmarks.odbench import ODBench
from data.meta_test_datasets import META_TEST_DATASET_DICT

from hpo_methods.our_hpo import OurAlgorithm
from utils import get_utility_function

from tqdm import trange
import json

import warnings
warnings.filterwarnings("ignore")

def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    U = get_utility_function(args.budget_limit, args.alpha, args.a, args.c)
        
    # data
    benchmark_name = args.benchmark_name
    if benchmark_name == 'lcbench':
        benchmark_data_path = os.path.join(args.data_dir, "data_2k.json")
    elif benchmark_name == 'taskset':
        benchmark_data_path = os.path.join(args.data_dir, 'taskset_chosen.json')
    elif benchmark_name == 'pd1':
        benchmark_data_path = os.path.join(args.data_dir, "pd1_preprocessed.json")
    elif benchmark_name == 'odbench':
        benchmark_data_path = os.path.join(args.data_dir, "od_datasets.json")
    else:
        raise NotImplementedError

    output_dir = os.path.join(
        args.output_dir,
        f'{benchmark_name}',
        args.dataset_name
    )
    os.makedirs(os.path.join(output_dir, args.dataset_name), exist_ok=True)
    benchmark_dict = {
        'lcbench': LCBench,
        'taskset': TaskSet,
        'pd1': PD1,
        'odbench': ODBench
    }
    benchmark = benchmark_dict[benchmark_name](benchmark_data_path, args.dataset_name)
    hp_candidates = benchmark.get_hyperparameter_candidates()

    # get u_max, u_min
    u_max, worst_performance = -10000, 10000  
    for config_id in range(benchmark.nr_hyperparameters):        
        curve = benchmark.get_curve(config_id, benchmark.max_budget)
        if curve[0] < worst_performance:
            worst_performance = curve[0]        
        for budget_idx, performance in enumerate(curve):
            u = U(budget_idx+1, performance)
            if u > u_max:
                u_max = u
    u_min = U(args.budget_limit, worst_performance)      

    algo = OurAlgorithm(
        U=U,
        beta=args.beta,        
        threshold=args.threshold, 
        y_0=benchmark.get_init_performance(),
        config_ckpt=args.config_ckpt,
        model_ckpt=args.model_ckpt,
        hp_candidates=hp_candidates,        
        max_benchmark_epochs=benchmark.max_budget,
        total_budget=args.budget_limit,
        device=device,
        dataset_name=args.dataset_name,
        output_path=output_dir,
        seed=args.seed
    )    

    incumbent = 0.
    trajectory = []
    utility_trajectory = []
    first_stop_sign = False
    stop_budget = args.budget_limit

    print(f"BO start, benchmark_name: {args.benchmark_name}, dataset_name: {args.dataset_name}, alpha: {args.alpha}, a: {args.a}, c: {args.c}") 
    for budget in trange(1, args.budget_limit+1):
        # suggest
        hp_index, t, stop_sign = algo.suggest()
        if stop_sign and not first_stop_sign:
            first_stop_sign = True
            stop_budget = budget - 1

        if first_stop_sign and args.stop:
            break
        
        # observe
        score = benchmark.get_performance(hp_index, t)
        algo.observe(hp_index, t, score)

        if score > incumbent:
            incumbent = score

        utility = U(budget, incumbent)
        trajectory.append(incumbent)
        utility_trajectory.append(utility)    

    utility_trajectory = np.array(utility_trajectory)    
    normalized_utility_trajectory = (u_max - utility_trajectory) / (u_max - u_min)

    print(f"BO end, stop_budget: {stop_budget}, normalize_utility: {normalized_utility_trajectory[stop_budget-1]}") 

    with open(os.path.join(output_dir, "trajectory.json"), "w") as fp:
        json.dump(trajectory, fp)

    with open(os.path.join(output_dir, "normalized_utility_trajectory.json"), "w") as fp:
        json.dump(normalized_utility_trajectory.tolist(), fp)
  
    with open(os.path.join(output_dir, "stop_budget.json"), "w") as fp:
        json.dump(stop_budget, fp)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='DyHPO experiments.',
    )    
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data',
    )
    parser.add_argument(
        '--benchmark_name',
        type=str,
        default='lcbench',
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='segment',
    )
    parser.add_argument(
        '--budget_limit',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.,
    )
    parser.add_argument(
        '--a',
        type=float,
        default=1.,
    )
    parser.add_argument(
        '--c',
        type=float,
        default=0.,
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=-1,
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.2,
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./output',
    )
    parser.add_argument(
        '--model_ckpt',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--config_ckpt',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--stop',
        action="store_true",
    )
    args = parser.parse_args()

    main(args)