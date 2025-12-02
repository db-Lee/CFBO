import os
import argparse
import json
import warnings
from typing import Dict, Any, List

import numpy as np
import torch
from tqdm import trange

from benchmark import (
    LCBench, TaskSet, PD1, ODBench,
    META_TEST_DATASET_DICT
)
from algorithm import (
    DyHPOConfig, DyHPO,
    ifBOConfig, ifBO,
    CFBOConfig, CFBO
)
from utils import get_utility_function

warnings.filterwarnings("ignore")


class BenchmarkRunner:
    BENCHMARKS = {
        'lcbench': (LCBench, 'data_2k.json', 'segment'),
        'taskset': (TaskSet, 'taskset_chosen.json', 'rnn_text_classification_family_seed19'),
        'pd1': (PD1, None, 'imagenet_resnet_batch_size_512'),
        'odbench': (ODBench, 'od_datasets.json', 'd1_a1')
    }
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.utility_fn = get_utility_function(
            args.budget_limit, args.alpha, args.c)
        self._setup_benchmark()
        
    def _setup_benchmark(self):        
        benchmark_class, data_file, dummy_name = self.BENCHMARKS[self.args.benchmark_name]
        if data_file is not None:
            data_file = os.path.join(self.args.data_dir, data_file)
        self.benchmark = benchmark_class(data_file, dummy_name)
    
    def get_datasets(self) -> List[str]:
        if self.args.dataset_name == "all":            
            return META_TEST_DATASET_DICT[self.args.benchmark_name]
        return [self.args.dataset_name]
    
    def run_single_dataset(self, dataset_name: str) -> None:
        self.benchmark.set_dataset_name(dataset_name)
        
        # Setup output directory
        output_dir = os.path.join(
            self.args.output_dir,
            self.args.benchmark_name,
            dataset_name,
            f"seed{self.args.seed}"
        )
        os.makedirs(output_dir, exist_ok=True)
        
        # Compute utility bounds
        u_max, u_min = float('-inf'), float('inf')
        for config_id in range(self.benchmark.nr_hyperparameters):
            curve = self.benchmark.get_curve(config_id, self.benchmark.max_budget)
            u_min = min(u_min, self.utility_fn(self.args.budget_limit, curve[0]))
            for i, perf in enumerate(curve):
                u_max = max(u_max, self.utility_fn(i + 1, perf))
        
        # Initialize optimizer
        if self.args.algorithm == "dyhpo":
            config = DyHPOConfig(
                model_ckpt=self.args.model_ckpt,
                seed=self.args.seed,
                max_benchmark_epochs=self.benchmark.max_budget,
                total_budget=self.args.budget_limit,
                dataset_name=dataset_name,
                output_path=str(output_dir)
            )            
            ALGO_CLASS = DyHPO
        elif self.args.algorithm == "ifbo":
            config = ifBOConfig(
                model_ckpt=None,
                seed=self.args.seed,
                max_benchmark_epochs=self.benchmark.max_budget,
                total_budget=self.args.budget_limit,
                dataset_name=dataset_name,
                output_path=str(output_dir)
            )            
            ALGO_CLASS = ifBO
        elif self.args.algorithm == "cfbo":
            config = CFBOConfig(
                model_ckpt=self.args.model_ckpt,
                y_0=self.benchmark.get_init_performance(),
                seed=self.args.seed,
                max_benchmark_epochs=self.benchmark.max_budget,
                total_budget=self.args.budget_limit,
                dataset_name=dataset_name,
                output_path=str(output_dir)
            )            
            ALGO_CLASS = CFBO
        
        algo = ALGO_CLASS(self.utility_fn, config, self.benchmark.get_hyperparameter_candidates())
        
        # Run optimization
        incumbent, trajectory, utilities = 0.0, [], []
        stop_budget = self.args.budget_limit
        first_stop = False
                
        print(f"\n{'='*100}")
        desc = f"{self.args.benchmark_name}/{dataset_name}"
        desc += f" (B={self.args.budget_limit}, c={self.args.c}, α={self.args.alpha})"        
        for budget in trange(1, self.args.budget_limit + 1, desc=desc):
            hp_index, t, stop_sign = algo.suggest()
            
            if stop_sign and not first_stop:
                first_stop, stop_budget = True, budget - 1
            
            if first_stop and not self.args.not_stop:
                print(f"Early stop at budget {stop_budget}")
                break
            
            score = self.benchmark.get_performance(hp_index, t)
            algo.observe(hp_index, t, score)
            incumbent = max(incumbent, score)
            
            trajectory.append(incumbent)
            utilities.append(self.utility_fn(budget, incumbent))
        
        # Normalize utilities and prepare results
        utilities = np.array(utilities)
        norm_utilities = (u_max - utilities) / (u_max - u_min)
        
        results = {
            'trajectory': trajectory,
            'utility_trajectory': utilities.tolist(),
            'normalized_utility_trajectory': norm_utilities.tolist(),
            'stop_budget': stop_budget,
            'final_incumbent': incumbent,
            'final_normalized_utility': norm_utilities[
                min(stop_budget - 1, len(norm_utilities) - 1)],
            'u_max': u_max,
            'u_min': u_min
        }
        
        with open(
            os.path.join(output_dir, 'complete_results.json'), 'w'
        ) as f:
            json.dump(results, f, indent=4)
        
        print(f"Results saved to: {output_dir}")
        print(f"Final: incumbent={incumbent:.4f}, normalized_utility={results['final_normalized_utility']:.4f}")
    
    def run(self):
        datasets = self.get_datasets()
        
        print(f"\nRunning {len(datasets)} dataset(s): {datasets}")        
        for dataset in datasets:
            self.run_single_dataset(dataset)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CFBO Optimization')
    
    # Data and benchmark
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--benchmark_name', default='lcbench', 
                       choices=['lcbench', 'taskset', 'pd1', 'odbench'])
    parser.add_argument('--dataset_name', default='all')
    
    # Optimization parameters
    parser.add_argument('--algorithm', default='cfbo', 
                       choices=['dyhpo', 'ifbo', 'cfbo'])
    parser.add_argument('--budget_limit', type=int, default=300)
    parser.add_argument('--c', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=0.0)    
    
    # Paths and execution
    parser.add_argument('--output_dir', default='./results')
    parser.add_argument('--model_ckpt', default=None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--not_stop', action='store_true')
    
    args = parser.parse_args()
    
    BenchmarkRunner(args).run()