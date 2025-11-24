from collections import OrderedDict
from typing import List

import math
import numpy as np
import torch

from benchmarks.benchmark import BaseBenchmark, pfn_normalize
from sklearn.preprocessing import MinMaxScaler

class LCBench(BaseBenchmark):

    nr_hyperparameters = 2000
    # Declaring the search space for LCBench
    param_space = OrderedDict([
        ('batch_size', [16, 512, int, True]),
        ('learning_rate', [0.0001, 0.1, float, True]),
        ('momentum', [0.1, 0.99, float, False]),
        ('weight_decay', [0.00001, 0.1, float, False]),
        ('num_layers', [1, 5, int, False]),
        ('max_units', [64, 1024, int, True]),
        ('max_dropout', [0.0, 1.0, float, False]),
    ])
    log_indicator = [True, True, False, False, False, True, False]
    max_budget = 51

    def __init__(self, path_to_json_file: str, dataset_name: str):

        super().__init__(path_to_json_file)
        self.benchmark = self._load_benchmark()
        self.dataset_name = dataset_name
        self.dataset_names = self.load_dataset_names()

        hp_names = list(LCBench.param_space.keys())
        hp_configs = []
        for i in range(LCBench.nr_hyperparameters):
            hp_config = []
            config = self.benchmark.query(
                dataset_name=self.dataset_name,
                tag='config',
                config_id=i,
            )
            for index, hp_name in enumerate(hp_names):
                if LCBench.log_indicator[index]:
                    hp_config.append(math.log(config[hp_name]))
                else:
                    hp_config.append(config[hp_name])
            hp_configs.append(hp_config)

        hp_configs = np.array(hp_configs)
        self.hp_candidates = MinMaxScaler().fit_transform(hp_configs)
        self.get_data()

    def get_data(self):
        # init data
        init_value = 0.
        for hp_index in range(0, LCBench.nr_hyperparameters):
            val_curve = self.benchmark.query(
                dataset_name=self.dataset_name,
                config_id=hp_index,
                tag='Train/val_balanced_accuracy',
            )
            init_value += val_curve[0] / LCBench.nr_hyperparameters
        
        # data
        data = []
        for hp_index in range(LCBench.nr_hyperparameters):
            val_curve = self.benchmark.query(
                dataset_name=self.dataset_name,
                config_id=hp_index,
                tag='Train/val_balanced_accuracy',
            )
            val_curve = val_curve[1:]
            data.append(val_curve)
        data = torch.FloatTensor(data)

        # min, max  
        min_value, max_value = torch.min(data).item(), torch.max(data).item()
        if init_value < min_value:
            min_value = init_value
        if init_value > max_value:
            max_value = init_value

        # transform
        transform, _ = pfn_normalize(lb=torch.tensor(0.), ub=torch.tensor(1.), soft_lb=min_value, soft_ub=max_value, minimize=False)
        self.data = transform(data).numpy().tolist()
        self.init_value = transform(torch.tensor(init_value)).item()

    def set_dataset_name(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.get_data()

