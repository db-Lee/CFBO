from collections import OrderedDict
from typing_extensions import override

import math
import numpy as np
import torch

from .base import BaseBenchmark
from .utils import pfn_normalize
from sklearn.preprocessing import MinMaxScaler

class TaskSet(BaseBenchmark):
    nr_hyperparameters = 1000
    # Declaring the search space for TaskSet
    param_space = OrderedDict([
        ('learning_rate', [1.02694156e-08, 9.68279083e+00, float, True]),
        ('beta1', [4.80782964e-04, 9.99998998e-01, float, False]),
        ('beta2', [1.83174047e-03, 9.99899930e-01, float, False]),
        ('epsilon', [1.04631967e-10, 9.75014812e+02, float, True]),
        ('l1', [1.00736426e-08, 9.63026461e+00, float, True]),
        ('l2', [1.00620944e-08, 9.31468283e+00, float, True]),
        ('linear_decay', [1.00272257e-07, 9.98544408e-05, float, True]),
        ('exponential_decay', [1.00340111e-06, 9.89970235e-04, float, True]),
    ])
    log_indicator = [True, False, False, True, True, True, True, True]
    max_budget = 50

    def __init__(self, path_to_json_file: str, dataset_name: str):

        super().__init__(path_to_json_file)
        self.benchmark = self._load_api()
        self.dataset_name = dataset_name
        self.dataset_names = self.load_dataset_names()

        hp_names = list(TaskSet.param_space.keys())
        hp_configs = []
        for i in range(TaskSet.nr_hyperparameters):
            hp_config = []
            config = self.benchmark.query(
                dataset_name=self.dataset_name,
                tag='config',
                config_id=i,
            )
            for index, hp_name in enumerate(hp_names):
                if TaskSet.log_indicator[index]:
                    hp_config.append(math.log(config[hp_name]))
                else:
                    hp_config.append(config[hp_name])
            hp_configs.append(hp_config)

        hp_configs = np.array(hp_configs)
        self.hp_candidates = MinMaxScaler().fit_transform(hp_configs)
        self.get_data()

    @ override
    def get_data(self) -> None:
        # init data
        init_value = 0.
        for hp_index in range(0, TaskSet.nr_hyperparameters):
            val_curve = self.benchmark.query(
                dataset_name=self.dataset_name,
                config_id=hp_index,
                tag='Train/val_balanced_accuracy',
            )
            init_value += val_curve[0] / TaskSet.nr_hyperparameters
        
        # data
        data = []
        for hp_index in range(TaskSet.nr_hyperparameters):
            val_curve = self.benchmark.query(
                dataset_name=self.dataset_name,
                config_id=hp_index,
                tag='Train/val_balanced_accuracy',
            )
            val_curve = val_curve[1:]
            data.append(val_curve)
        data = torch.FloatTensor(data)

        # transform
        transform, _ = pfn_normalize(lb=torch.tensor(0.), ub=torch.tensor(float("inf")), soft_lb=0., soft_ub=init_value, minimize=True)
        self.data = transform(data).numpy().tolist()
        self.init_value = transform(torch.tensor(init_value)).item()

