from collections import OrderedDict
from typing import Dict, Any, List
from typing_extensions import override

import numpy as np
import torch

from .base import BaseBenchmark
from .utils import pfn_normalize, VALID_INDICES
from syne_tune.blackbox_repository import load_blackbox
from sklearn.preprocessing import MinMaxScaler

class PD1(BaseBenchmark):
    nr_hyperparameters = 240
    # Declaring the search space for PD1
    param_space = OrderedDict([
        ('lr_initial_value', [1.00564e-05, 9.7743115561, float, True]),
        ('lr_power', [0.1238067716, 1.9967467513, float, False]),
        ('lr_decay_steps_factor', [0.010543063200000001, 0.988565263, float, False]),
        ('one_minus_momentum', [0.0010070363999999943, 0.9999413886, float, True])
    ])
    log_indicator = [True, False, False, True]
    max_budget = 50

    def __init__(self, path_to_json_file: str, dataset_name: str):

        super().__init__()
        self.api = self._load_api()
        self.dataset_name = dataset_name
        self.dataset_names = self.load_dataset_names()

        hp_names = list(PD1.param_space.keys())
        hp_configs = self.api[dataset_name].hyperparameters.to_numpy()[VALID_INDICES]
        for index, hp_name in enumerate(hp_names):
            if PD1.log_indicator[index]:
                hp_configs[:, index] = np.log(hp_configs[:, index])
        self.hp_candidates = MinMaxScaler().fit_transform(hp_configs)
        self.get_data()    

    @ override
    def _load_api(self) -> Dict[str, Any]:
        return load_blackbox('pd1')
    
    @ override
    def load_dataset_names(self) -> List[str]:
        return list(self.api.keys())

    @ override
    def get_data(self) -> None:
        validation_error_rate = self.api[self.dataset_name].objectives_evaluations[:, :, :, 0]
        validation_error_rate = np.mean(validation_error_rate, axis=1)[VALID_INDICES]
        data = 1. - validation_error_rate
        data = self.linear_interpolation(data, PD1.max_budget+1)

        # init data
        init_value = np.mean(data[:, 0]).item()
        
        # data
        data = data[:, 1:]
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

    def linear_interpolation(
        self, original_array: np.array, new_length: int
    ) -> np.array:
        n, T = original_array.shape
        x_old = np.arange(T)
        x_new = np.linspace(0, T - 1, new_length)
        interpolated_array = np.zeros((n, new_length))
        for i in range(n):
            interpolated_array[i] = np.interp(x_new, x_old, original_array[i])
        return interpolated_array