from collections import OrderedDict
from typing import List

import math
import numpy as np
import torch

from benchmarks.benchmark import BaseBenchmark, pfn_normalize
from syne_tune.blackbox_repository import load_blackbox
from sklearn.preprocessing import MinMaxScaler

VALID_INDICES = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 17, 18, 24, 25, 26, 28, 29, 30, 31, 33, 35, 37, 42, 43, 44, 46, 53, 56, 58, 61, 63, 64, 68, 72, 74, 75, 76, 78, 79, 82, 83, 85, 87, 89, 91, 93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 106, 107, 108, 109, 110, 111, 112, 113, 114, 117, 120, 122, 123, 124, 125, 127, 128, 129, 132, 133, 134, 136, 137, 139, 140, 141, 143, 144, 145, 147, 148, 149, 151, 152, 154, 155, 157, 159, 160, 161, 162, 167, 169, 171, 172, 173, 174, 176, 177, 178, 180, 181, 185, 186, 187, 188, 189, 190, 192, 193, 194, 196, 198, 199, 206, 207, 208, 214, 215, 216, 217, 218, 220, 221, 222, 223, 224, 226, 227, 230, 231, 232, 235, 237, 238, 243, 244, 245, 247, 249, 252, 253, 254, 256, 257, 258, 261, 262, 264, 266, 268, 271, 272, 273, 274, 276, 277, 278, 279, 280, 281, 282, 284, 285, 286, 288, 291, 294, 295, 299, 300, 301, 303, 304, 309, 310, 312, 313, 314, 316, 317, 321, 322, 323, 324, 325, 326, 327, 328, 329, 331, 333, 334, 335, 337, 338, 340, 341, 343, 344, 345, 346, 348, 350, 352, 353, 354, 355, 356, 358, 361, 366, 370, 371, 372, 373, 376, 377, 378, 380, 382, 383, 385, 386, 388, 390, 391, 393, 394, 395, 396, 397, 399]

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

        super().__init__(path_to_json_file)
        self.benchmark = self._load_benchmark()
        self.dataset_name = dataset_name
        self.dataset_names = self.load_dataset_names()

        hp_names = list(PD1.param_space.keys())
        hp_configs = self.benchmark[dataset_name].hyperparameters.to_numpy()[VALID_INDICES]
        for index, hp_name in enumerate(hp_names):
            if PD1.log_indicator[index]:
                hp_configs[:, index] = np.log(hp_configs[:, index])
        self.hp_candidates = MinMaxScaler().fit_transform(hp_configs)
        self.get_data()    

    def _load_benchmark(self):
        return load_blackbox('pd1')
    
    def load_dataset_names(self):
        return list(self.benchmark.keys())

    def get_data(self):

        validation_error_rate = self.benchmark[self.dataset_name].objectives_evaluations[:, :, :, 0]
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

    def linear_interpolation(self, original_array, new_length):
        n, T = original_array.shape
        x_old = np.arange(T)
        x_new = np.linspace(0, T - 1, new_length)
        interpolated_array = np.zeros((n, new_length))
        for i in range(n):
            interpolated_array[i] = np.interp(x_new, x_old, original_array[i])
        return interpolated_array

    def set_dataset_name(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.get_data()