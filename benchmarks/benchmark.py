import torch
from data.api import Benchmark

import torch

def pfn_normalize(lb=torch.tensor(float('-inf')), ub=torch.tensor(float('inf')), soft_lb=0.0, soft_ub=1.0, minimize=False):
    """
    LC-PFN curve prior assumes curves to be normalized within the range [0,1] and to be maximized.
    This function allows to normalize and denormalize data to fit this assumption.
    
    Parameters:
        lb (torch.Tensor): Lower bound of the data.
        ub (torch.Tensor): Upper bound of the data.
        soft_lb (float): Soft lower bound for normalization. Default is 0.0.
        soft_ub (float): Soft upper bound for normalization. Default is 1.0.
        minimize (bool): If True, the original curve is a minization. Default is False.
    
    Returns: Two functions for normalizing and denormalizing the data.
    """
    assert(lb <= soft_lb and soft_lb < soft_ub and soft_ub <= ub)
    # step 1: linearly transform [soft_lb,soft_ub] [-1,1] (where the sigmoid behaves approx linearly)
    #    2.0/(soft_ub - soft_lb)*(x - soft_lb) - 1.0
    # step 2: apply a vertically scaled/shifted the sigmoid such that [lb,ub] --> [0,1]

    def cinv(x):
        return 1-x if minimize else x

    def lin_soft(x):
        return 2/(soft_ub - soft_lb)*(x - soft_lb)-1

    def lin_soft_inv(y):
        return (y+1)/2*(soft_ub - soft_lb)+soft_lb
    
    try:
        if torch.exp(-lin_soft(lb)) > 1e300: raise RuntimeError
        # otherwise overflow causes issues, treat these cases as if the lower bound was -infinite
        # print(f"WARNING: {lb} --> NINF to avoid overflows ({np.exp(-lin_soft(lb))})")
    except RuntimeError:
        lb = torch.tensor(float('-inf'))
    if torch.isinf(lb) and torch.isinf(ub):
        return lambda x: cinv(1/(1 + torch.exp(-lin_soft(x)))), lambda y: lin_soft_inv(torch.log(cinv(y)/(1-cinv(y))))
    elif torch.isinf(lb):
        a = 1 + torch.exp(-lin_soft(ub))
        return lambda x: cinv(a/(1 + torch.exp(-lin_soft(x)))), lambda y: lin_soft_inv(torch.log((cinv(y)/a)/(1-(cinv(y)/a))))
    elif torch.isinf(ub):
        a = 1/(1-1/(1+torch.exp(-lin_soft(lb))))
        b = 1-a
        return lambda x: cinv(a/(1 + torch.exp(-lin_soft(x))) + b), lambda y: lin_soft_inv(torch.log(((cinv(y)-b)/a)/(1-((cinv(y)-b)/a))))
    else:
        a = (1 + torch.exp(-lin_soft(ub)) + torch.exp(-lin_soft(lb)) + torch.exp(-lin_soft(ub)-lin_soft(lb))) / (torch.exp(-lin_soft(lb)) - torch.exp(-lin_soft(ub)))
        b = - a / (1 + torch.exp(-lin_soft(lb)))
        return lambda x: cinv(a/(1 + torch.exp(-lin_soft(x))) + b), lambda y: lin_soft_inv(torch.log(((cinv(y)-b)/a)/(1-((cinv(y)-b)/a))))

class BaseBenchmark:

    def __init__(self, path_to_json_file: str):

        self.path_to_json_file = path_to_json_file

    def _load_benchmark(self):
        bench = Benchmark(
            data_dir=self.path_to_json_file,
        )

        return bench

    def load_dataset_names(self):
        return self.benchmark.get_dataset_names()  
    
    def get_hyperparameter_candidates(self):
        return self.hp_candidates

    def get_init_performance(self):
        return self.init_value

    def get_performance(self, hp_index: int, budget: int):
        return self.data[hp_index][budget-1]

    def get_curve(self, hp_index: int, budget: int):
        return self.data[hp_index][:budget]
