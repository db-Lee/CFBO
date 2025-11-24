from copy import deepcopy
import logging
import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cat
import math

import gpytorch
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions import Categorical

import pfns4hpo

class IFBO:
    """
    The DyHPO DeepGP model.
    """
    def __init__(
        self,
        device: torch.device
    ):
        """
        The constructor for the DyHPO model.

        Args:
            configuration: The configuration to be used
                for the different parts of the surrogate.
            device: The device where the experiments will be run on.
            dataset_name: The name of the dataset for the current run.
            output_path: The path where the intermediate/final results
                will be stored.
            seed: The seed that will be used to store the checkpoint
                properly.
        """
        super(IFBO, self).__init__()
        self.dev = device
        self.batch_size = 500

        self.model = pfns4hpo.PFN_MODEL(name="bopfn_broken_unisep_1000curves_10params_2M").to(device)

    def sample(self, logits, num_samples=100):
        bucket_idx = Categorical(logits=logits).sample(torch.Size([num_samples,]))    
        sampled = []
        for b_idx in bucket_idx:
            bucket_values = self.model.model.criterion.borders[:-1] + self.model.model.criterion.bucket_widths*torch.rand_like(self.model.model.criterion.bucket_widths)
            sampled.append(bucket_values[b_idx])            
        
        return torch.stack(sampled, dim=0)
        
    def train_pipeline(self, data: Dict[str, torch.Tensor], load_checkpoint: bool = False):
        pass

    @ torch.no_grad()
    def predict_pipeline(
        self,
        train_data: Dict[str, torch.Tensor],
        test_data: Dict[str, torch.Tensor],
        present_b: list = [],
        h: int = 50,
        best_f: float = 0.0,
        sample: bool = False,
        num_mc_samples: int = 10
    ) -> torch.Tensor:

        xc = train_data['xc'][:, :] # num_context, dim_x
        tc = train_data['tc'][:, :] # num_context, 1
        xc = torch.cat([torch.zeros_like(tc), tc, xc], dim=-1)
        yc = train_data['yc'][:, :] # num_context, 1

        xt = test_data['xt'] # num_hps, dim_x
        tt = test_data['tt'] # max_benchmark_epochs, 1
        num_hps, dim_x = xt.shape
        max_benchmark_epochs = tt.shape[0]
        xt = xt[:, None, :].repeat(1, max_benchmark_epochs, 1).reshape(-1, dim_x) # num_hps*max_benchmark_epochs, dim_x
        tt = tt[None, :, :].repeat(num_hps, 1, 1).reshape(-1, 1) # num_hps*max_benchmark_epochs, 1
        
        xt = torch.cat([torch.zeros_like(tt), tt, xt], dim=-1)
        
        logits = self.model(xc, yc, xt) # num_hps*max_benchmark_epochs, num_logits
        logits = logits.reshape(num_hps, max_benchmark_epochs, -1)

        if sample:
            return self.sample(logits, num_samples=num_mc_samples).permute(1, 0, 2)
        else:
            pi = []
            for logit, b in zip(logits, present_b):
                b = min(b+h, max_benchmark_epochs)
                logit = logit[b-1] # num_logits
                pi.append(self.model.model.criterion.pi(logit, best_f, maximize=True).item())
            
            return pi