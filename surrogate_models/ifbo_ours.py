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

from lcpfn.transformer import TransformerModel
from lcpfn.bar_distribution import get_bucket_limits, BarDistribution

class ifBO_Ours:
    """
    The DyHPO DeepGP model.
    """
    def __init__(
        self,
        model_ckpt: str,
        configuration: Dict,
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
        super(ifBO_Ours, self).__init__()
        self.dev = device
        self.batch_size = 500

        borders = get_bucket_limits(num_outputs=configuration['d_output'], full_range=(0,1))
        criterion = BarDistribution(borders)        

        # model and opt
        self.model = TransformerModel(
            dim_x=configuration['dim_x'],
            d_output=configuration['d_output'],
            d_model=configuration['d_model'],
            dim_feedforward=2*configuration['d_model'],
            nlayers=configuration['nlayers'],
            dropout=configuration['dropout'],
            data_stats=None,
            activation="gelu",
            criterion=criterion
        ).to(device)
        self.model.load_state_dict(
            torch.load(model_ckpt, map_location="cpu"), strict=True)
        
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
        self.model.eval()

        t_0 = train_data['t_0'][None, None, :] # 1, 1, 1
        y_0 = train_data['y_0'][None, None, :] # 1, 1, 1

        if train_data["xc"] is None:
            xc, tc, yc = None, None, None
        else:
            xc = train_data['xc'][None, :, :] # 1, num_context, dim_x
            tc = train_data['tc'][None, :, :] # 1, num_context, 1
            yc = train_data['yc'][None, :, :] # 1, num_context, 1

        xt = test_data['xt'] # num_hps, dim_x
        tt = test_data['tt'] # max_benchmark_epochs, 1
        dim_x, T = xt.shape[-1], tt.shape[0]
        dl = DataLoader(TensorDataset(xt), batch_size=self.batch_size, shuffle=False)

        logits = []
        for xt_batch, in dl:
            B = xt_batch.shape[0]
            xt_batch = xt_batch[:, None, :].repeat(1, T, 1) # B, T, dim_x
            tt_batch = tt[None, :, :].repeat(B, 1, 1) # B, T, 1

            if xc is None:
                logit = self.model(
                    t_0.repeat(B, 1, 1), 
                    y_0.repeat(B, 1, 1), 
                    None, 
                    None, 
                    None, 
                    xt_batch,
                    tt_batch                    
                ) # B, T, d_output                
            else:
                logit = self.model(
                    t_0.repeat(B, 1, 1), 
                    y_0.repeat(B, 1, 1), 
                    xc.repeat(B, 1, 1), 
                    tc.repeat(B, 1, 1), 
                    yc.repeat(B, 1, 1), 
                    xt_batch,
                    tt_batch 
                ) # B, T, d_output

            logits.append(logit)
        
        logits = torch.cat(logits, dim=0) # num_hps, T, d_output 

        if sample:
            return self.model.criterion.sample(logits, num_samples=num_mc_samples).permute(1, 0, 2)
        else:
            pi = []
            for logit, b in zip(logits, present_b):
                b = min(b+h, T)
                logit = logit[b-1] # num_logits
                pi.append(self.model.criterion.pi(logit, best_f, maximize=True).item())
            
            return pi