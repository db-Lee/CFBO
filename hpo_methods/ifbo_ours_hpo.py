import copy
import json
import logging
import math
import os
import time
from typing import Dict, List, Optional, Tuple
import pickle

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy

from surrogate_models.ifbo_ours import ifBO_Ours

class IFBO_OursAlgorithm:

    def __init__(
        self,
        y_0: float,    
        config_ckpt: dict,
        model_ckpt: str,
        hp_candidates: np.ndarray,        
        seed: int = 11,
        max_benchmark_epochs: int = 52,
        total_budget: int = 500,
        device: str = None,
        dataset_name: str = 'unknown',
        output_path: str = '.'
    ):

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)      

        if device is None:
            self.dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.dev = torch.device(device)

        self.hp_candidates = hp_candidates
        self.seed = seed
        self.max_benchmark_epochs = max_benchmark_epochs
        self.output_path = output_path
        self.dataset_name = dataset_name
        self.t_0 = torch.FloatTensor([0.0]).to(self.dev)
        self.y_0 = torch.FloatTensor([y_0]).to(self.dev)

        # the keys will be hyperparameter indices while the value
        # will be a list with all the budgets evaluated for examples
        # and with all performances for the performances
        self.num_hps = self.hp_candidates.shape[0]
        self.budget_spent = 0
        self.total_budget = total_budget
        self.performances = dict()
        self.max_score = 0.
        self.hps = []

        with open(config_ckpt, "r") as fb:
            config = json.load(fb)
        config['dim_x'] = self.hp_candidates.shape[1]        
        self.model = ifBO_Ours(
            model_ckpt,
            config,
            self.dev
        )

        self.initialized = False
        
        xt = []
        for hp_index in range(self.num_hps):
            xt.append(self.hp_candidates[hp_index])

        self.xt = torch.FloatTensor(xt).to(self.dev) # [num_hps, dim_x]
        tt = [ time_index+1 for time_index in range(self.max_benchmark_epochs) ]
        self.tt = torch.FloatTensor(tt)[:, None].to(self.dev) / self.max_benchmark_epochs # [max_benchmark_epochs, 1]

    def _prepare_train_dataset(self) -> Dict[str, torch.Tensor]:        
        context_indices = []
        if self.initialized:            
            xc, tc, yc = [], [], []
            for hp_index in self.performances:
                hp_candidate = self.hp_candidates[hp_index]
                performances = self.performances[hp_index]                

                for time_index, performance in enumerate(performances):
                    xc.append(hp_candidate) # [dim_x]
                    tc.append(time_index+1) # []
                    yc.append(performance) # []
                    context_indices.append([hp_index, time_index])

            xc = torch.FloatTensor(xc).to(self.dev) # [num_context, dim_x]
            tc = torch.FloatTensor(tc)[:, None].to(self.dev) / self.max_benchmark_epochs # [num_context, 1]
            yc = torch.FloatTensor(yc)[:, None].to(self.dev) # [num_context, 1]

        else:
            xc, tc, yc = None, None, None

        data = {
            't_0': self.t_0,
            'y_0': self.y_0,
            'xc': xc,
            'tc': tc,
            'yc': yc
        }
            
        return data
    
    def _prepare_test_dataset(self) -> Dict[str, torch.Tensor]:

        data = {
            'xt': self.xt,
            'tt': self.tt
        }

        return data

    def suggest(self):
        best_index = self.hpo()

        if best_index in self.performances:
            max_t = len(self.performances[best_index])
            next_t = max_t + 1
        else:
            next_t = 1

        return best_index, next_t

    def observe(
        self,
        hp_index: int,
        t: int,
        score: float
    ):        
        self.budget_spent += 1
        self.hps.append(hp_index)

        if score > self.max_score:
            self.max_score = score

        if hp_index in self.performances:
            self.performances[hp_index].append(score)
        else:            
            self.performances[hp_index] = [score]

        assert len(self.performances[hp_index]) == t

        self.log_performances()
        self.initialized = True

    @ torch.no_grad()
    def hpo(
        self
    ):  
        
        if self.initialized:
            present_b = []
            for hp_index in range(self.num_hps):
                if hp_index in self.performances:
                    present_b.append(len(self.performances[hp_index]))
                else:
                    present_b.append(0)
        else:
            present_b = [ 0 for _ in range(self.num_hps) ]

        h = np.random.randint(1, self.max_benchmark_epochs+1)
        tau = np.random.uniform(low=-4, high=-1)
        best_f = self.max_score + (10**tau)*(1.-self.max_score)

        pi = self.model.predict_pipeline(
            train_data=self._prepare_train_dataset(),
            test_data=self._prepare_test_dataset(),
            present_b=present_b,
            h=h,
            best_f=best_f
        ) # [num_hps]

        best_pi = 0.
        for hp_index in range(self.num_hps):            
            if pi[hp_index] > best_pi:
                if hp_index in self.performances:
                    if len(self.performances[hp_index]) < self.max_benchmark_epochs:
                        best_pi = pi[hp_index]
                        best_hp_index = hp_index
                else: 
                    best_pi = pi[hp_index]
                    best_hp_index = hp_index
        
        return best_hp_index


    def log_performances(self):
        with open(os.path.join(self.output_path, 'performances.json'), 'w') as fp:
            json.dump(self.performances, fp)
        with open(os.path.join(self.output_path, 'hps.json'), 'w') as fp:
            json.dump(self.hps, fp) 