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

from surrogate_models.ours import Ours

class OurAlgorithm:

    def __init__(
        self,
        U,
        beta: float,
        threshold: float,
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

        self.U = U
        gamma = -np.log2(threshold)
        self.betacdf = lambda p: scipy.stats.beta.cdf(p, np.exp(beta), np.exp(beta))**gamma
        self.u_max = -np.inf
        self.u_min = np.inf

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
        self.num_mc_samples = 1000
        self.max_score = 0.
        self.hps = []
        self.probs = []

        if model_ckpt is None or model_ckpt == "None" or model_ckpt == "none": 
            self.use_ifbo = True
        else:
            self.use_ifbo = False
        if not self.use_ifbo:
            with open(config_ckpt, "r") as fb:
                config = json.load(fb)
            config['dim_x'] = self.hp_candidates.shape[1]
        
        if self.use_ifbo:
            from surrogate_models.ifbo import IFBO
            self.model = IFBO(device)
        else:
            self.model = Ours(
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
        if self.initialized:
            best_index, stop_sign = self.hpo()
        else:
            if self.use_ifbo:
                best_index, stop_sign = np.random.randint(0, self.num_hps), False
            else:
                best_index, stop_sign = self.hpo()

        if best_index in self.performances:
            max_t = len(self.performances[best_index])
            next_t = max_t + 1
        else:
            next_t = 1

        return best_index, next_t, stop_sign

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

        if not self.initialized:
            self.u_min = self.U(self.total_budget, score)
        u_max = self.U(self.budget_spent, self.max_score)
        if u_max > self.u_max:
            self.u_max = u_max            

        assert len(self.performances[hp_index]) == t

        self.log_performances()
        self.initialized = True

    @ torch.no_grad()
    def hpo(
        self
    ):  
        import time

        sampled_graphs = self.model.predict_pipeline(
            train_data=self._prepare_train_dataset(),
            test_data=self._prepare_test_dataset(),
            num_mc_samples=self.num_mc_samples*5,
            sample=True
        ) # [num_hps, 5*num_mc_samples, max_benchmark_epochs]
        sampled_graphs = sampled_graphs.reshape(
            self.num_hps, 5, self.num_mc_samples, self.max_benchmark_epochs)
        sampled_graphs = sampled_graphs.mean(dim=1) # [num_hps, num_mc_samples, max_benchmark_epochs]
        present_utility = self.U(self.budget_spent, self.max_score)

        max_t = torch.tensor(
            [len(self.performances[hp_index]) if hp_index in self.performances else 0 for hp_index in range(self.num_hps)],
            dtype=torch.float32,
            device=self.dev,
        )
        base_budget = torch.arange(1, self.max_benchmark_epochs + 1, device=self.dev) + self.budget_spent
        budget = torch.stack([
            torch.cat([torch.zeros(int(max_t_), device=self.dev), base_budget[:self.max_benchmark_epochs - int(max_t_)]])
            for max_t_ in max_t
        ], dim=0)[:, None, :].repeat(1, self.num_mc_samples, 1)
        idx = torch.arange(self.max_benchmark_epochs, device=self.dev)[None, None, :].repeat(self.num_hps, self.num_mc_samples, 1)
        mask = (idx < max_t.view(-1, 1, 1))

        # we only consider cumulative best performance so far
        sampled_graphs[sampled_graphs < self.max_score] = self.max_score
        sampled_graphs[mask] = float('-inf')
        sampled_graphs = torch.cummax(sampled_graphs, dim=-1)[0]

        # num_hps, num_mc_samples, postfix_len
        u = self.U(budget, sampled_graphs)         
        # num_hps
        acq = torch.max(torch.mean(F.relu(u - present_utility), dim=1), dim=-1)[0]
        # num_hps
        prob = torch.max(torch.mean((u >= present_utility).float(), dim=1), dim=-1)[0]

        best_index = torch.argmax(acq, dim=0).item()
        best_prob = prob[best_index].item()
        if max_t[best_index].item() >= self.max_benchmark_epochs:
            best_prob = 0.
            # algorithm should stop, but we run it for analysis
            for topk in range(2, self.max_benchmark_epochs+1):
                best_index = torch.topk(acq, k=topk, dim=0)[1][-1].item()
                if max_t[best_index].item() < self.max_benchmark_epochs:
                    break

        if self.u_max == self.u_min: # there are no penalty
            LHS = - np.inf
        else:
            LHS = (self.u_max - present_utility) / (self.u_max - self.u_min)
        RHS = self.betacdf(best_prob)
        stop_sign = LHS >= RHS

        self.probs.append(best_prob)

        return best_index, stop_sign

    def log_performances(self):
        with open(os.path.join(self.output_path, 'performances.json'), 'w') as fp:
            json.dump(self.performances, fp)
        with open(os.path.join(self.output_path, 'hps.json'), 'w') as fp:
            json.dump(self.hps, fp)
        with open(os.path.join(self.output_path, 'probs.json'), 'w') as fp:
            json.dump(self.probs, fp)