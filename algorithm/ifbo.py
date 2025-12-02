from typing import (
    Callable,
    Dict,
    Optional,
    Tuple,
    List
)
from typing_extensions import override
from dataclasses import dataclass

import numpy as np
import torch

from ifbo.surrogate import FTPFN
from .base import BaseConfig, BaseAlgorithm


@dataclass
class ifBOConfig(BaseConfig):
    # hpo
    h_min: int = 1
    tau_min: float = -4.0
    tau_max: float = -1.0


class ifBO(BaseAlgorithm):
    
    def __init__(
        self,
        U: Callable,
        config: ifBOConfig,
        hp_candidates: Optional[np.ndarray] = None
    ):
        super().__init__(U, config, hp_candidates)
        self._initialize_model()
        self._precompute_tensors()
    
    @override
    def _initialize_model(self) -> None:
        self.model = FTPFN(version="0.0.1").to(self.dev)
        self.criterion = self.model.model.criterion
        
        # Enable eval mode for inference
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    @override
    def _prepare_train_data(self) -> Dict[str, torch.Tensor]:
        if not self.initialized:
            return {
                'xc': None,
                'tc': None,
                'yc': None
            }
        
        # Collect all observed data points
        data_points = [
            (self.hp_candidates[hp_idx], t + 1, perf)
            for hp_idx, perfs in self.performances.items()
            for t, perf in enumerate(perfs)
        ]
        
        if not data_points:
            return {
                'xc': None,
                'tc': None,
                'yc': None
            }
        
        # Unpack and convert to tensors
        xc_list, tc_list, yc_list = zip(*data_points)
        
        xc = torch.tensor(xc_list, dtype=torch.float32, device=self.dev)
        tc = torch.tensor(tc_list, dtype=torch.float32, device=self.dev).unsqueeze(1)
        tc = tc / self.config.max_benchmark_epochs
        yc = torch.tensor(yc_list, dtype=torch.float32, device=self.dev).unsqueeze(1)
        
        return {
            'xc': xc,
            'tc': tc,
            'yc': yc
        }
    
    @torch.no_grad()
    @override
    def _prepare_test_data(self) -> Dict[str, torch.Tensor]:
        return {'xt': self.xt, 'tt': self.tt}
    
    @torch.no_grad()
    @override
    def hpo(self) -> Tuple[int, bool]:        
        if not self.initialized:            
            best_index = np.random.randint(0, self.num_hps)
            stop_sign = False
            
            return best_index, stop_sign
                
        # Sample hyperparameters for PI calculation
        h = np.random.randint(self.config.h_min, self.config.max_benchmark_epochs + 1)
        tau = np.random.uniform(low=self.config.tau_min, high=self.config.tau_max)
        best_f = self.max_score + (10**tau) * (1.0 - self.max_score)
        
        # Get current performance lengths
        present_b = [
            len(self.performances.get(i, [])) for i in range(self.num_hps)
        ]
        
        # Get PI for all configurations
        pi = self.predict(
            train_data=self._prepare_train_data(),
            test_data=self._prepare_test_data(),
            present_b=present_b,
            h=h,
            best_f=best_f
        )
        
        # Select best configuration based on PI
        best_index = self._select_best_index(pi, present_b)
        
        # Compute stopping criterion
        stop_sign = self._compute_stop_sign()
        
        return best_index, stop_sign
    
    @ override
    def train(self) -> None:
        pass
    
    @torch.no_grad()
    def predict(
        self,
        train_data: Dict[str, torch.Tensor],
        test_data: Dict[str, torch.Tensor],
        present_b: List[int],
        h: int,
        best_f: float,
    ) -> List[float]:
        
        self.model.eval()
        
        xc = train_data.get('xc')
        tc = train_data.get('tc')
        yc = train_data.get('yc')
        
        xt = test_data['xt']  # num_hps, dim_x
        tt = test_data['tt']  # T, 1
        
        num_hps, dim_x = xt.shape
        T = tt.shape[0]
        
        # Add placeholder dimension for PFN model format
        xc = torch.cat([torch.zeros_like(tc), tc, xc], dim=-1)
        
        # Prepare test data for all HP/time combinations
        xt_expanded = xt[:, None, :].repeat(1, T, 1).reshape(-1, dim_x)
        tt_expanded = tt[None, :, :].repeat(num_hps, 1, 1).reshape(-1, 1)
        xt_formatted = torch.cat([torch.zeros_like(tt_expanded), tt_expanded, xt_expanded], dim=-1)
        
        # Get model predictions
        logits = self.model(xc, yc, xt_formatted)
        
        # Reshape logits
        logits = logits.reshape(num_hps, T, -1)
        
        # Calculate PI for each configuration
        pi = []
        for logit, b in zip(logits, present_b):
            # Determine evaluation budget
            eval_budget = min(b + h, T)
            
            if eval_budget > b and eval_budget <= T:
                # Get logits at the evaluation point
                logit_at_budget = logit[eval_budget - 1]  # -1 for 0-indexing
                
                # Calculate probability of improvement
                pi_value = self.criterion.pi(
                    logit_at_budget, 
                    best_f, 
                    maximize=True
                ).item()
            else:
                # Configuration already fully evaluated or invalid budget
                pi_value = 0.0
            
            pi.append(pi_value)
        
        return pi   
    
    def _precompute_tensors(self) -> None:
        # Convert hp_candidates to tensor
        self.xt = torch.tensor(
            self.hp_candidates,
            dtype=torch.float32,
            device=self.dev
        )
        
        # Pre-compute normalized time indices
        max_epochs = self.config.max_benchmark_epochs
        self.tt = torch.arange(
            1, max_epochs + 1,
            dtype=torch.float32,
            device=self.dev
        ).unsqueeze(1) / max_epochs
        
    def _select_best_index(
        self, 
        pi: List[float],
        present_b: List[int]
    ) -> int:
        
        best_pi = 0.0
        best_index = 0
        
        for hp_index in range(self.num_hps):
            # Only consider configurations that haven't been fully evaluated
            if present_b[hp_index] < self.config.max_benchmark_epochs:
                if pi[hp_index] > best_pi:
                    best_pi = pi[hp_index]
                    best_index = hp_index
        
        if best_pi == 0.0:
            best_index = np.argmax(pi)
        
        return best_index