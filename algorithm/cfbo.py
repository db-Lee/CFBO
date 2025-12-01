from typing import (
    Callable,
    Dict,
    Optional,
    Tuple
)
from typing_extensions import override
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import scipy.stats

from lcpfn import TransformerModel, get_bucket_limits, BarDistribution
from .base import BaseConfig, BaseAlgorithm


@dataclass
class CFBOConfig(BaseConfig):
    # model    
    dim_x: Optional[int] = None
    d_output: int = 1000
    d_model: int = 512
    nlayers: int = 12
    dropout: float = 0.2
    activation: str = "gelu"
    
    # hpo
    y_0: float = 0.
    beta: float = -1.    
    num_mc_samples: int = 1000
    batch_size: int = 500

class CFBO(BaseAlgorithm):
    
    def __init__(
        self,
        U: Callable,
        config: CFBOConfig,
        hp_candidates: Optional[np.ndarray] = None
    ):
        super().__init__(U, config, hp_candidates)
        self._initialize_model()
        self._precompute_tensors()
    
    @ override
    def _initialize_utility_function(self, U: Callable) -> None:
        self.U = U
        gamma = -np.log2(self.config.threshold)
        beta_exp = np.exp(self.config.beta)
        
        self.beta_dist = scipy.stats.beta(beta_exp, beta_exp)
        self.betacdf = lambda p: self.beta_dist.cdf(p) ** gamma
        
        self.u_max = -np.inf
        self.u_min = np.inf
    
    @ override
    def _initialize_model(self) -> None:        
        config = self.config
        
        # Setup criterion
        borders = get_bucket_limits(
            num_outputs=config.d_output,
            full_range=(0, 1)
        )
        self.criterion = BarDistribution(borders)
        
        # Initialize model
        self.model = TransformerModel(
            dim_x=self.hp_candidates.shape[1],
            d_output=config.d_output,
            d_model=config.d_model,
            dim_feedforward=2 * config.d_model,
            nlayers=config.nlayers,
            dropout=config.dropout,
            data_stats=None,
            activation="gelu",
            criterion=self.criterion
        ).to(self.dev)
        
        # Load checkpoint
        checkpoint = torch.load(
            self.config.model_ckpt, map_location="cpu")
        self.model.load_state_dict(checkpoint, strict=True)
        
        # Enable eval mode and gradient checkpointing for memory efficiency
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    @ override
    def _prepare_train_data(self) -> Dict[str, torch.Tensor]:
        if not self.initialized:
            return {
                't_0': self.t_0,
                'y_0': self.y_0,
                'xc': None,
                'tc': None,
                'yc': None
            }
        
        # Use list comprehension for better performance
        data_points = [
            (self.hp_candidates[hp_idx], t + 1, perf)
            for hp_idx, perfs in self.performances.items()
            for t, perf in enumerate(perfs)
        ]
        
        if not data_points:
            return {
                't_0': self.t_0,
                'y_0': self.y_0,
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
            't_0': self.t_0,
            'y_0': self.y_0,
            'xc': xc,
            'tc': tc,
            'yc': yc
        }
    
    @torch.no_grad()
    @ override
    def _prepare_test_data(self) -> Dict[str, torch.Tensor]:
        return {'xt': self.xt, 'tt': self.tt}
    
    def _get_performance_lengths(self) -> torch.Tensor:
        if self._cache_invalidated:
            self._performance_lengths_cache = torch.tensor(
                [len(self.performances.get(i, [])) for i in range(self.num_hps)],
                dtype=torch.float32,
                device=self.dev
            )
            self._cache_invalidated = False
        return self._performance_lengths_cache
    
    @torch.no_grad()
    @ override
    def hpo(self) -> Tuple[int, bool]:
        # Get model predictions
        num_samples = self.config.num_mc_samples
        sampled_graphs = self.predict(
            train_data=self._prepare_train_data(),
            test_data=self._prepare_test_data(),
            num_mc_samples=num_samples * 5
        )
        
        # Reshape and average for stability
        sampled_graphs = sampled_graphs.view(
            self.num_hps, 5, num_samples, self.config.max_benchmark_epochs
        ).mean(dim=1)
        
        # Current utility
        current_utility = self.U(self.budget_spent, self.max_score)
        
        # Get performance lengths
        max_t = self._get_performance_lengths()
        
        # Compute budget tensor
        budget = self._compute_budget_tensor_vectorized(max_t)
        
        # Create observation mask
        mask = self.idx_mask.expand(
            self.num_hps, num_samples, -1
        ) < max_t.view(-1, 1, 1)
        
        # Apply constraints
        sampled_graphs = torch.clamp_min(sampled_graphs, self.max_score)
        sampled_graphs[mask] = float('-inf')
        sampled_graphs = torch.cummax(sampled_graphs, dim=-1)[0]
        
        # Compute utilities and acquisition
        u = self.U(budget, sampled_graphs)
        advantage = F.relu(u - current_utility)
        
        # Compute acquisition function and probability
        acq = advantage.mean(dim=1).max(dim=-1)[0]
        prob = (u >= current_utility).float().mean(dim=1).max(dim=-1)[0]
        
        # Select best index
        best_index = self._select_best_index(acq, max_t)
        best_prob = prob[best_index].item() \
            if max_t[best_index] < self.config.max_benchmark_epochs else 0.0
        
        # Compute stopping criterion
        stop_sign = self._compute_stop_sign(current_utility, best_prob)
        
        self.probs.append(best_prob)
        return best_index, stop_sign
    
    def train(self) -> None:
        pass
    
    @torch.no_grad()
    def predict(
        self,
        train_data: Dict[str, torch.Tensor],
        test_data: Dict[str, torch.Tensor],
        num_mc_samples: int
    ) -> torch.Tensor:        
        # Transformer model prediction
        self.model.eval()
        
        # Prepare training data
        t_0 = train_data['t_0'][None, None, :]
        y_0 = train_data['y_0'][None, None, :]
        
        xc, tc, yc = None, None, None
        if train_data.get("xc") is not None:
            xc = train_data['xc'][None, :, :]
            tc = train_data['tc'][None, :, :]
            yc = train_data['yc'][None, :, :]
        
        # Prepare test data
        xt = test_data['xt']
        tt = test_data['tt']
        T = tt.shape[0]
        
        # Create DataLoader for batch processing
        if self.config.batch_size is None:
            batch_size = xt.shape[0]
        else:
            batch_size = self.config.batch_size 
        dl = DataLoader(
            TensorDataset(xt),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        logits_list = []
        
        for xt_batch, in dl:
            B = xt_batch.shape[0]
            
            # Expand dimensions efficiently
            xt_batch = xt_batch.unsqueeze(1).expand(-1, T, -1)
            tt_batch = tt.unsqueeze(0).expand(B, -1, -1)
            
            # Prepare context data
            if xc is None:
                context_data = (None, None, None)
            else:
                context_data = (
                    xc.expand(B, -1, -1),
                    tc.expand(B, -1, -1),
                    yc.expand(B, -1, -1)
                )
            
            # Forward pass
            logit = self.model(
                t_0.expand(B, -1, -1),
                y_0.expand(B, -1, -1),
                *context_data,
                xt_batch,
                tt_batch
            )
            
            logits_list.append(logit)
        
        # Concatenate and sample
        logits = torch.cat(logits_list, dim=0)
        sampled_graphs = self.criterion.sample(
            logits, num_samples=num_mc_samples
        ).permute(1, 0, 2)
        
        return sampled_graphs 
    
    def _precompute_tensors(self) -> None:
        # Initial values
        self.t_0 = torch.tensor([0.0], device=self.dev, dtype=torch.float32)
        self.y_0 = torch.tensor([self.config.y_0], device=self.dev, dtype=torch.float32)
        
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
        
        # Pre-compute base budget tensor
        self.base_budget = torch.arange(
            1, max_epochs + 1,
            device=self.dev,
            dtype=torch.float32
        )
        
        # Pre-compute index mask for efficient masking
        self.idx_mask = torch.arange(
            max_epochs,
            device=self.dev
        ).unsqueeze(0).unsqueeze(0)
    
    def _compute_budget_tensor_vectorized(self, max_t: torch.Tensor) -> torch.Tensor:
        max_epochs = self.config.max_benchmark_epochs
        num_samples = self.config.num_mc_samples
        
        # Initialize budget tensor
        budget = torch.zeros(
            self.num_hps, num_samples, max_epochs,
            device=self.dev, dtype=torch.float32
        )
        
        # Vectorized computation
        for i in range(self.num_hps):
            mt = int(max_t[i].item())
            if mt < max_epochs:
                remaining = max_epochs - mt
                budget[i, :, mt:] = (
                    self.base_budget[:remaining] + self.budget_spent
                ).unsqueeze(0)
        
        return budget
    
    def _select_best_index(self, acq: torch.Tensor, max_t: torch.Tensor) -> int:
        max_epochs = self.config.max_benchmark_epochs
        
        # Find valid candidates
        valid_mask = max_t < max_epochs
        
        if valid_mask.any():
            # Use masked selection
            masked_acq = torch.where(
                valid_mask,
                acq,
                torch.tensor(float('-inf'), device=self.dev)
            )
            return masked_acq.argmax().item()
        else:
            # All fully explored
            return acq.argmax().item()
    
    def get_best_configuration(self) -> Tuple[int, float]:
        if not self.performances:
            return -1, 0.0
        
        best_hp = -1
        best_score = -np.inf
        
        for hp_idx, scores in self.performances.items():
            max_score = max(scores) if scores else -np.inf
            if max_score > best_score:
                best_score = max_score
                best_hp = hp_idx
        
        return best_hp, best_score