import os
from copy import deepcopy
from typing import (
    Callable,
    Dict,
    Optional,
    Tuple,
    Any
)
from typing_extensions import override
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm
import torch
import torch.nn as nn
from torch import cat
import gpytorch

from .base import BaseConfig, BaseAlgorithm


@dataclass
class DyHPOConfig(BaseConfig):
    # model
    nr_layers: int = 2
    layer1_units: int = 64
    layer2_units: int = 128
    layer3_units: Optional[int] = None
    layer4_units: Optional[int] = None
    layer5_units: Optional[int] = None
    cnn_nr_channels: int = 4
    cnn_kernel_size: int = 3
    
    # hpo
    batch_size: int = 64
    nr_epochs: int = 1000
    refine_epochs: int = 50
    nr_patience_epochs: int = 10
    learning_rate: float = 0.001
    initial_nr_points: int = 10


class FeatureExtractor(nn.Module):    
    def __init__(self, initial_features: int, config: DyHPOConfig):
        super(FeatureExtractor, self).__init__()
        
        self.config = config
        self.nr_layers = config.nr_layers
        self.act_func = nn.LeakyReLU()
        
        self.fc1 = nn.Linear(initial_features, config.layer1_units)
        self.bn1 = nn.BatchNorm1d(config.layer1_units)
        
        # Hidden layers
        for i in range(2, self.nr_layers):
            prev_units = getattr(config, f'layer{i-1}_units')
            curr_units = getattr(config, f'layer{i}_units')
            
            setattr(self, f'fc{i}', nn.Linear(prev_units, curr_units))
            setattr(self, f'bn{i}', nn.BatchNorm1d(curr_units))
        
        # Final layer (includes CNN features)
        prev_units = getattr(config, f'layer{self.nr_layers-1}_units')
        final_units = getattr(config, f'layer{self.nr_layers}_units')
        self.fc_final = nn.Linear(
            prev_units + config.cnn_nr_channels,
            final_units
        )
        
        # CNN for learning curve features
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                kernel_size=(config.cnn_kernel_size,),
                out_channels=config.cnn_nr_channels
            ),
            nn.AdaptiveMaxPool1d(1),
        )
    
    def forward(self, x, budgets, learning_curves):
        # Add budget dimension
        budgets = torch.unsqueeze(budgets, dim=1)
        x = cat((x, budgets), dim=1)
        
        # First layer
        x = self.fc1(x)
        x = self.act_func(self.bn1(x))
        
        # Hidden layers
        for i in range(2, self.nr_layers):
            fc = getattr(self, f'fc{i}')
            bn = getattr(self, f'bn{i}')
            x = self.act_func(bn(fc(x)))
        
        # Process learning curves
        learning_curves = torch.unsqueeze(learning_curves, 1)
        lc_features = self.cnn(learning_curves)
        lc_features = torch.squeeze(lc_features, 2)
        
        # Final layer with learning curve features
        x = cat((x, lc_features), dim=1)
        x = self.act_func(self.fc_final(x))
        
        return x


class GPRegressionModel(gpytorch.models.ExactGP):    
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
    ):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DyHPO(BaseAlgorithm):
        
    def __init__(
        self,
        U: Callable,
        config: DyHPOConfig,
        hp_candidates: Optional[np.ndarray] = None
    ):
        super().__init__(U, config, hp_candidates)
        self._initialize_model()
        self.restart = True
    
    @ override
    def _initialize_model(self) -> None:
        initial_features = self.hp_candidates.shape[1] + 1
        self.feature_extractor = FeatureExtractor(
            initial_features, self.config).to(self.dev)
        
        output_dim = getattr(
            self.config, f'layer{self.config.nr_layers}_units')
        train_x = torch.ones(output_dim, output_dim).to(self.dev)
        train_y = torch.ones(output_dim).to(self.dev)
        
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.dev)
        self.model = GPRegressionModel(train_x, train_y, self.likelihood).to(self.dev)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model).to(self.dev)
        
        ckpt_dir = self.config.model_ckpt
        if ckpt_dir is not None:
            self.model.load_state_dict(
                torch.load(os.path.join(ckpt_dir, "model.pt"), map_location="cpu")
            )
            self.likelihood.load_state_dict(
                torch.load(os.path.join(ckpt_dir, "likelihood.pt"), map_location="cpu")
            )
            self.feature_extractor.load_state_dict(
                torch.load(os.path.join(ckpt_dir, "feature_extractor.pt"), map_location="cpu")
            )
    
    @torch.no_grad()
    @ override
    def _prepare_train_data(self) -> Dict[str, torch.Tensor]:
        if not self.performances:
            return {}
        
        X_list, budget_list, curve_list, y_list = [], [], [], []
        
        for hp_idx, perfs in self.performances.items():
            for t, perf in enumerate(perfs):
                X_list.append(self.hp_candidates[hp_idx])
                budget_list.append(t + 1)
                # Pad learning curve to fixed length
                curve = perfs[:t+1] + [0] * (self.config.max_benchmark_epochs - t - 1)
                curve_list.append(curve[:self.config.max_benchmark_epochs])
                y_list.append(perf)
        
        return {
            'X_train': torch.tensor(X_list, dtype=torch.float32, device=self.dev),
            'train_budgets': torch.tensor(budget_list, dtype=torch.float32, device=self.dev),
            'train_curves': torch.tensor(curve_list, dtype=torch.float32, device=self.dev),
            'y_train': torch.tensor(y_list, dtype=torch.float32, device=self.dev)
        }
    
    @torch.no_grad()
    @ override 
    def _prepare_test_data(self) -> Dict[str, torch.Tensor]:
        X_test, test_budgets, test_curves = [], [], []
        
        for hp_idx in range(self.num_hps):
            perfs = self.performances.get(hp_idx, [])
            next_t = len(perfs) + 1
            
            if next_t <= self.config.max_benchmark_epochs:
                X_test.append(self.hp_candidates[hp_idx])
                test_budgets.append(next_t)
                # Pad learning curve
                curve = perfs + [0] * (self.config.max_benchmark_epochs - len(perfs))
                test_curves.append(curve[:self.config.max_benchmark_epochs])
        
        if not X_test:
            return {}
        
        return {
            'X_test': torch.tensor(X_test, dtype=torch.float32, device=self.dev),
            'test_budgets': torch.tensor(test_budgets, dtype=torch.float32, device=self.dev),
            'test_curves': torch.tensor(test_curves, dtype=torch.float32, device=self.dev)
        }
    
    @torch.no_grad()
    @ override
    def hpo(self) -> Tuple[int, bool]:
        if not self.initialized:            
            best_index = np.random.randint(0, self.num_hps)
            stop_sign = False
            
            return best_index, stop_sign
            
        # Prepare data for all candidates
        train_data = self._prepare_train_data()
        test_data = self._prepare_test_data()
        
        # Get predictions
        means, stds = self.predict(train_data, test_data)
        
        # valid_hp_indices
        valid_hp_indices = []
        for hp_idx in range(self.num_hps):
            perfs = self.performances.get(hp_idx, [])
            next_t = len(perfs) + 1
            
            if next_t <= self.config.max_benchmark_epochs:
                valid_hp_indices.append(hp_idx)
        
        # select best
        max_acq = - np.inf
        for hp_idx, mean, std in zip(valid_hp_indices, means, stds):            
            acq_value = self._compute_acquisition(
                self.max_score, mean.item(), std.item())
            if acq_value > max_acq:
                best_index = hp_idx
                max_acq = acq_value
        
        # Compute stopping criterion
        stop_sign = self._compute_stop_sign()
        
        return best_index, stop_sign
    
    @ override
    def train(self) -> None:
        self.iterations += 1
        
        train_data = self._prepare_train_data()
        if self.restart:
            self._initialize_model()
            nr_epochs = self.config.nr_epochs
            if self.config.initial_nr_points <= self.iterations:
                self.restart = False
        else:
            nr_epochs = self.config.refine_epochs
            
        
        self.model.train()
        self.likelihood.train()
        self.feature_extractor.train()
        
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': self.config.learning_rate},
            {'params': self.feature_extractor.parameters(), 'lr': self.config.learning_rate}
        ])
        
        training_errored = False
        initial_state = self.get_state()
        for _ in range(nr_epochs):
            # Skip if only one example (BatchNorm issue)
            if train_data['X_train'].size(0) == 1:
                continue
            
            optimizer.zero_grad()
            
            # Forward pass
            projected_x = self.feature_extractor(
                train_data['X_train'],
                train_data['train_budgets'],
                train_data['train_curves']
            )
            
            self.model.set_train_data(projected_x, train_data['y_train'], strict=False)
            output = self.model(projected_x)
            try:
                # Compute loss
                loss = -self.mll(output, self.model.train_targets)
                loss.backward()
                optimizer.step()
            except Exception:
                self.restart = True
                training_errored = True
                break
            
        if training_errored:
            self.load_state(initial_state)
    
    @ torch.no_grad()
    def predict(
        self,
        train_data: Dict[str, torch.Tensor],
        test_data: Dict[str, torch.Tensor]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions for test configurations."""
        if not train_data or not test_data:
            return np.array([]), np.array([])
        
        self.model.eval()
        self.feature_extractor.eval()
        self.likelihood.eval()
        
        # Project training data
        projected_train_x = self.feature_extractor(
            train_data['X_train'],
            train_data['train_budgets'],
            train_data['train_curves']
        )
        
        self.model.set_train_data(
            inputs=projected_train_x,
            targets=train_data['y_train'],
            strict=False
        )
        
        # Project test data
        projected_test_x = self.feature_extractor(
            test_data['X_test'],
            test_data['test_budgets'],
            test_data['test_curves']
        )
        
        # Predict
        preds = self.likelihood(self.model(projected_test_x))
        
        means = preds.mean.detach().cpu().numpy().reshape(-1)
        stds = preds.stddev.detach().cpu().numpy().reshape(-1)
        
        return means, stds
    
    def _compute_acquisition(
        self,
        best_value: float,
        mean: float,
        std: float,
        explore_factor: Optional[float] = 0.25,
        acq_fc: str = 'ei',
    ) -> float:
        if acq_fc == 'ei':
            if std == 0:
                return 0
            z = (mean - best_value) / std
            acq_value = (mean - best_value) * norm.cdf(z) + std * norm.pdf(z)
        elif acq_fc == 'ucb':
            acq_value = mean + explore_factor * std
        elif acq_fc == 'thompson':
            acq_value = np.random.normal(mean, std)
        elif acq_fc == 'exploit':
            acq_value = mean
        else:
            raise NotImplementedError(
                f'Acquisition function {acq_fc} has not been'
                f'implemented',
            )

        return acq_value
    
    def load_state(self, state: Dict[str, Dict]) -> None:   
        self.model.load_state_dict(state['gp_state_dict'])
        self.feature_extractor.load_state_dict(state['feature_extractor_state_dict'])
        self.likelihood.load_state_dict(state['likelihood_state_dict'])

    def get_state(self) -> Dict[str, Dict]:
        current_state = {
            'gp_state_dict': deepcopy(self.model.state_dict()),
            'feature_extractor_state_dict': deepcopy(self.feature_extractor.state_dict()),
            'likelihood_state_dict': deepcopy(self.likelihood.state_dict()),
        }
        return current_state