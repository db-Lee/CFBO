
from abc import ABC, abstractmethod
import os
import json
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple
)
from dataclasses import dataclass

import random
import numpy as np
import torch

@dataclass
class BaseConfig:
    model_ckpt: str
    threshold: float = 0.2
    seed: int = 11
    max_benchmark_epochs: int = 52
    total_budget: int = 300
    device: Optional[str] = None
    dataset_name: str = '.'
    output_path: str = '.'

class BaseAlgorithm(ABC):
    
    def __init__(
        self,
        U: Callable,
        config: BaseConfig,
        hp_candidates: Optional[np.ndarray] = None
    ):
        
        self.config = config
        self._setup_device()
        self._set_random_seeds()
        
        self._initialize_utility_function(U)
        self._initialize_hyperparameters(hp_candidates)
        self._initialize_tracking()
        
    def _setup_device(self) -> None:
        if self.config.device:
            self.dev = torch.device(self.config.device)
        else:
            self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _set_random_seeds(self) -> None:
        seed = self.config.seed
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
    def _initialize_utility_function(self, U: Callable) -> None:
        self.U = U        
        self.u_max = -np.inf
        self.u_min = np.inf
    
    def _initialize_hyperparameters(self, hp_candidates: np.ndarray) -> None:
        self.hp_candidates = hp_candidates
        self.num_hps = len(hp_candidates)
    
    def _initialize_tracking(self) -> None:
        self.budget_spent = 0
        self.performances: Dict[int, List[float]] = {}
        self.max_score = 0.0
        self.hps: List[int] = []
        self.probs: List[float] = []
        self.iterations = 0
        self.initialized = False
        
        # Cache for frequently accessed values
        self._performance_lengths_cache = {}
        self._cache_invalidated = True

    @ abstractmethod
    def _initialize_model(self) -> None:
        pass
        
    @ abstractmethod
    def _prepare_train_data(self) -> Dict[str, torch.Tensor]:
        pass
    
    @ abstractmethod
    def _prepare_test_data(self) -> Dict[str, torch.Tensor]:
        pass
    
    def observe(self, hp_index: int, t: int, score: float) -> None:
        self.budget_spent += 1
        self.hps.append(hp_index)
        self._cache_invalidated = True
        
        # Update max score
        if score > self.max_score:
            self.max_score = score
        
        # Store performance
        if hp_index in self.performances:
            self.performances[hp_index].append(score)
        else:
            self.performances[hp_index] = [score]
        
        # Update utility bounds
        if not self.initialized:
            self.u_min = self.U(self.config.total_budget, score)
        
        u_max = self.U(self.budget_spent, self.max_score)
        if u_max > self.u_max:
            self.u_max = u_max
        
        # Validation
        assert len(self.performances[hp_index]) == t, \
            f"Mismatch: expected {t} observations, got {len(self.performances[hp_index])}"
            
        # Train model
        self.train()
        
        self.log_performances()
        self.initialized = True
        
    def suggest(self) -> Tuple[int, int, bool]:
        best_index, stop_sign = self.hpo()
        next_t = len(self.performances.get(best_index, [])) + 1
        
        return best_index, next_t, stop_sign
    
    @ abstractmethod
    def hpo(self) -> Tuple[int, bool]:
        pass
    
    @ abstractmethod
    def train(self) -> None:
        pass
        
    def _compute_stop_sign(
        self,
        current_utility: Optional[float] = None,
        best_prob: Optional[float] = None
    ) -> bool:
        if current_utility is None:
            current_utility = self.U(self.budget_spent, self.max_score)
        
        if np.isclose(self.u_max, self.u_min):
            lhs = -np.inf
        else:
            lhs = (self.u_max - current_utility) / (self.u_max - self.u_min)
        
        # CFBO
        if hasattr(self, "betacdf") and best_prob is not None:
            rhs = self.betacdf(best_prob)
        else:
            rhs = self.config.threshold
        return lhs >= rhs
    
    def log_performances(self) -> None:
        output_path = self.config.output_path
        # Save performances
        with open(
            os.path.join(output_path, 'performances.json'), 'w'
        ) as f:
            json.dump(self.performances, f, indent=4)
        
        # Save hyperparameter sequence
        with open(
            os.path.join(output_path, 'hps.json'), 'w'
        ) as f:
            json.dump(self.hps, f, indent=4)
        
        # Save probabilities
        if hasattr(self, "probs"):
            with open(
                os.path.join(output_path, 'probs.json'), 'w'
            ) as f:
                json.dump(self.probs, f, indent=4)