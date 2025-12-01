from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from .api import API

class BaseBenchmark(ABC):
    
    def __init__(self, path_to_json_file: Optional[str] = None):
        if path_to_json_file is not None:
            self.api = self._load_api(path_to_json_file)

    def _load_api(self, path_to_json_file: str) -> API:
        api = API(
            data_dir=path_to_json_file,
        )
        return api

    def load_dataset_names(self) -> List[str]:
        return self.api.get_dataset_names()  
    
    def get_hyperparameter_candidates(self) -> np.array:
        return self.hp_candidates

    def get_init_performance(self) -> float:
        return self.init_value

    def get_performance(self, hp_index: int, budget: int) -> float:
        return self.data[hp_index][budget-1]

    def get_curve(self, hp_index: int, budget: int) -> List[float]:
        return self.data[hp_index][:budget]
    
    def set_dataset_name(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.get_data()
    
    @abstractmethod
    def get_data(self, *args, **kwargs) -> None:
        pass    
