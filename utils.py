from typing import (
    Callable,
    Union,
    Optional,
    Dict,
    Any,
    Union,
    Tuple
)

import os
from datetime import datetime
import math
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

import wandb

from benchmark import LCBench, TaskSet, PD1, ODBench, META_TEST_DATASET_DICT

def get_utility_function(
    budget_limit: int,
    alpha: float,
    c: float = 1.,
    bias: float = 0.
) -> Callable[
        [
            Union[float, np.ndarray, torch.Tensor],
            Union[float, np.ndarray, torch.Tensor]
        ],
        Union[float, np.ndarray]
    ]:
    
    def U(
        budget: Union[float, np.ndarray],
        performance: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        u = performance - alpha*(budget / budget_limit)**c + bias
        if type(u) == float:
            if budget > budget_limit:
                u = float('-inf')
        else:
            u[budget > budget_limit] = float('-inf')
        return u
    
    return U

def get_dataset(
    data_dir: str, benchmark_name: str
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    # data
    if benchmark_name == 'lcbench':
        benchmark_data_path = os.path.join(data_dir, "data_2k.json")
        dummy_dataset_name = 'segment'
    elif benchmark_name == 'taskset':
        benchmark_data_path = os.path.join(data_dir, "taskset_chosen.json")
        dummy_dataset_name = 'rnn_text_classification_family_seed19'
    elif benchmark_name == 'pd1':
        benchmark_data_path = os.path.join(data_dir, "pd1_preprocessed.json")
        dummy_dataset_name = 'imagenet_resnet_batch_size_512'
    elif benchmark_name == 'odbench':
        benchmark_data_path = os.path.join(data_dir, "od_datasets.json")
        dummy_dataset_name = 'd1_a1'
    else:
        raise NotImplementedError

    benchmark_dict = {
        'lcbench': LCBench,
        'taskset': TaskSet,
        'pd1': PD1,
        'odbench': ODBench
    }    
    
    benchmark = benchmark_dict[benchmark_name](benchmark_data_path, dummy_dataset_name)
    dataset_names = benchmark.load_dataset_names()

    meta_train = {}
    meta_test = {}
    for dataset_name in tqdm(dataset_names):
        benchmark.set_dataset_name(dataset_name)
        hp_candidates = benchmark.get_hyperparameter_candidates().tolist()

        x, y = [], []

        for hp_index, hp_candidate in enumerate(hp_candidates):
            curve = benchmark.get_curve(hp_index, benchmark.max_budget)
            x.append(hp_candidate) # [ dim_x ]
            y.append(curve) # [ max_budget ]

        y_0 = torch.FloatTensor([benchmark.get_init_performance()]) # [ 1 ]
        x = torch.FloatTensor(x) # [ num_hps, dim_x ]
        y = torch.FloatTensor(y) # [ num_hps, max_budget ]
        if dataset_name in META_TEST_DATASET_DICT[benchmark_name]:
            meta_test[dataset_name] = {"y_0": y_0, "x": x, "y": y}
        else:
            meta_train[dataset_name] = {"y_0": y_0, "x": x, "y": y}

    return meta_train, meta_test

# https://github.com/automl/lcpfn/blob/ba892f6f451027f69c50edf00c765ded98c75d30/lcpfn/utils.py#L13
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

class Logger:
    def __init__(
        self,
        exp_name: str,
        save_dir: Optional[str] = None,
        save_only_last: bool = True,
        print_every: int = 100,
        save_every: int = 100,
        total_step: int = 0,
        print_to_stdout: bool = True,
        wandb_entity: Optional[str] = None,
        wandb_project_name: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
    ):
        if save_dir is not None:
            self.save_dir = save_dir
            os.makedirs(self.save_dir, exist_ok=True)
        else:
            self.save_dir = None

        self.print_every = print_every
        self.save_every = save_every
        self.save_only_last = save_only_last
        self.step_count = 0
        self.total_step = total_step
        self.print_to_stdout = print_to_stdout

        self.writer = None
        self.start_time = None
        self.groups = dict()
        self.models_to_save = dict()
        self.objects_to_save = dict()
        wandb.init(
            entity=wandb_entity,
            project=wandb_project_name,
            name=exp_name,
            reinit=True
        )
        wandb.config.update(wandb_config)

    def register_model_to_save(self, model: nn.Module, name: str) -> None:
        assert name not in self.models_to_save.keys(), "Name is already registered."

        self.models_to_save[name] = model

    def register_object_to_save(self, object: Any, name: str) -> None:
        assert name not in self.objects_to_save.keys(), "Name is already registered."

        self.objects_to_save[name] = object

    def step(self) -> None:
        self.step_count += 1
        if self.step_count % self.print_every == 0:
            if self.print_to_stdout:
                self.print_log(self.step_count, self.total_step, elapsed_time=datetime.now() - self.start_time)
            self.write_log(self.step_count)

        if self.step_count % self.save_every == 0:
            if self.save_only_last:
                self.save_models()
                self.save_objects()
            else:
                self.save_models(self.step_count)
                self.save_objects(self.step_count)

    def meter(
        self, group_name: str, log_name: str, value: Union[int, float, torch.Tensor]
    ) -> None:
        if group_name not in self.groups.keys():
            self.groups[group_name] = dict()

        if log_name not in self.groups[group_name].keys():
            self.groups[group_name][log_name] = Accumulator()

        self.groups[group_name][log_name].update_state(value)

    def reset_state(self) -> None:
        for _, group in self.groups.items():
            for _, log in group.items():
                log.reset_state()

    def print_log(
        self, step: int, total_step: int, elapsed_time: Optional[float] = None
    ) -> None:
        print(f"[Step {step:5d}/{total_step}]", end="  ")

        for name, group in self.groups.items():
            print(f"({name})", end="  ")
            for log_name, log in group.items():
                res = log.result()
                if res is None:
                    continue

                if "acc" in log_name.lower():
                    print(f"{log_name} {res:.2f}", end=" | ")
                else:
                    print(f"{log_name} {res:.4f}", end=" | ")

        if elapsed_time is not None:
            print(f"(Elapsed time) {elapsed_time}")
        else:
            print()

    def write_log(self, step: int) -> None:
        log_dict = {}
        for group_name, group in self.groups.items():
            for log_name, log in group.items():
                res = log.result()
                if res is None:
                    continue
                log_dict["{}/{}".format(group_name, log_name)] = res
        wandb.log(log_dict, step=step)

        self.reset_state()

    def write_log_individually(self, name: str, value: Any, step: int) -> None:
        wandb.log({name: value}, step=step)

    def save_models(self, suffix: Optional[str] = None) -> None:
        if self.save_dir is None:
            return
        for name, model in self.models_to_save.items():
            _name = name
            if suffix:
                _name += f"_{suffix}"
            torch.save(model.state_dict(), os.path.join(self.save_dir, f"{_name}.pt"))

            if self.print_to_stdout:
                print(f"{name} is saved to {self.save_dir}")

    def save_objects(self, suffix: Optional[str] = None) -> None:
        if self.save_dir is None:
            return

        for name, obj in self.objects_to_save.items():
            _name = name
            if suffix:
                _name += f"_{suffix}"
            torch.save(obj, os.path.join(self.save_dir, f"{_name}.pt"))

            if self.print_to_stdout:
                print(f"{name} is saved to {self.save_dir}")

    def start(self) -> None:
        if self.print_to_stdout:
            print("Training starts!")
        self.start_time = datetime.now()

    def finish(self) -> None:
        if self.step_count % self.save_every != 0:
            if self.save_only_last:
                self.save_models()
                self.save_objects()
            else:
                self.save_models(self.step_count)
                self.save_objects(self.step_count)

        if self.print_to_stdout:
            print("Training is finished!")
        wandb.join()

class Accumulator:
    def __init__(self):
        self.data = 0
        self.num_data = 0

    def reset_state(self) -> None:
        self.data = 0
        self.num_data = 0

    @ torch.no_grad()
    def update_state(
        self, tensor: Union[int, float, torch.Tensor]
    ) -> None:
        self.data += tensor
        self.num_data += 1

    def result(self) -> Optional[float]:
        if self.num_data == 0:
            return None
        data = self.data.item() if hasattr(self.data, 'item') else self.data
        return float(data) / self.num_data
