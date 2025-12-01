from typing import (
    Dict,
    Iterator,
    Any,
    Tuple,
    Optional
)

import math
import numpy as np

import torch
from torch.utils.data import (
    Dataset,
    TensorDataset,
    DataLoader
)

class InfIterator:
    def __init__(self, dataloader: DataLoader):        
        self.dataloader = dataloader
        self.iterator = iter(self.dataloader)
    
    def __iter__(self) -> Iterator[Any]:
        return self
    
    def __next__(self) -> Dict[str, torch.LongTensor]:        
        try:
            batch = next(self.iterator)            
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)            
        return batch
    
    def __len__(self) -> int:        
        return len(self.dataloader)

class HPODataset(Dataset):
    
    def __init__(
            self, 
            data_dict: Dict[str, torch.Tensor],
            meta_batch_size: int = 4, 
            batch_size: int = 2048, 
            prior_batch_size: int = 256,
            max_context: int = 100,
            meta_mixup_coeff: float = 0.0, 
            hparam_mixup_coeff: float = 0.0,
            device: torch.device = torch.device("cpu")
        ):

        super(HPODataset, self).__init__()

        self.meta_batch_size = meta_batch_size
        self.batch_size = batch_size
        self.prior_batch_size = prior_batch_size
        self.device = device
        self.meta_mixup_coeff = meta_mixup_coeff
        self.hparam_mixup_coeff = hparam_mixup_coeff

        self.dataset_names = list(data_dict.keys())
        self.num_datasets = len(self.dataset_names)
        self.num_hps = data_dict[self.dataset_names[0]]["x"].shape[0]                
        self.dim_x = data_dict[self.dataset_names[0]]["x"].shape[1]        
        self.max_budget = data_dict[self.dataset_names[0]]["y"].shape[-1]

        total_x, total_y, total_y_0 = [], [], []
        for dataset_name in self.dataset_names:
            # [ 1, 1 ]
            y_0 = data_dict[dataset_name]["y_0"][None, :].to(device)
            # [ num_hps, max_budget, d_x ]
            x = data_dict[dataset_name]["x"][:, None, :].repeat(1, self.max_budget, 1).to(device)
            # [ num_hp, max_budget, 1 ]
            y = data_dict[dataset_name]["y"][:, :, None].to(device)

            total_x.append(x); total_y.append(y); total_y_0.append(y_0)

        # [num_datasets, 1, 1]
        self.total_y_0 = torch.stack(total_y_0, dim=0)
        # [num_datasets, num_hp, max_budget, d_x]
        self.total_x = torch.stack(total_x, dim=0)
        # [num_datasets, num_hp, max_budget, 1]
        self.total_y = torch.stack(total_y, dim=0)
        self.total_y[self.total_y < 0] = 0.0

        # [1, 1]
        self.t_0 = torch.tensor([[0.]]).float().to(device)
        # [max_budget, 1]
        self.t = torch.arange(1, self.max_budget+1).float()[:, None].to(device) / self.max_budget
        # [num_hps, max_budget, 1]
        self.total_t = self.t[None, :, :].repeat(self.num_hps, 1, 1)

        self.num_context_range = np.arange(0, max_context+1).tolist()
        if self.meta_mixup_coeff > 0.:
            self.meta_mixup_dist = torch.distributions.beta.Beta(
                torch.tensor(self.meta_mixup_coeff, dtype=torch.float, device=device),
                torch.tensor(self.meta_mixup_coeff, dtype=torch.float, device=device)
            )
        if self.hparam_mixup_coeff > 0.:
            self.hparam_mixup_dist = torch.distributions.beta.Beta(
                torch.tensor(self.hparam_mixup_coeff, dtype=torch.float, device=device),
                torch.tensor(self.hparam_mixup_coeff, dtype=torch.float, device=device)
            )

    @ torch.no_grad()
    def generate_random_y(
        self, num_samples: int = 10000
    ) -> torch.Tensor:
        y = torch.FloatTensor([]).to(self.device)

        while len(y) < num_samples:
            total_y_0, _, total_y = self.mixup()
            y_ = torch.cat([total_y.reshape(-1), total_y_0.reshape(-1)])
            y = torch.cat([y, y_])

        y = y[torch.randperm(len(y))][:num_samples]

        return y
    
    @ property
    def all_y(self) -> torch.Tensor:
        y = torch.cat([self.total_y.reshape(-1), self.total_y_0.reshape(-1)])
        y = y[torch.randperm(len(y))]
        return y

    @ property
    def data_stats(self) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, 
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        # [dim_x]
        x_mean = torch.mean(self.total_x[:, :, 0, :], dim=[0, 1])
        x_std = torch.std(self.total_x[:, :, 0, :], dim=[0, 1])

        return (
            x_mean,
            x_std,
            torch.FloatTensor([0.5]),
            torch.FloatTensor([math.sqrt(1/12)]),
            torch.FloatTensor([0.5]),
            torch.FloatTensor([math.sqrt(1/12)])
        )

    @ torch.no_grad()
    def sample_graph(self, hp_index: int, num_context: int) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        t_0, y_0, xc, tc, yc, xt, tt, yt = [], [], [], [], [], [], [], []

        t_0 = self.t_0[None, :, :].repeat(self.num_datasets, 1, 1) # num_datasets, 1, 1
        y_0 = self.total_y_0 # num_datasets, 1, 1
        xc = self.total_x[:, hp_index, :num_context, :] # num_datasets, num_context, d_x
        tc = self.t[None, :num_context, :].repeat(self.num_datasets, 1, 1) # num_datasets, num_context, 1
        yc = self.total_y[:, hp_index, :num_context, :] # num_datasets, num_context, 1
        xt = self.total_x[:, hp_index, :, :] # num_datasets, max_budget, d_x
        tt = self.t[None, :, :].repeat(self.num_datasets, 1, 1) # num_datasets, max_budget, 1
        yt = self.total_y[:, hp_index, :, :] # num_datasets, max_budget, 1

        return t_0, y_0, xc, tc, yc, xt, tt, yt

    @ torch.no_grad()
    def mixup(self) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        total_y_0, total_x, total_y = self.total_y_0, self.total_x, self.total_y

        """Meta Mixup"""
        if self.meta_mixup_coeff > 0.0:
            shuffled_dataset_indices = torch.randperm(self.num_datasets)
            betas = self.meta_mixup_dist.sample(torch.Size([self.num_datasets,]))

            # [num_datasets, 1, 1]
            total_y_0 = betas[:, None, None]*total_y_0 + \
                (1.-betas[:, None, None])*total_y_0[shuffled_dataset_indices, :, :]

            # [num_datasets, num_hps, max_budget, 1]
            total_y = betas[:, None, None, None]*total_y + \
                (1.-betas[:, None, None, None])*total_y[shuffled_dataset_indices, :, :, :]

        """Hparam Mixup"""
        if self.hparam_mixup_coeff > 0.0:
            shuffled_hp_indices = torch.randperm(self.num_hps)
            betas = self.hparam_mixup_dist.sample(torch.Size([self.num_hps,]))

            # [num_datasets, num_hps, max_budget, d_x]
            total_x = betas[None, :, None, None]*total_x + \
                (1.-betas[None, :, None, None])*total_x[:, shuffled_hp_indices, :, :]
            # [num_datasets, num_hps, max_budget, 1]
            total_y = betas[None, :, None, None]*total_y + \
                (1.-betas[None, :, None, None])*total_y[:, shuffled_hp_indices, :, :]

        return total_y_0, total_x, total_y

    @ torch.no_grad()
    def sample(self, num_context: Optional[int] = None) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        if num_context is None:
            num_context = np.random.choice(self.num_context_range)

        """Mixup"""
        total_y_0, total_x, total_y = self.mixup()
        self.total_y_0_mixup, self.total_x_mixup, self.total_y_mixup = total_y_0, total_x, total_y

        """Sample Meta Batch"""
        dataset_indices = torch.randperm(self.num_datasets)[:self.meta_batch_size]

        t_0 = self.t_0[None, :, :].repeat(self.meta_batch_size, 1, 1) # [meta_batch_size, 1, 1]
        y_0 = total_y_0[dataset_indices, :, :] # [meta_batch_size, 1, 1]

        total_x = total_x[dataset_indices, :, :, :] # [meta_batch_size, num_hps, max_budget, d_x]
        total_x = total_x.reshape(self.meta_batch_size, self.num_hps*self.max_budget, self.dim_x)

        total_t = self.total_t.reshape(self.num_hps*self.max_budget, 1)

        total_y = total_y[dataset_indices, :, :, :] # [meta_batch_size, num_hps, max_budget, 1]
        total_y = total_y.reshape(self.meta_batch_size, self.num_hps*self.max_budget, 1)

        """Sampling Context and Target Indices for Each Dataset"""   
        xc, tc, yc, xt, tt, yt = [], [], [], [], [], []
        for dataset_index in range(self.meta_batch_size):                  
            
            indices = torch.randperm(self.num_hps*self.max_budget)
            context_indices, target_indices = indices[:num_context], indices[num_context:num_context+self.batch_size]

            xc.append(total_x[dataset_index, context_indices, :]) # [num_context, d_x]
            tc.append(total_t[context_indices, :]) # [num_context, 1]
            yc.append(total_y[dataset_index, context_indices, :]) # [num_context, 1]
            xt.append(total_x[dataset_index, target_indices, :]) # [batch_size, d_x]
            tt.append(total_t[target_indices, :]) # [batch_size, 1]
            yt.append(total_y[dataset_index, target_indices, :]) # [batch_size, 1]

        """Construct Dataset"""
        t_0 = self.t_0[None, :, :].repeat(self.meta_batch_size, 1, 1) # [meta_batch_size, 1, 1] 
        y_0 = total_y_0[dataset_indices, :, :] # [meta_batch_size, 1, 1] 
        xc = torch.stack(xc, dim=0) # [meta_batch_size, num_context, d_x]
        tc = torch.stack(tc, dim=0) # [meta_batch_size, num_context, 1]
        yc = torch.stack(yc, dim=0) # [meta_batch_size, num_context, 1]
        xt = torch.stack(xt, dim=0) # [meta_batch_size, batch_size, d_x]
        tt = torch.stack(tt, dim=0) # [meta_batch_size, batch_size, 1]
        yt = torch.stack(yt, dim=0) # [meta_batch_size, batch_size, 1]

        return t_0, y_0, xc, tc, yc, xt, tt, yt

    @ torch.no_grad()
    def sample_prior(self, num_context: Optional[int] = None) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        if num_context is None:
            num_context = np.random.choice([ _ for _ in range(self.max_budget) ])

        """Mixup"""
        total_y_0, total_x, total_y = self.total_y_0_mixup, self.total_x_mixup, self.total_y_mixup

        indices = torch.randperm(self.num_datasets*self.num_hps)
        batch_indices = indices[:self.prior_batch_size]

        # [prior_batch_size, 1, 1]
        t_0 = self.t_0[None, :, :].repeat(self.prior_batch_size, 1, 1) 
        # [num_datasets, num_hps, 1, 1]
        y_0 = total_y_0[:, :, :, None].repeat(1, self.num_hps, 1, 1)
        # [num_datasets*num_hps, 1, 1]
        y_0 = y_0.reshape(self.num_datasets*self.num_hps, 1, 1)
        # [prior_batch_size, 1, 1]
        y_0 = y_0[batch_indices, :, :]

        # [num_datasets*num_hps, max_budget, d_x]
        total_x = total_x.reshape(self.num_datasets*self.num_hps, self.max_budget, self.dim_x)
        # [prior_batch_size, max_budget, d_x]
        x = total_x[batch_indices, :, :]
        # [prior_batch_size, max_budget, 1]
        t = self.t[None, :, :].repeat(self.prior_batch_size, 1, 1)
        # [num_datasets*num_hps, max_budget, 1]
        total_y = total_y.reshape(self.num_datasets*self.num_hps, self.max_budget, 1)
        # [prior_batch_size, max_budget, d_x]
        y = total_y[batch_indices, :, :]

        xc = x[:, :num_context, :]
        tc = t[:, :num_context, :]
        yc = y[:, :num_context, :]
        xt = x[:, num_context:, :]
        tt = t[:, num_context:, :]
        yt = y[:, num_context:, :]

        return t_0, y_0, xc, tc, yc, xt, tt, yt

class HPOPlainSampler:
    
    def __init__(
        self,
        data_dict: Dict[str, torch.Tensor],
        batch_size: int,
        device: torch.device = torch.device("cpu")
    ):
        super(HPOPlainSampler, self).__init__()
        self.batch_size = batch_size
        self.device = device
        
        self.dataset_names = list(data_dict.keys())
        self.num_datasets = len(self.dataset_names)
        self.num_hps = data_dict[self.dataset_names[0]]["x"].shape[0]
        self.max_budget = data_dict[self.dataset_names[0]]["y"].shape[-1]

        self.data_dict = data_dict
        x, t, lc, y = [], [], [], []
        for dataset_name in self.dataset_names:
            for hp_index in range(self.num_hps):
                x_ = data_dict[dataset_name]["x"][hp_index]
                y_ = data_dict[dataset_name]["y"][hp_index]
                for budget in range(1, self.max_budget+1):
                    x.append(x_)
                    t.append(budget)             
                    lc.append(torch.cat([y_[:budget-1], torch.tensor([0.]*(self.max_budget-budget+1))], dim=0))
                    y.append(y_[budget-1])
        x, t, lc, y = torch.stack(x, dim=0), torch.tensor(t), torch.stack(lc, dim=0), torch.stack(y, dim=0)

        self.iterator = InfIterator(
            DataLoader(TensorDataset(x, t, lc, y), batch_size=self.batch_size, shuffle=True, drop_last=True)
        )

    @ torch.no_grad()
    def sample_graph(self, dataset_name: str, hp_index: int)-> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        x, t, lc, y = self.data_dict[dataset_name]["x"], self.data_dict[dataset_name]["t"], self.data_dict[dataset_name]["lc"], self.data_dict[dataset_name]["y"]
        x = x.reshape(self.num_hps, self.max_budget, -1)
        t = t.reshape(self.num_hps, self.max_budget)
        lc = lc.reshape(self.num_hps, self.max_budget, self.max_budget)
        y = y.reshape(self.num_hps, self.max_budget)

        x = x[hp_index].to(self.device)
        t = t[hp_index].to(self.device)
        lc = lc[hp_index].to(self.device)
        y = y[hp_index].to(self.device)

        return x, t, lc, y
    
    @ torch.no_grad()
    def sample(self) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        x, t, lc, y = next(self.iterator)

        x = x.to(self.device)
        t = t.to(self.device)
        lc = lc.to(self.device)
        y = y.to(self.device)

        return x, t, lc, y