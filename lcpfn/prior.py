import os
import torch
import math
import numpy as np
import random
from scipy.stats import norm
from lcpfn.transformer import Normalize

def pow3(x, a, c, alpha, *args):
    return c - a * torch.pow(x+1, -alpha)

class DatasetPrior:

    output_sorted = None

    def init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight)

    def __init__(self, num_features, num_outputs):
        self.num_features = num_features
        self.num_outputs = num_outputs
        
        self.num_inputs = 2*(num_features+2)
        num_hidden = 100
        N = 1000
        
        self.model = torch.nn.Sequential(
            Normalize(torch.tensor(0.5), torch.tensor(math.sqrt(1 / 12))),
            torch.nn.Linear(self.num_inputs, num_hidden),
            torch.nn.ELU(),
            torch.nn.Linear(num_hidden, num_hidden),
            torch.nn.ELU(),
            torch.nn.Linear(num_hidden, self.num_outputs)
        )

        if DatasetPrior.output_sorted is None:
            # generate samples to approximate the CDF of the BNN output distribution
            output = torch.zeros((N, num_outputs))
            input = torch.from_numpy(np.random.uniform(size=(N,self.num_inputs))).to(torch.float32)
            with torch.no_grad():
                for i in range(N):
                    self.model.apply(DatasetPrior.init_weights)
                    output[i] = self.model(input[i])
            
            DatasetPrior.output_sorted = np.sort(torch.flatten(output).numpy())

        # fix the parameters of the BNN
        self.model.apply(DatasetPrior.init_weights)

        # fix other dataset specific
        self.input_features = np.random.uniform(size=(self.num_inputs,))
        p_alloc = np.random.dirichlet(tuple([1 for _ in range(self.num_features)] + [1, 1]))
        self.alloc = np.random.choice(self.num_features+2, size=(self.num_inputs,), p=p_alloc)
        #print(self.alloc)

    
    def input_for_config(self, config):
        input_noise = np.random.uniform(size=(self.num_inputs,))
        input = torch.zeros((self.num_inputs,))
        for j in range(self.num_inputs):
            if self.alloc[j] < self.num_features:
                input[j] = config[self.alloc[j]]
            elif self.alloc[j] == self.num_features:
                input[j] = self.input_features[j]
            else:
                input[j] = input_noise[j]
        return input
        
    def output_for_config(self, config):
        input = self.input_for_config(config)
        return self.model(input)

    def uniform(self, bnn_output, a=0.0, b=1.0):
        indices = np.searchsorted(DatasetPrior.output_sorted, bnn_output, side='left')
        return (b-a) * indices / len(DatasetPrior.output_sorted) + a
    
    def normal(self, bnn_output, loc=0, scale=1):
        eps = 0.5 / len(DatasetPrior.output_sorted) # to avoid infinite samples
        u = self.uniform(bnn_output, a=eps, b=1-eps)
        return norm.ppf(u, loc=loc, scale=scale)


# function producing batches for PFN training
@torch.no_grad()
def get_batch(
    batch_size,
    seq_len,
    num_features,
    device=torch.device("cpu"),
    hyperparameters=None,
    **kwargs,
):
    # assert num_features == 2
    ncurves = hyperparameters.get("ncurves", 50)
    nepochs = hyperparameters.get("nepochs", 50)

    assert seq_len == ncurves * nepochs
    assert num_features >= 2

    if hyperparameters.get("fix_nparams", False):
        num_params = num_features - 2
    else:
        num_params = np.random.randint(1, num_features - 2)

    x_ = torch.arange(1, nepochs + 1)

    x = []
    y = []

    for i in range(batch_size):
        epoch = torch.zeros(nepochs * ncurves)
        id_curve = torch.zeros(nepochs * ncurves)
        curve_val = torch.zeros(nepochs * ncurves)
        config = torch.zeros(nepochs * ncurves, num_params)

        # sample a collection of curves
        dataset = DatasetPrior(num_params, 3)
        curves = []
        curve_configs = []
        c_minus_a = np.random.uniform()
        for i in range(ncurves):
            c = np.random.uniform(size=(num_params,))  # random configurations
            curve_configs.append(c)
            output = dataset.output_for_config(c).numpy()
            c = dataset.uniform(output[0], a=c_minus_a, b=1.0)
            a = c - c_minus_a
            alpha = np.exp(dataset.normal(output[1], scale=2))
            sigma = np.exp(dataset.normal(output[2], loc=-5, scale=1))
            y_ = pow3(x_, a, c, alpha)
            noise = np.random.normal(size=x_.shape, scale=sigma)
            y_noisy = y_+noise
            curves.append(y_noisy)
            if torch.isnan(torch.sum(y_noisy)) or not torch.isfinite(torch.sum(y_noisy)):
                print(f"{a}, {c}, {alpha}, {sigma}")
                print(y_+noise)

        # determine an ordering
        p_new = 10**np.random.uniform(-3, 0)
        greediness = 10**np.random.uniform(0, 3)  # Select prob 'best' / 'worst' possible (1 = URS)
        ordering = hyperparameters.get("ordering", "URS")
        ids = np.arange(ncurves)
        np.random.shuffle(ids)  # randomize the order of IDs to avoid learning weird biases
        cutoff = torch.zeros(ncurves).type(torch.int64)
        for i in range(ncurves * nepochs):
            if ordering == "URS":
                candidates = [j for j, c in enumerate(cutoff) if c < nepochs]
                selected = np.random.choice(candidates)
            elif ordering == "BFS":
                selected = i % ncurves
            elif ordering == "DFS":
                selected = i // nepochs
            elif ordering == "SoftGreedy":
                u = np.random.uniform()
                if u < p_new:
                    new_candidates = [j for j, c in enumerate(cutoff) if c == 0]
                if u < p_new and len(new_candidates) > 0:
                    selected = np.random.choice(new_candidates)
                else:
                    candidates = [j for j, c in enumerate(cutoff) if c < nepochs and c > 0]
                    if len(candidates) == 0:
                        if u >= p_new:
                            new_candidates = [j for j, c in enumerate(cutoff) if c == 0]
                        selected = np.random.choice(new_candidates)
                    else:
                        # use softmax selection based on performance and selected cutoff
                        # select a cutoff to compare on (based on current / last)
                        selected_cutoff = np.random.randint(nepochs)
                        values = []
                        for j in candidates:
                            values.append(curves[j][min(selected_cutoff,cutoff[j]-1)])
                        sm_values = np.power(greediness, np.asarray(values))
                        sm_values = sm_values / np.sum(sm_values)
                        if np.isnan(np.sum(sm_values)):
                            print(values)
                            print(greediness)
                            print(selected_cutoff)
                            print(cutoff[j])
                            for j in candidates:
                                print(curves[:cutoff[j]])
                        selected = np.random.choice(candidates, p=sm_values)
            else:
                raise NotImplementedError
            id_curve[i] = ids[selected]
            curve_val[i] = curves[selected][cutoff[selected]]
            config[i] = torch.from_numpy(curve_configs[selected])
            cutoff[selected] += 1
            epoch[i] = cutoff[selected]
        x.append(torch.cat([torch.stack([id_curve, epoch], dim=1), config], dim=1))
        y.append(curve_val)
    
        x = torch.stack(x, dim=1).to(device).float()
        y = torch.stack(y, dim=1).to(device).float()
        return x, y