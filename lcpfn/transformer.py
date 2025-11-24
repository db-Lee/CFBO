import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from lcpfn.utils import bool_mask_to_att_mask

class Normalize(nn.Module):
    def __init__(self, mean: torch.FloatTensor, std: torch.FloatTensor):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x):
        return (x-self.mean)/(self.std+1e-8)

class TransformerModel(nn.Module):
    def __init__(
        self,
        dim_x,
        d_output,
        d_model,
        dim_feedforward, 
        nlayers, 
        dropout=0.0, 
        data_stats=None,
        activation='gelu',
        criterion=None
    ):
        super().__init__()
        self.model_type = 'Transformer'
        transformer_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=dim_feedforward, dropout=dropout,
            activation=activation, batch_first=True)
        self.transformer = TransformerEncoder(transformer_layer, nlayers)
        
        if data_stats is None:
            x_mean, x_std = torch.zeros(dim_x)+0.5, torch.zeros(dim_x)+math.sqrt(1/12)
            t_mean, t_std = torch.zeros(1)+0.5, torch.zeros(1)+math.sqrt(1/12)
            y_mean, y_std = torch.zeros(1)+0.5, torch.zeros(1)+math.sqrt(1/12)
        else:
            x_mean, x_std, t_mean, t_std, y_mean, y_std = data_stats
        
        self.x_encoder = nn.Sequential(
            Normalize(x_mean, x_std),
            nn.Linear(dim_x, d_model),
        )
        self.t_encoder = nn.Sequential(
            Normalize(t_mean, t_std),
            nn.Linear(1, d_model),
        )
        self.y_encoder = nn.Sequential(
            Normalize(y_mean, y_std),
            nn.Linear(1, d_model),
        )

        self.decoder = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Linear(dim_feedforward, d_output))
        self.criterion = criterion

        self.init_weights()

    @staticmethod
    def generate_D_q_matrix(sz, query_size):
        train_size = sz-query_size
        mask = torch.zeros(sz,sz) == 0
        mask[:,train_size:].zero_()
        mask |= torch.eye(sz) == 1
        return bool_mask_to_att_mask(mask)
    
    def init_weights(self):
        for layer in self.transformer.layers:
            nn.init.zeros_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)
            attns = layer.self_attn if isinstance(layer.self_attn, nn.ModuleList) else [layer.self_attn]
            for attn in attns:
                nn.init.zeros_(attn.out_proj.weight)
                nn.init.zeros_(attn.out_proj.bias)

    def forward(self, tc_0, yc_0, xc, tc, yc, xt, tt):
        device = tt.device        
        B, M, N = tt.shape[0], 1, tt.shape[1]
        
        # hc_0
        hc_0 = self.t_encoder(tc_0) + self.y_encoder(yc_0)

        # hc
        if xc is None or xc.shape[1] == 0:
            hc = hc_0            
        else:
            M += tc.shape[1]            
            hc = self.x_encoder(xc) + self.t_encoder(tc) + self.y_encoder(yc)            
            hc = torch.cat([hc_0, hc], dim=1)                        
        
        # ht
        ht = self.x_encoder(xt) + self.t_encoder(tt)

        # h
        h = torch.cat([hc, ht], 1)

        # transformer
        mask = self.generate_D_q_matrix(M+N, N).to(device)
        output = self.transformer(h, mask)
        output = self.decoder(output)

        return output[:, M:, :].contiguous()
