import numpy as np
import torch.nn as nn
import torch
from torch.autograd import grad
import torch.nn.functional as F
import sys

class skip_connection(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        width,
        depth,
        weight_norm=True,
        skip_layer=[],
        relu=False
    ):
        super().__init__()

        dims = [d_in] + [width] * depth + [d_out]
        self.num_layers = len(dims)

        self.skip_layer = skip_layer

        for l in range(0, self.num_layers - 1):

            if l in self.skip_layer:
                lin = torch.nn.Linear(dims[l] + dims[0], dims[l+1])
            else:
                lin = torch.nn.Linear(dims[l], dims[l+1])

            if weight_norm:
                lin = torch.nn.utils.weight_norm(lin)
            else:
                torch.nn.init.xavier_uniform_(lin.weight)
                torch.nn.init.zeros_(lin.bias)


            setattr(self, "lin" + str(l), lin)

        if relu:
            self.activation = torch.nn.ReLU()
        else:
            self.activation = torch.nn.LeakyReLU()

    def forward(self, input, softmax=False):
        """MPL query.

        Tensor shape abbreviation:
            B: batch size
            T: length
            D: input dimension
            
        Args:
            input (tensor): network input. shape: [B, T, D]

        Returns:
            output (tensor): network output. Might contains placehold if mask!=None shape: [B, T, ?]
        """

        batch_size, len_seq, n_dim = input.shape
        input = input.reshape(batch_size * len_seq, n_dim)

        x = input
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_layer:
                x_mid = x.clone()
                x = torch.cat([x, input], 1)
            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        x_full = x.reshape(batch_size, len_seq, -1)
        if softmax:
            x_full = x_full.softmax(dim=-1)
        return x_full


class lbs_pbs_module(torch.nn.Module):
    def __init__(self, d_in, d_out, hidden_size=256, matrix=False, skip=False, init=True):
        super().__init__()

        self.matrix = matrix
        self.lin1 = torch.nn.Linear(d_in, hidden_size)
        self.lin2 = torch.nn.Linear(hidden_size, hidden_size)
        if skip:
            self.lin3 = torch.nn.Linear(hidden_size+d_in, hidden_size) 
        else:
            self.lin3 = torch.nn.Linear(hidden_size, hidden_size)
        self.lin4 = torch.nn.Linear(hidden_size, hidden_size)
        self.lin5 = torch.nn.Linear(hidden_size, d_out)
        self.skip = skip
        
        self.relu = torch.nn.LeakyReLU()
        
        if init:
            self.__init__weight()

    def __init__weight(self,):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.zeros_(self.lin1.bias)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        torch.nn.init.zeros_(self.lin2.bias)
        torch.nn.init.xavier_uniform_(self.lin3.weight)
        torch.nn.init.zeros_(self.lin3.bias)
        torch.nn.init.xavier_uniform_(self.lin4.weight)
        torch.nn.init.zeros_(self.lin4.bias)
        torch.nn.init.xavier_uniform_(self.lin5.weight)
        torch.nn.init.zeros_(self.lin5.bias)


    def forward(self, input, return_feature=False):
        """
        Tensor shape abbreviation:
            B: batch size
            N: number of points
            D: input dimension
            
        Args:
            input (tensor): network input. shape: [B, N, D]

        Returns:
            output (tensor): network output.
        """


        n_batch, n_point, n_dim = input.shape

        if n_batch * n_point == 0:
            return input

        input = input.reshape(n_batch * n_point, n_dim)

        x = input
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        if self.skip:
            x = torch.cat((x, input), dim=-1)
        x = self.lin3(x)
        x = self.relu(x)
        x = self.lin4(x)
        feature = x.clone()
        x = self.relu(x)
        x = self.lin5(x)

        if self.matrix:
            x_full = x.reshape(n_batch, n_point, -1, 3)
        else:
            x_full = x.reshape(n_batch, n_point, -1)

        feature = feature.reshape(n_batch, n_point, -1)
        if return_feature:
            return x_full, feature
        else:
            return x_full


class lbs_pbs(torch.nn.Module):
    def __init__(self, d_in_theta, d_in_x, d_out_p, hidden_theta=256, hidden_matrix=256, skip=False, soft_max=False, init=True):
        super().__init__()

        self.soft_max = soft_max
        self.lbs_theta = lbs_pbs_module(d_in=d_in_theta, d_out=d_out_p, hidden_size=hidden_theta, skip=skip, init=init)
        self.lbs_matrix = lbs_pbs_module(d_in=d_in_x, d_out=d_out_p*3, hidden_size=hidden_matrix, matrix=True, skip=skip, init=init)

    def forward(self, theta, x, return_encode=False):
        """
        Tensor shape abbreviation:
            B: batch size
            N: number of points
            D: input dimension
            
        Args:
            theta (tensor): network input. shape: [B, N, D]
            x (tensor): network input. shape: [B, N, D]

        Returns:
            output (tensor): network output.
        """

        theta_encode = self.lbs_theta(theta)

        input = x
        deform_matrix = self.lbs_matrix(input)

        if self.soft_max:
            theta_encode = F.softmax(theta_encode, dim=-1)

        delta_x = torch.einsum("bpi,bpij->bpj", theta_encode, deform_matrix)

        if return_encode:
            return delta_x, theta_encode
        else:
            return delta_x


""" Positional encoding embedding. Code was taken from https://github.com/bmild/nerf. """
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.0 ** 0.0, 2.0 ** max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires):
    embed_kwargs = {
        "include_input": True,
        "input_dims": 3,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, eo=embedder_obj):
        return eo.embed(x)

    return embed, embedder_obj.out_dim