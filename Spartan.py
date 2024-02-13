from typing import Optional, Tuple
import gpytorch
from gpytorch.constraints import Interval
from gpytorch.priors import Prior
import torch
from torch._C import Size
from torch.nn import ModuleList
from ast import match_case
from math import sqrt
from statistics import linear_regression
from typing import Iterable
from gpytorch.kernels import Kernel
from linear_operator.operators import ZeroLinearOperator
from gpytorch.distributions import MultivariateNormal


class SpartanKernel(Kernel):

        has_lengthscale = False

        def __init__(self, global_kernel: Kernel, local_kernels: Iterable[Kernel], 
                     ard_num_dims: int = 1,
                     local_position_prior: Optional[Prior] = None,
                     local_position_constraint: Optional[Interval] = None,
                     eps: float = 0.000001, **kwargs):
                
                
                super(SpartanKernel, self).__init__(ard_num_dims=ard_num_dims)

                self.global_kernel = global_kernel
                self.local_kernels = ModuleList(local_kernels)
                self.register_parameter(
                        name="raw_local_position", 
                        parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape,1,ard_num_dims))
                )
                if local_position_prior is not None:
                        if not isinstance(local_position_prior, Prior):
                                raise TypeError("Expected gpytorch.priors.Prior but got " + type(local_position_prior).__name__)
                        self.register_prior(
                                "local_position_prior",
                                local_position_prior,
                                lambda m: m.local_position,
                                lambda m, v: m._set_local_position(v),
                        )
                if local_position_constraint is None:
                        local_position_constraint = Interval(torch.zeros(1, ard_num_dims).squeeze(), 
                                                             torch.ones(1, ard_num_dims).squeeze())
                        # Constrained between 0 to 1, can be modified for the inputs
                self.register_constraint("raw_local_position", local_position_constraint)
                # Weight parameters other than center:
                weight_params = {'psi' : torch.ones([1,ard_num_dims])*0.5,
                                 'sigma_g' : sqrt(10.),
                                 'Normalized' : True,
                                 'sigma_l' : torch.tensor([sqrt(0.01)])}
                #                 'local_num_samples' : None}
                weight_params.update(kwargs)
                self.eps = eps
                self.register_buffer('psi', weight_params['psi'])
                self.register_buffer('sigma_g', torch.as_tensor(weight_params['sigma_g']))
                self.Normalized = weight_params['Normalized']
                #if weight_params['local_num_samples'] is not None:
                #        self.sigma_l = sqrt(weight_params['local_num_samples']/2)
                #else:
                #self.register_buffer('sigma_l', weight_params['sigma_l'])
                # TO DO: What if we want seperate sigma_l for different kernels?

                


        @property
        def local_position(self):
                return self.raw_local_position_constraint.transform(self.raw_local_position)

        @local_position.setter
        def local_position(self, value):
                return self._set_local_position(value)
        
        def _set_local_position(self, value):
                if not torch.is_tensor(value):
                        value = torch.as_tensor(value).to(self.raw_local_position)
                
                self.initialize(raw_local_position = self.raw_local_position_constraint.inverse_transform(value))

        def omega_g(self, x):
                """Helper function for unnormalized weights"""

        def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool=False, **params):
                #print(x1.shape,x2.shape)
                # This is just to make mll work:
                res = ZeroLinearOperator() if not diag else 0
                #if diag and torch.equal(x1, x2):
                #       return torch.ones(*x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device)
                _k_g = self.global_kernel(x1, x2, diag=diag).to_dense()
                _w_g_1 = MultivariateNormal(self.psi, torch.eye(self.ard_num_dims, device=self.device)*self.sigma_g**2).log_prob(x1)
                _w_g_2 = MultivariateNormal(self.psi, torch.eye(self.ard_num_dims, device=self.device)*self.sigma_g**2).log_prob(x2)
                #print("w",_w_g_1.shape,_w_g_2.shape)
                _w_sum_1 = torch.exp(_w_g_1) # keep track of total weights
                _w_sum_2 = torch.exp(_w_g_2)
                if not diag:
                        res = res + torch.mul(_k_g, torch.sqrt(torch.matmul(_w_sum_1.unsqueeze(-1),
                                                                     _w_sum_2.unsqueeze(-2)))) # TO DO: decide on whether logsum is needed
                else:
                        res = res + torch.mul(_k_g, torch.exp(_w_g_1 + _w_g_2))
                for _k in self.local_kernels:
                        # Only same sigma_l for different local kernels considered
                        _w_l_dist = MultivariateNormal(self.local_position, torch.eye(self.ard_num_dims, device=self.device)*self.sigma_l**2)
                        _w_l_1 = _w_l_dist.log_prob(x1)
                        _w_l_2 = _w_l_dist.log_prob(x2)
                        _w_sum_1 = _w_sum_1 + torch.exp(_w_l_1)
                        _w_sum_2 = _w_sum_2 + torch.exp(_w_l_2)
                        if not diag:
                                res = res + torch.mul(_k(x1, x2, diag=diag).to_dense(), 
                                                torch.matmul(torch.exp(_w_l_1/2).unsqueeze(-1), 
                                                                torch.exp(_w_l_2/2).unsqueeze(-2)))
                        else:
                                res = res + torch.mul(_k(x1, x2, diag=diag).to_dense(),
                                                                torch.exp(_w_l_1/2 + _w_l_2/2))
                        
                # Now apply normalization
                if not diag:
                        res = torch.div(res, torch.unsqueeze(torch.sqrt(_w_sum_1), -1))
                        if res.dim() > 2:
                                res = torch.div(res, torch.unsqueeze(torch.sqrt(_w_sum_2), -2))
                        else:
                                res = torch.div(res, torch.sqrt(_w_sum_2))
                else:
                        res = torch.div(res, torch.sqrt(torch.mul(_w_sum_1, _w_sum_2)))        
                return res