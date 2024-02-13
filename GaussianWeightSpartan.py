from typing import Optional, Tuple
import gpytorch
from gpytorch.constraints import Interval
from gpytorch.priors import Prior
import torch
from torch._C import Size
from torch.nn import ModuleList
from ast import match_case
from statistics import linear_regression
from typing import Iterable
from gpytorch.kernels import Kernel
from linear_operator.operators import ZeroLinearOperator
from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy import ZeroLazyTensor


class Gaussian_Weight_Spartan_Kernel(Kernel):
        has_lengthscale = False
        def __init__(self, global_kernel: Kernel, local_kernels: Iterable[Kernel], 
                     ard_num_dims: int = 1,
                     local_position_prior: Optional[Prior] = None,
                     #local_position_constraint: Optional[Interval] = None,
                     local_weight_var_prior: Optional[Prior] = None,
                     local_weight_var_constraint: Optional[Interval] = None,
                     eps: float = 0.000001, **kwargs):
                
                
                super().__init__(ard_num_dims=ard_num_dims)

                self.global_kernel = global_kernel
                self.local_kernels = ModuleList(local_kernels)
                # numbers of local kernels for calculation
                self.local_kernels_num = len(self.local_kernels)
                # hyperparameters for weights
                # Note: The logistic functions used to set interval constraints might cause a problem. The optimizer might favour either ends of the interval where the convergence seems to be reached because of almost 0 differentials.
                self.register_parameter(
                        name="local_position", 
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

                self.register_parameter(
                        name="raw_local_weight_var",
                        parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, self.local_kernels_num))
                )
                if local_weight_var_prior is not None:
                        if not isinstance(local_weight_var_prior, Prior):
                                raise TypeError("Expected gpytorch.priors.Prior but got " + type(local_weight_var_prior).__name__)
                        self.register_prior(
                                "local_weight_var_prior",
                                local_weight_var_prior,
                                lambda m: m.local_weight_var,
                                lambda m, v: m._set_local_weight_var(v),
                        )
                
                if local_weight_var_constraint is None:
                        local_weight_var_constraint = Interval(torch.zeros(self.local_kernels_num).squeeze(),
                                                                torch.ones(self.local_kernels_num).squeeze())       
                self.register_constraint("raw_local_weight_var", local_weight_var_constraint)

                # Weight parameters other than center:
                weight_params = {'psi' : torch.ones([1,ard_num_dims])*0.5,
                                 'sigma_g' : sqrt(10.),
                                 'Normalized' : True,
                                 #'sigma_l' : torch.tensor([sqrt(0.01)])
                                 }
                #                 'local_num_samples' : None}
                weight_params.update(kwargs)
                self.eps = eps
                self.register_buffer('psi', weight_params['psi'])
                self.register_buffer('sigma_g', torch.as_tensor(weight_params['sigma_g']))
                self.Normalized = weight_params['Normalized']
                #
                #if weight_params['local_num_samples'] is not None:
                #        self.sigma_l = sqrt(weight_params['local_num_samples']/2)
                #else:
                #self.register_buffer('sigma_l', weight_params['sigma_l'])
                # TO DO: Incorporate information on samples for local weight priors?

                


        @property
        def local_position(self):
                return self._local_position

        @local_position.setter
        def local_position(self, value):
                return self._set_local_position(value)
        
        def _set_local_position(self, value):
                if not torch.is_tensor(value):
                        value = torch.as_tensor(value).to(self.local_position)
                
                self._local_position = torch.nn.Parameter(value)

        @property
        def local_weight_var(self):
                return self.raw_local_weight_var_constraint.transform(self.raw_local_weight_var)
        
        @local_weight_var.setter
        def local_weight_var(self, value):
                return self._set_local_weight_var(value)
        
        def _set_local_weight_var(self, value):
                if not torch.is_tensor(value):
                        value = torch.as_tensor(value).to(self.raw_local_weight_var)

                self.initialize(raw_local_weight_var = self.raw_local_weight_var_constraint.inverse_transform(value))

        def omega_g(self, x):
                """Helper function for calculating global weights
                   Input x: tensor for the location where weight is calculated. 
                        dimension should be (batches) * length * ard_num_dims
                """

                return torch.exp(-(x-self.psi).norm(dim=-1).pow(2)/(2*self.sigma_g**2))
        
        def omega_l(self, x, firstkernel = False):

                """Helper function for calculating local weights
                   Input x: tensor for the location where weight is calculated. 
                        Dimension should be (batches) * length * ard_num_dims. 
                        Returns a list that matches the length of the kernel's list for the
                        individual weights of the kernels.
                        If firstkernel = True, then only return the first local kernel's weight,
                        and not in a list.
                """
                if firstkernel:
                        return torch.exp(-(x - self.local_position).norm(dim=-1).pow(2)/(2*self.local_weight_var[0]**2))
                else:
                        return [torch.exp(-(x - self.local_position).norm(dim=-1).pow(2)/(2*local_var**2)) for local_var in self.local_weight_var]


        def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool=False, **params):
                res = ZeroLazyTensor() if not diag else 0
                _k_g = self.global_kernel(x1, x2, diag=diag)
                if not diag:
                        _w_g = (torch.unsqueeze((x1 - self.psi).norm(dim=-1).pow(2), -1) + torch.unsqueeze((x2 - self.psi).norm(dim=-1).pow(2), -2))/(2*self.sigma_g**2)
                else:
                        _w_g = ((x1 - self.psi).norm(dim=-1).pow(2) + (x2 - self.psi).norm(dim=-1).pow(2))/(2*self.sigma_g**2)

                res = res + _k_g.mul(torch.exp(- _w_g)/3)
                for _kernel, local_var in zip(self.local_kernels, self.local_weight_var):
                        _k_l = _kernel(x1, x2, diag = diag)
                        if not diag:
                                _w_l = (torch.unsqueeze((x1 - self.local_position).norm(dim=-1).pow(2), -1) + torch.unsqueeze((x2 - self.local_position).norm(dim=-1).pow(2), -2))/(2*local_var**2)
                        else:
                                _w_l = (x1 - self.local_position).norm(dim=-1).pow(2) + (x2 - self.local_position).norm(dim=-1).pow(2)/(2*local_var**2)
                        res = res + _k_l.mul(torch.exp(- _w_l))
                
                return res