import numpy as np
import torch

class ExpBasis(torch.nn.Module):
    def __init__(self, n_rbf, cutoff, trainable=True):
        super().__init__()

        if n_rbf == 1:
            r0 = torch.tensor(1.0, dtype=torch.get_default_dtype())
        else:
            r0 = torch.linspace(0.0, 10/cutoff, n_rbf + 2, dtype=torch.get_default_dtype())[1:-1]
        
        if trainable:
            self.r0 = torch.nn.Parameter(r0.view(-1,1))
        else:
            self.register_buffer('r0',r0.view(-1,1))

        self.n_rbf = n_rbf

    def forward(self, dist):

        basis =  torch.nn.functional.linear(dist.view(-1,1),self.r0,None)
        basis_exp = torch.exp(torch.clamp(basis.view(dist.shape[0],self.n_rbf), max=50.0))
        return basis_exp

class BesselBasis(torch.nn.Module):
    def __init__(self, n_rbf, cutoff):
        super().__init__()

        self.n_rbf = n_rbf
        self.cutoff = cutoff

        d_k_list = [1]
        e_k_list = [self.e_k(0)]
        f_k_coeff_list = [self.f_k_coeff(0)]

        for k in range(1,self.n_rbf):
            e_k_list.append(self.e_k(k))
            f_k_coeff_list.append(self.f_k_coeff(k))
            d_k_list.append(1-e_k_list[k]/d_k_list[k-1])

        sinc_coeff_list = []
        g_coeff_list = []

        for k in range(1,self.n_rbf):
            sinc_coeff_list.append(f_k_coeff_list[k]/d_k_list[k]**0.5)
            g_coeff_list.append((e_k_list[k]/d_k_list[k-1]/d_k_list[k])**0.5)

        self.sinc_coeff_list = sinc_coeff_list
        self.g_coeff_list = g_coeff_list
            
    def f_k_coeff(self, k):
        
        coeff = (-1)**k*(2**0.5*np.pi)*(self.cutoff)**(-1.5)*((k+1)*(k+2)/((k+1)**2+(k+2)**2)**0.5)

        return coeff
    
    def e_k(self, k):

        e_k = (k*(k+2))**2/(4*(k+1)**4+1)

        return e_k

    def forward(self, dist):

        dist_scaled = dist/self.cutoff

        g = self.f_k_coeff(0)*(torch.special.sinc(dist_scaled)+torch.special.sinc(2*dist_scaled))

        g_array = [g]

        for k in range(1,self.n_rbf):
            g = self.sinc_coeff_list[k-1]*(torch.special.sinc((k+1)*dist_scaled)+torch.special.sinc((k+2)*dist_scaled)) + self.g_coeff_list[k-1]*g
            g_array.append(g)


        basis = torch.stack(g_array,dim=1)

        return basis

class PowerBasis(torch.nn.Module):
    def __init__(self, n_rbf, trainable = False):
        super().__init__()

        if trainable:
            self.exponent = torch.nn.Parameter(torch.arange(start=1, end=n_rbf+1, dtype=torch.get_default_dtype()))
        else:
            self.register_buffer("exponent",torch.arange(start=1, end=n_rbf+1))
    
    def forward(self, dist):

        return torch.pow(dist.view(-1,1),self.exponent)

class SlaterBasis(torch.nn.Module):
    def __init__(self, n_rbf, cutoff):
        super().__init__()

        if n_rbf == 1:
            r0 = torch.tensor(1/cutoff, dtype=torch.get_default_dtype())
        else:
            r0 = torch.linspace(0.0, 10/cutoff, n_rbf + 2, dtype=torch.get_default_dtype())[1:-1]
        

        self.r0 = torch.nn.Parameter(r0.view(-1,1))
        self.invexponent = torch.nn.Parameter((torch.rand(n_rbf)-0.5)*0.01)
        self.n_rbf = n_rbf

        self.positive_constraint = torch.nn.Softplus()

    def forward(self, dist):

        exponent = self.positive_constraint(self.invexponent)
        basis =  torch.nn.functional.linear(dist.view(-1,1),self.r0,None)
        basis_exp = torch.exp(basis.view(dist.shape[0],self.n_rbf))

        return torch.pow((1/dist).view(-1,1),exponent)*basis_exp


