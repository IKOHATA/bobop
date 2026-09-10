import torch
from typing import List

class InversePowOutput(torch.nn.Module):
    def __init__(
        self,
        input_dicts: List[dict],
        zeta_size: int,
        nelem: int,
        embedding_dim: int,
        hidden_dimensions: List[int],
        max_eta: int = 5, 
    ) -> None:
        super().__init__()

        self.nelem = nelem
        self.embedding_dim = embedding_dim
        self.body_order = len(input_dicts) + 2
        self.hidden_dimensions = hidden_dimensions

        assert len(hidden_dimensions)+2 == self.body_order

        self.zeta_size = zeta_size

        self.element_embedding_T = torch.nn.Embedding(self.nelem*self.nelem,self.embedding_dim,max_norm=1.0)

        self.linear = torch.nn.Linear(self.zeta_size, 1, bias=False)

        self.invn = torch.nn.Parameter(torch.zeros(1))

        self.register_buffer('dummy_edge_type',torch.arange(self.nelem*self.nelem))

        self.cached_weights = None

        self.extpow = ExtendedPow(ninput=self.zeta_size, max_eta = max_eta)

        if self.body_order == 3:
            self.compute_weights = self.compute_weights_three
        elif self.body_order == 4:
            self.compute_weights = self.compute_weights_four
        elif self.body_order == 5:
            self.compute_weights = self.compute_weights_five

        if self.body_order >= 3:
            indices_threebody = torch.tensor(list(input_dicts[0].keys()))
            n_size, l_size = torch.max(indices_threebody,dim=0).values + 1

            self.register_buffer("indices_threebody",indices_threebody)

            self.params1_threebody = torch.nn.Parameter(torch.rand((embedding_dim,n_size,hidden_dimensions[0]))-0.5)
            self.params2_threebody = torch.nn.Parameter(torch.rand((hidden_dimensions[0],l_size,zeta_size))-0.5)

        if self.body_order >= 4:
            indices_fourbody = torch.tensor(list(input_dicts[1].keys()))
            n1_size, n2_size, xi_size = torch.max(indices_fourbody,dim=0).values + 1

            values_fourbody = torch.tensor(list(input_dicts[1].values()))

            self.register_buffer("indices_fourbody",indices_fourbody)
            self.register_buffer("multiplicity_fourbody",values_fourbody[:,1])

            self.params1_fourbody = torch.nn.Parameter(torch.rand((embedding_dim,n1_size,n2_size,hidden_dimensions[1]))-0.5)
            self.params2_fourbody = torch.nn.Parameter(torch.rand((hidden_dimensions[1],xi_size,zeta_size))-0.5)

        if self.body_order >= 5:
            indices_fivebody = torch.tensor(list(input_dicts[2].keys()))
            n1_size, n2_size, n3_size, xi_size = torch.max(indices_fivebody,dim=0).values + 1

            values_fivebody = torch.tensor(list(input_dicts[2].values()))

            self.register_buffer("indices_fivebody",indices_fivebody)
            self.register_buffer("multiplicity_fivebody",values_fivebody[:,1])

            self.params1_fivebody = torch.nn.Parameter(torch.rand((embedding_dim,n1_size,n2_size,n3_size,hidden_dimensions[2]))-0.5)
            self.params2_fivebody = torch.nn.Parameter(torch.rand((hidden_dimensions[2],xi_size,zeta_size))-0.5)

        if self.embedding_dim == 1:
            self.element_embedding_T.weight.data.fill_(1)
            self.element_embedding_T.weight.requires_grad = False

        self.reset_parameters()

    def reset_parameters(self):

        torch.nn.init.xavier_uniform_(self.linear.weight, gain=0.01)

    def compute_l2loss(self):

        G3 = torch.einsum("crm,crn->mn",self.params1_threebody,self.params1_threebody)
        X3 = torch.einsum("mxp,nxp->mn",self.params2_threebody,self.params2_threebody)

        threebody_l2loss = (G3*X3).sum()
        
        l2loss = threebody_l2loss

        if self.body_order >= 4:
            param1_fourbody_flatten = self.params1_fourbody.flatten(start_dim=1,end_dim=2)
            G4 = torch.einsum("crm,crn->mn",param1_fourbody_flatten,param1_fourbody_flatten)
            X4 = torch.einsum("mxp,nxp->mn",self.params2_fourbody,self.params2_fourbody)

            fourbody_l2loss = (G4*X4).sum()

            l2loss = l2loss + fourbody_l2loss
        
        if self.body_order >= 5:
            param1_fivebody_flatten = self.params1_fivebody.flatten(start_dim=1,end_dim=3)
            G5 = torch.einsum("crm,crn->mn",param1_fivebody_flatten,param1_fivebody_flatten)
            X5 = torch.einsum("mxp,nxp->mn",self.params2_fivebody,self.params2_fivebody)

            fivebody_l2loss = (G5*X5).sum()

            l2loss = l2loss + fivebody_l2loss
     
        return l2loss
    
    def compute_weights_three(self):

        weights_three = torch.einsum("cbm,mbp->pcb",self.params1_threebody[:,self.indices_threebody[:,0],:],self.params2_threebody[:,self.indices_threebody[:,1],:])

        self.cached_weights = weights_three.flatten(start_dim=1)
    
    def compute_weights_four(self):

        weights_three = torch.einsum("cbm,mbp->pcb",self.params1_threebody[:,self.indices_threebody[:,0],:],self.params2_threebody[:,self.indices_threebody[:,1],:])

        weights_four = torch.einsum("cbm,mbp->pcb",
        self.params1_fourbody[:,self.indices_fourbody[:,0],self.indices_fourbody[:,1],:],
        self.params2_fourbody[:,self.indices_fourbody[:,2],:]
        )*self.multiplicity_fourbody

        weights = torch.concat([weights_three,weights_four],dim=-1)

        self.cached_weights = weights.flatten(start_dim=1)

    def compute_weights_five(self):

        weights_three = torch.einsum("cbm,mbp->pcb",self.params1_threebody[:,self.indices_threebody[:,0],:],self.params2_threebody[:,self.indices_threebody[:,1],:])

        weights_four = torch.einsum("cbm,mbp->pcb",
        self.params1_fourbody[:,self.indices_fourbody[:,0],self.indices_fourbody[:,1],:],
        self.params2_fourbody[:,self.indices_fourbody[:,2],:]
        )*self.multiplicity_fourbody

        weights_five = torch.einsum("cbm,mbp->pcb",
        self.params1_fivebody[:,self.indices_fivebody[:,0],self.indices_fivebody[:,1],self.indices_fivebody[:,2],:],
        self.params2_fivebody[:,self.indices_fivebody[:,3],:]
        )*self.multiplicity_fivebody

        weights = torch.concat([weights_three,weights_four,weights_five],dim=-1)

        self.cached_weights = weights.flatten(start_dim=1)

    def forward(self, B_basis, edge_type):

        if self.training or self.cached_weights is None:
            self.compute_weights()

        element_embedding = self.element_embedding_T(edge_type)

        if self.embedding_dim == 1:
            xi = torch.concat(B_basis,dim=-1).flatten(start_dim=1)
        else:
            xi = torch.einsum("ec,eb->ecb",element_embedding,torch.concat(B_basis,dim=-1)).flatten(start_dim=1)

        phi = torch.nn.functional.linear(xi,self.cached_weights,None)

        zeta = self.extpow(phi)

        zeta_norm = torch.clamp(torch.sqrt(zeta.pow(2).sum(dim=1)+1e-4) - 1e-2, min=0.0)

        n0 = torch.nn.functional.softplus(self.invn)
        n = torch.nn.functional.softplus(self.invn) + 1.0

        bij0 = (1+zeta_norm).pow(-n0)

        bij1 = self.linear(zeta)*(1+zeta_norm).unsqueeze(dim=-1).pow(-n)

        bij = bij0 + bij1.view(-1)

        return bij

class InversePowMultiOutput(torch.nn.Module):
    def __init__(
        self,
        input_dicts: List[dict],
        zeta_size: int,
        out_size: int, 
        nelem: int,
        embedding_dim: int,
        hidden_dimensions: List[int],
        max_eta: int = 5, 
    ) -> None:
        super().__init__()

        self.nelem = nelem
        self.embedding_dim = embedding_dim
        self.body_order = len(input_dicts) + 2
        self.hidden_dimensions = hidden_dimensions
        self.out_size = out_size

        assert len(hidden_dimensions)+2 == self.body_order

        self.zeta_size = zeta_size

        self.element_embedding_T = torch.nn.Embedding(self.nelem*self.nelem,self.embedding_dim,max_norm=1.0)

        self.linear = torch.nn.Linear(self.zeta_size, self.out_size, bias=False)

        self.invn0 = torch.nn.Parameter(torch.zeros(1))
        self.invn = torch.nn.Parameter(torch.zeros(self.out_size))

        self.register_buffer('dummy_edge_type',torch.arange(self.nelem*self.nelem))

        self.cached_weights = None

        self.extpow = ExtendedPow(ninput=self.zeta_size, max_eta = max_eta)

        if self.body_order == 3:
            self.compute_weights = self.compute_weights_three
        elif self.body_order == 4:
            self.compute_weights = self.compute_weights_four

        if self.body_order >= 3:
            indices_threebody = torch.tensor(list(input_dicts[0].keys()))
            n_size, l_size = torch.max(indices_threebody,dim=0).values + 1

            self.register_buffer("indices_threebody",indices_threebody)

            self.params1_threebody = torch.nn.Parameter(torch.rand((embedding_dim,n_size,hidden_dimensions[0]))-0.5)
            self.params2_threebody = torch.nn.Parameter(torch.rand((hidden_dimensions[0],l_size,zeta_size))-0.5)

        if self.body_order >= 4:
            indices_fourbody = torch.tensor(list(input_dicts[1].keys()))
            n1_size, n2_size, xi_size = torch.max(indices_fourbody,dim=0).values + 1

            values_fourbody = torch.tensor(list(input_dicts[1].values()))

            self.register_buffer("indices_fourbody",indices_fourbody)
            self.register_buffer("multiplicity_fourbody",values_fourbody[:,1])

            self.params1_fourbody = torch.nn.Parameter(torch.rand((embedding_dim,n1_size,n2_size,hidden_dimensions[1]))-0.5)
            self.params2_fourbody = torch.nn.Parameter(torch.rand((hidden_dimensions[1],xi_size,zeta_size))-0.5)

        if self.embedding_dim == 1:
            self.element_embedding_T.weight.data.fill_(1)
            self.element_embedding_T.weight.requires_grad = False

        self.reset_parameters()

    def reset_parameters(self):

        torch.nn.init.xavier_uniform_(self.linear.weight, gain=0.01)

    def compute_l2loss(self):

        G3 = torch.einsum("crm,crn->mn",self.params1_threebody,self.params1_threebody)
        X3 = torch.einsum("mxp,nxp->mn",self.params2_threebody,self.params2_threebody)

        threebody_l2loss = (G3*X3).sum()
        
        l2loss = threebody_l2loss

        if self.body_order >= 4:
            param1_fourbody_flatten = self.params1_fourbody.flatten(start_dim=1,end_dim=2)
            G4 = torch.einsum("crm,crn->mn",param1_fourbody_flatten,param1_fourbody_flatten)
            X4 = torch.einsum("mxp,nxp->mn",self.params2_fourbody,self.params2_fourbody)

            fourbody_l2loss = (G4*X4).sum()

            l2loss = l2loss + fourbody_l2loss
        
        if self.body_order >= 5:
            param1_fivebody_flatten = self.params1_fivebody.flatten(start_dim=1,end_dim=3)
            G5 = torch.einsum("crm,crn->mn",param1_fivebody_flatten,param1_fivebody_flatten)
            X5 = torch.einsum("mxp,nxp->mn",self.params2_fivebody,self.params2_fivebody)

            fivebody_l2loss = (G5*X5).sum()

            l2loss = l2loss + fivebody_l2loss
     
        return l2loss
    
    def compute_weights_three(self):

        weights_three = torch.einsum("cbm,mbp->pcb",self.params1_threebody[:,self.indices_threebody[:,0],:],self.params2_threebody[:,self.indices_threebody[:,1],:])

        self.cached_weights = weights_three.flatten(start_dim=1)
    
    def compute_weights_four(self):

        weights_three = torch.einsum("cbm,mbp->pcb",self.params1_threebody[:,self.indices_threebody[:,0],:],self.params2_threebody[:,self.indices_threebody[:,1],:])

        weights_four = torch.einsum("cbm,mbp->pcb",
        self.params1_fourbody[:,self.indices_fourbody[:,0],self.indices_fourbody[:,1],:],
        self.params2_fourbody[:,self.indices_fourbody[:,2],:]
        )*self.multiplicity_fourbody

        weights = torch.concat([weights_three,weights_four],dim=-1)

        self.cached_weights = weights.flatten(start_dim=1)

    def compute_weights_five(self):

        weights_three = torch.einsum("cbm,mbp->pcb",self.params1_threebody[:,self.indices_threebody[:,0],:],self.params2_threebody[:,self.indices_threebody[:,1],:])

        weights_four = torch.einsum("cbm,mbp->pcb",
        self.params1_fourbody[:,self.indices_fourbody[:,0],self.indices_fourbody[:,1],:],
        self.params2_fourbody[:,self.indices_fourbody[:,2],:]
        )*self.multiplicity_fourbody

        weights_five = torch.einsum("cbm,mbp->pcb",
        self.params1_fivebody[:,self.indices_fivebody[:,0],self.indices_fivebody[:,1],self.indices_fivebody[:,2],:],
        self.params2_fivebody[:,self.indices_fivebody[:,3],:]
        )*self.multiplicity_fivebody

        weights = torch.concat([weights_three,weights_four,weights_five],dim=-1)

        self.cached_weights = weights.flatten(start_dim=1)

    def forward(self, B_basis, edge_type):

        if self.training or self.cached_weights is None:
            self.compute_weights()

        element_embedding = self.element_embedding_T(edge_type)

        if self.embedding_dim == 1:
            xi = torch.concat(B_basis,dim=-1).flatten(start_dim=1)
        else:
            xi = torch.einsum("ec,eb->ecb",element_embedding,torch.concat(B_basis,dim=-1)).flatten(start_dim=1)

        phi = torch.nn.functional.linear(xi,self.cached_weights,None)

        zeta = self.extpow(phi)

        zeta_norm = torch.clamp(torch.sqrt(zeta.pow(2).sum(dim=1)+1e-4) - 1e-2, min=0.0)

        n0 = torch.nn.functional.softplus(self.invn0)
        n = torch.nn.functional.softplus(self.invn) + 1.0

        bij0 = (1+zeta_norm).pow(-n0)

        bij1 = self.linear(zeta)*(1+zeta_norm).unsqueeze(dim=-1).pow(-n)

        bij = bij0 + bij1.sum(dim=-1)

        return bij

class SafePow(torch.nn.Module):
    def __init__(self, exponent: list[int], clamp_value=1e12):

        super().__init__()
        self.register_buffer("exponent", torch.tensor(exponent))
        self.register_buffer("clamp_value", torch.tensor(clamp_value)**(1/self.exponent))

    def forward(self, base: torch.Tensor):
        result = torch.sign(base)*torch.pow(torch.clamp(base.abs(), max=self.clamp_value), self.exponent)
        return result

        
class ExtendedPow(torch.nn.Module):
    def __init__(self, ninput: int, max_eta: int, clamp_value=1e12, eps=1e-3):

        super().__init__()
        self.invexponent = torch.nn.Parameter(torch.randn(ninput))
        self.smoothclamp = torch.nn.Sigmoid()
        self.register_buffer("coeff",torch.tensor(max_eta-1.0))
        self.register_buffer("clamp_value",torch.tensor(clamp_value))
        self.register_buffer("eps",torch.tensor(eps))

    def forward(self, inputs):

        exponent = self.coeff*self.smoothclamp(self.invexponent) + 1.0
        clamp_value = self.clamp_value**(1/exponent)

        return torch.sign(inputs)*(torch.pow(torch.clamp(inputs.abs()+self.eps, max=clamp_value), exponent)-torch.pow(self.eps,exponent))

def l2norm(inputs):

    norm = inputs.pow(2).sum(dim=1).pow(1/2)

    return norm

def l1norm(inputs):

    norm = torch.sqrt(inputs.pow(2)+1) - 1

    return torch.clamp(norm.sum(dim=1), min=0.0)

