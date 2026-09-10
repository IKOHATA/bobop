import numpy as np
import torch
import torch.nn as nn
from math import factorial
from collections import OrderedDict
from genbop.utils.torch_scatter import scatter


class AngularComponent(torch.nn.Module):
    """ Angular component of the edge basis functions forked from (https://github.com/BingqingCheng/cace)
        Optimized for CPU usage (use recursive formula)
    """
    def __init__(self, l_max):
        super().__init__()
        self.l_max = l_max
        self.precompute_lxlylz()

    def precompute_lxlylz(self):
        self.lxlylz_dict = OrderedDict({l: [] for l in range(self.l_max + 1)})
        self.lxlylz_dict[0] = [(0, 0, 0)]
        for l in range(1, self.l_max + 1):
            for prev_lxlylz_combination in self.lxlylz_dict[l - 1]:
                for i in range(3):
                    lxlylz_combination = list(prev_lxlylz_combination)
                    lxlylz_combination[i] += 1
                    lxlylz_combination_tuple = tuple(lxlylz_combination)
                    if lxlylz_combination_tuple not in self.lxlylz_dict[l]:
                        self.lxlylz_dict[l].append(lxlylz_combination_tuple)
        self.lxlylz_list = self._convert_lxlylz_to_list()
        # get the start and the end index of the lxlylz_list for each l
        self.lxlylz_index = torch.zeros((self.l_max+1, 2), dtype=torch.long)
        for l in range(self.l_max+1):
            self.lxlylz_index[l, 0] = 0 if l == 0 else self.lxlylz_index[l-1, 1] 
            self.lxlylz_index[l, 1] = compute_length_lmax(l) 

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:

        computed_values = {(0, 0, 0): torch.ones(vectors.size(0), device=vectors.device, dtype=vectors.dtype)}
        for l in range(1, self.l_max + 1):
            for lxlylz_combination in self.lxlylz_dict[l]:
                prev_lxlylz_combination = tuple(l - 1 if i == lxlylz_combination.index(max(lxlylz_combination)) else l for i, l in enumerate(lxlylz_combination))
                i = lxlylz_combination.index(max(lxlylz_combination))
                computed_values[lxlylz_combination] = computed_values[prev_lxlylz_combination] * vectors[:, i]

        computed_values_list = self._convert_computed_values_to_list(computed_values)
        return torch.stack(computed_values_list, dim=1)

    def _convert_lxlylz_to_list(self):
        lxlylz_list = []
        for l, combinations in self.lxlylz_dict.items():
            lxlylz_list.extend(combinations)
        return lxlylz_list

    def _convert_computed_values_to_list(self, computed_values):
        return [computed_values[comb] for comb in self.lxlylz_list]

    def get_lxlylz_list(self):
        if self.lxlylz_list is None:
            raise ValueError("You must call forward before getting lxlylz_list")
        return self.lxlylz_list

    def get_lxlylz_dict(self):
        return self.lxlylz_dict
 
    def get_lxlylz_index(self):
        return self.lxlylz_index


    def __repr__(self):
        return f"AngularComponent(l_max={self.l_max})"

def compute_length_lxlylz(l):
    """ compute the length of the lxlylz list based on l """
    return int((l+1)*(l+2)/2)

def compute_length_lmax(l_max):
    """ compute the length of the lxlylz list based on l_max """
    return int((l_max+1)*(l_max+2)*(l_max+3)/6)

class AngularCoupling(torch.nn.Module):
    def __init__(
        self,
        l_max: int,
        lxlylz_list,
        n_rbf: int,
        nelem: int,
        body_order: int,
        embedding_dim: int,
        l_max_list: list[int] = None,
    ):
        super().__init__()

        assert body_order <= 5 and body_order >= 3

        self.lxlylz_list = lxlylz_list
        self.l_max = l_max
        self.body_order = body_order
        self.embedding_dim = embedding_dim
        
        self.n_rbf = n_rbf
        self.nelem = nelem
        self.n_basis_dict = []
        self.xi_size = []

        self.make_coupling_indices()
        self.coupling_threebody()
        if self.body_order >= 4:
            self.coupling_fourbody()
        if self.body_order == 5:
            self.coupling_fivebody()

        self.element_embedding_chi = torch.nn.Embedding(self.nelem*self.nelem,self.n_rbf,max_norm=1.0)

        if self.embedding_dim == 1:
            self.element_embedding_chi.weight.data.fill_(1)
            self.element_embedding_chi.weight.requires_grad = False

        self.lamb = torch.nn.Parameter(torch.rand(self.n_rbf))

        self.register_buffer('dummy_edge_type',torch.arange(self.nelem*self.nelem))

    def make_coupling_indices(self):

        lxlylz_factorial_coef_list = []
        l_list = []
        self.l_reverse_dict = OrderedDict()

        for n,lxlylz in enumerate(self.lxlylz_list):
            l = sum(lxlylz)
            lxlylz_sorted = sorted(lxlylz, reverse=True)
            
            coef = factorial(l)
            for lxly in lxlylz_sorted:
                coef //= factorial(lxly)
            
            lxlylz_factorial_coef_list.append(coef)
            l_list.append(l)
            self.l_reverse_dict[lxlylz] = n
    
        
        self.register_buffer("lxlylz_factorial_coef_list", torch.tensor(lxlylz_factorial_coef_list,dtype=torch.long))
        self.register_buffer("l_list", torch.tensor(l_list,dtype=torch.long))
    
        """ self.l_reverse_dict: reverse lookup ordered dictionary; key: (lx,ly,lz), value: index """
        """ self.lxlylz_factorial_coef_list: list of trinomial coefficient C(l) corresponding to l_reverse_dict """
        """ self.l_list: list of l = lx + ly + lz corresponding to l_reverse_dict """
    
    def coupling_threebody(self):

        l1_dict = OrderedDict()
        n = 0
        for l1 in range(self.l_max+1):
            l1_dict[l1] = n
            n += 1


        threebody_flattened_indices_dict = {
            (j, k): idx
            for idx, (j, k) in enumerate([(j, k) for j in range(self.n_rbf) for k in range(self.l_max+1)])
        }

        """ flattened_indices for threebody features (n,l)"""
        self.threebody_flattened_indices_dict = threebody_flattened_indices_dict 

        self.n_basis_dict.append(self.threebody_flattened_indices_dict)
    
        self.n_angular_basis_three = len(self.threebody_flattened_indices_dict)

        lxlylz1_indices = []
        threebody_coupling_indices = []
        threebody_to_angular = []
        for n1,l1 in enumerate(self.l_list):
            lxlylz1_indices.append(n1)
            threebody_coupling_indices.append(n1)

            threebody_to_angular.append(l1_dict[l1.item()])

        """ threebody_coupling_indices: index of angular basis """
        self.register_buffer("threebody_coupling_indices",torch.tensor(threebody_coupling_indices,dtype=torch.long))

        """ lxlylz_threebody_factorial_coef_list: trinomial coefficient C(l1) corresponding to threebody_coupling_indices """
        self.register_buffer("lxlylz_threebody_factorial_coef_list", self.lxlylz_factorial_coef_list[lxlylz1_indices])

        """ threebody_to_angular: scattering indices from l1 to angular momentum number L1 """
        self.register_buffer("threebody_to_angular",torch.tensor(threebody_to_angular,dtype=torch.long))

    def coupling_fourbody(self):


        fourbody_flattened_indices_dict = OrderedDict()
        l1l2l3_dict = OrderedDict()
        n = 0
        m = 0
        
        for l1 in range(self.l_max+1):
            for l2 in range(self.l_max+1):
                if l1 <= l2 and l1+l2 <= self.l_max:
                    for l3 in range(self.l_max+1):
                        if l3 > 0 and l1+l3 <= self.l_max and l2+l3 <= self.l_max:
                            l1l2l3_dict[(l1,l2,l3)] = n
                            n += 1

        flattened_indices = [(j, k, l) for j in range(self.n_rbf) for k in range(self.n_rbf) for l in range(len(l1l2l3_dict))]
        keys = list(l1l2l3_dict.keys())
        mask = []
        n2_indices = []
        n1_l1l3_indices = []
        n1_indices = []
        n2_l2l3_indices = []
        for index in flattened_indices:
            angular_ind = keys[index[2]]
            n1 = index[0]
            n2 = index[1]
            l1,l2,l3 = angular_ind
            if l1 != l2 or n1 <= n2:
                if l1 == l2 and n1 == n2:
                    multiplicity = 1
                else:
                    multiplicity = 2
                mask.append(True)
                fourbody_flattened_indices_dict[(n1,n2,l1l2l3_dict[(l1,l2,l3)])] = (m,multiplicity)
                n2_indices.append(n2)
                n1_l1l3_indices.append(self.threebody_flattened_indices_dict[(n1,l1+l3)])
                n1_indices.append(n1)
                n2_l2l3_indices.append(self.threebody_flattened_indices_dict[(n2,l2+l3)])
                m += 1
            else:
                mask.append(False)

        """ mask for remove duplications """
        self.register_buffer("fourbody_mask",torch.tensor(mask,dtype=torch.bool))
        """ fourbody_to_threebody_indices: indices of threebody features used for self-interaction correction [n1,n2,n1(l1+l3),n2(l2+l3)] """
        self.register_buffer("fourbody_to_threebody_indices",torch.tensor(np.array([n1_indices,n2_indices,n1_l1l3_indices,n2_l2l3_indices]),dtype=torch.long))

        for l1 in range(self.l_max+1):
            for l2 in range(l1,self.l_max+1):
                l1l2l3_dict[(l1,l2,0)] = n
                n += 1

        threebody2fourbody_indices = []
        for n1 in range(self.n_rbf):
            for n2 in range(self.n_rbf):
                for l1 in range(self.l_max+1):
                    for l2 in range(l1,self.l_max+1):
                        if l1 != l2 or n1 <= n2:
                            if l1 == l2 and n1 == n2:
                                multiplicity = 1
                            else:
                                multiplicity = 2
                            fourbody_flattened_indices_dict[(n1,n2,l1l2l3_dict[(l1,l2,0)])] = (m,multiplicity)
                            m += 1
                            threebody2fourbody_indices.append([self.threebody_flattened_indices_dict[(n1,l1)],self.threebody_flattened_indices_dict[(n2,l2)]])
    
        
        """ flattened_indices for fourbody features (n1,n2,[l1,l2,l3]): (count, inverse_flag)"""
        self.fourbody_flattened_indices_dict = fourbody_flattened_indices_dict

        self.n_basis_dict.append(self.fourbody_flattened_indices_dict)

        self.n_angular_basis_four = len(self.fourbody_flattened_indices_dict)

        self.register_buffer("threebody2fourbody_indices",torch.tensor(threebody2fourbody_indices,dtype=torch.long))
        self.l1l2l3_dict = l1l2l3_dict

        lxlylz1_indices = []
        lxlylz2_indices = []
        lxlylz3_indices = []
        lxlylz12_indices = []
        lxlylz13_indices = []
        lxlylz23_indices = []
        n1n2l3_dict = OrderedDict()
        fourbody_to_n1n2l3 = []
        n1n2l3_to_angular = []
        n = 0
        for n1,l1 in enumerate(self.l_list):
            if l1 <= self.l_max:
                for n2,l2 in enumerate(self.l_list):
                    if l2 <= self.l_max:
                        if l1 <= l2 and l1 + l2 <= self.l_max:
                            for l3 in range(self.l_max+1):
                                if l3 > 0 and l1 + l3 <= self.l_max and l2 + l3 <= self.l_max:
                                    n1n2l3_dict[(n1,n2,l3)] = n     
                                    n += 1

                                    lxlylz12 = tuple([sum(x) for x in zip(self.lxlylz_list[n1],self.lxlylz_list[n2])])
                                    lxlylz12_indices.append(self.l_reverse_dict[lxlylz12])

                                    n1n2l3_to_angular.append(l1l2l3_dict[(l1.item(),l2.item(),l3)])

                            for n3,l3 in enumerate(self.l_list):
                                if l3 <= self.l_max:
                                    if l3 > 0 and l1 + l3 <= self.l_max and l2 + l3 <= self.l_max:

                                        lxlylz1_indices.append(n1)
                                        lxlylz2_indices.append(n2)
                                        lxlylz3_indices.append(n3)

                                        lxlylz13 = tuple([sum(x) for x in zip(self.lxlylz_list[n1],self.lxlylz_list[n3])])
                                        lxlylz13_indices.append(self.l_reverse_dict[lxlylz13])

                                        lxlylz23 = tuple([sum(x) for x in zip(self.lxlylz_list[n2],self.lxlylz_list[n3])])
                                        lxlylz23_indices.append(self.l_reverse_dict[lxlylz23])

                                        fourbody_to_n1n2l3.append(n1n2l3_dict[(n1,n2,l3.item())])
        

        """ fourbody_coupling_indices: triplet indices of (l1+l3,l2+l3) for four-body contribution """
        self.register_buffer("fourbody_coupling_indices",torch.tensor(np.array([lxlylz13_indices,lxlylz23_indices]).T,dtype=torch.long))

        self.register_buffer("lxlylz12_indices",torch.tensor(lxlylz12_indices,dtype=torch.long))

        """ lxlylz_fourbody_factorial_coef_list: products of trinomial coefficients C(l1)*C(l2)*C(l3) corresponding to self.fourbody_coupling_indices"""
        self.register_buffer(
            "lxlylz_fourbody_factorial_coef_list", 
            self.lxlylz_factorial_coef_list[lxlylz1_indices]*self.lxlylz_factorial_coef_list[lxlylz2_indices]*self.lxlylz_factorial_coef_list[lxlylz3_indices]
        )

        """ fourbody_to_angular: scattering indices from (l1+l2,l1+l3,l2+l3) to index of triple angular momentum numbers (L1,L2,L3)  """
        self.register_buffer("fourbody_to_n1n2l3", torch.tensor(fourbody_to_n1n2l3,dtype=torch.long))

        self.register_buffer("n1n2l3_to_angular", torch.tensor(n1n2l3_to_angular,dtype=torch.long))
    
    def coupling_fivebody(self):
        fivebody_flattened_indices_dict = OrderedDict()
        l1l2l3l4l5_dict = OrderedDict()

        n = 0
        m = 0
        
        for l1 in range(self.l_max+1):
            for l2 in range(self.l_max+1):
                for l3 in range(self.l_max+1):
                    if l1 <= l3 and l1+l2+l3 <= self.l_max:
                        for l4 in range(self.l_max+1):
                            if l1+l4 <= self.l_max:
                                for l5 in range(self.l_max+1):
                                    if l2+l3+l4 <= self.l_max and l2+l4+l5 <= self.l_max and l3+l5 <= self.l_max and l4 > 0 and l5 > 0:
                                        if l1 != l3 or l4 <= l5:
                                            l1l2l3l4l5_dict[(l1,l2,l3,l4,l5)] = n
                                            n += 1
        
        flattened_indices = [(i, j, k, l) for i in range(self.n_rbf) for j in range(self.n_rbf) for k in range(self.n_rbf) for l in range(len(l1l2l3l4l5_dict))]
        keys = list(l1l2l3l4l5_dict.keys())
        n1_indices = []
        n2_indices = []
        n3_indices = []
        n1_n2_l1_l2l5_l4_indices = []
        n1_n3_l1l4_l3l5_0_indices = []
        n2_n3_l2l4_l3_l5_indices = []
        n1_l1l4_indices = []
        n2_l2l4l5_indices = []
        n3_l3l5_indices = []
        mask = []
        for index in flattened_indices:
            
            angular_ind = keys[index[3]]
            n1 = index[0]
            n2 = index[1]
            n3 = index[2]
            l1,l2,l3,l4,l5 = angular_ind
            if l1 != l3 or l4 != l5 or n1 <= n3:
                if l1 == l3 and l4 == l5 and n1 == n3:
                    multiplicity = 1
                else:
                    multiplicity = 2
                mask.append(True)
                fivebody_flattened_indices_dict[(n1,n2,n3,l1l2l3l4l5_dict[(l1,l2,l3,l4,l5)])] = (m,multiplicity)
                n3_indices.append(n3)
                
                if l1 < l2 + l5:
                    n1_n2_l1_l2l5_l4_indices.append(self.fourbody_flattened_indices_dict[(n1,n2,self.l1l2l3_dict[(l1,l2+l5,l4)])][0])
                elif l1 > l2 + l5:
                    n1_n2_l1_l2l5_l4_indices.append(self.fourbody_flattened_indices_dict[(n2,n1,self.l1l2l3_dict[(l2+l5,l1,l4)])][0])
                elif n1 <= n2:
                    n1_n2_l1_l2l5_l4_indices.append(self.fourbody_flattened_indices_dict[(n1,n2,self.l1l2l3_dict[(l1,l2+l5,l4)])][0])
                else:
                    n1_n2_l1_l2l5_l4_indices.append(self.fourbody_flattened_indices_dict[(n2,n1,self.l1l2l3_dict[(l1,l2+l5,l4)])][0])
                    
                n2_indices.append(n2)

                if l1 + l4 < l3 + l5:
                    n1_n3_l1l4_l3l5_0_indices.append(self.fourbody_flattened_indices_dict[(n1,n3,self.l1l2l3_dict[(l1+l4,l3+l5,0)])][0])
                elif l1 + l4 > l3 + l5:
                    n1_n3_l1l4_l3l5_0_indices.append(self.fourbody_flattened_indices_dict[(n3,n1,self.l1l2l3_dict[(l3+l5,l1+l4,0)])][0])
                elif n1 <= n3:
                    n1_n3_l1l4_l3l5_0_indices.append(self.fourbody_flattened_indices_dict[(n1,n3,self.l1l2l3_dict[(l1+l4,l3+l5,0)])][0])
                else:
                    n1_n3_l1l4_l3l5_0_indices.append(self.fourbody_flattened_indices_dict[(n3,n1,self.l1l2l3_dict[(l1+l4,l3+l5,0)])][0])

                n1_indices.append(n1)
                if l2 + l4 < l3:
                    n2_n3_l2l4_l3_l5_indices.append(self.fourbody_flattened_indices_dict[(n2,n3,self.l1l2l3_dict[(l2+l4,l3,l5)])][0])
                elif l2 + l4 > l3:
                    n2_n3_l2l4_l3_l5_indices.append(self.fourbody_flattened_indices_dict[(n3,n2,self.l1l2l3_dict[(l3,l2+l4,l5)])][0])
                elif n2 <= n3:
                    n2_n3_l2l4_l3_l5_indices.append(self.fourbody_flattened_indices_dict[(n2,n3,self.l1l2l3_dict[(l2+l4,l3,l5)])][0])
                else:
                    n2_n3_l2l4_l3_l5_indices.append(self.fourbody_flattened_indices_dict[(n3,n2,self.l1l2l3_dict[(l2+l4,l3,l5)])][0])
                
                n1_l1l4_indices.append(self.threebody_flattened_indices_dict[(n1,l1+l4)])
                n2_l2l4l5_indices.append(self.threebody_flattened_indices_dict[(n2,l2+l4+l5)])
                n3_l3l5_indices.append(self.threebody_flattened_indices_dict[(n3,l3+l5)])


                m += 1
            else:
                mask.append(False)
        
        """ mask for remove duplications """
        self.register_buffer("fivebody_mask",torch.tensor(mask,dtype=torch.bool))
        """ fivebody_to_threebody_indices: indices of fourbody features used for self-interaction correction [n1,n2,n3,n1(l1+l3),n2(l2+l3)] """

        self.register_buffer("fivebody_to_fourthreebody_indices",torch.tensor(np.array([
            n1_indices,n2_indices,n3_indices,
            n1_n2_l1_l2l5_l4_indices,n1_n3_l1l4_l3l5_0_indices,n2_n3_l2l4_l3_l5_indices,
            n1_l1l4_indices,n2_l2l4l5_indices,n3_l3l5_indices]),dtype=torch.long))
        
        for l1 in range(self.l_max+1):
            for l2 in range(self.l_max+1):
                for l3 in range(self.l_max+1):
                    for l4 in range(self.l_max+1):
                        if l1 <= l2:
                            if l4 == 0:
                                if l2 <= l3:
                                    l1l2l3l4l5_dict[(l1,l2,l3,l4,0)] = n
                                    n += 1
                            elif l1 + l2 <= self.l_max and l1 + l4 <= self.l_max and l2 + l4 <= self.l_max:
                                l1l2l3l4l5_dict[(l1,l2,l3,l4,0)] = n
                                n += 1

        
        fourthreebody2fivebody_indices = []
        for n1 in range(self.n_rbf):
            for n2 in range(self.n_rbf):
                for n3 in range(self.n_rbf):
                    for l1 in range(self.l_max+1):
                        for l2 in range(self.l_max+1):
                            for l3 in range(self.l_max+1):
                                for l4 in range(self.l_max+1):
                                    if l1 <= l2:
                                        if l4 == 0:
                                            if l2 <= l3:
                                                if l1 == l2 and l2 == l3:
                                                    if n1 <= n2 and n2 <= n3:
                                                        if n1 == n2 and n2 == n3:
                                                            multiplicity = 1
                                                        elif n1 == n2:
                                                            multiplicity = 3
                                                        elif n2 == n3:
                                                            multiplicity = 3
                                                        else:
                                                            multiplicity = 6

                                                        #print(n1,n2,n3,l1,l2,l3,l4,0,multiplicity)
                                                        fivebody_flattened_indices_dict[(n1,n2,n3,l1l2l3l4l5_dict[(l1,l2,l3,l4,0)])] = (m,multiplicity)
                                                        m += 1
                                                        fourthreebody2fivebody_indices.append([
                                                            self.fourbody_flattened_indices_dict[(n1,n2,self.l1l2l3_dict[(l1,l2,l4)])][0],self.threebody_flattened_indices_dict[(n3,l3)]
                                                            ])
                                                elif l1 == l2:
                                                    if n1 <= n2:
                                                        if n1 == n2:
                                                            multiplicity = 3
                                                        else:
                                                            multiplicity = 6
                                                        #print(n1,n2,n3,l1,l2,l3,l4,0,multiplicity)
                                                        fivebody_flattened_indices_dict[(n1,n2,n3,l1l2l3l4l5_dict[(l1,l2,l3,l4,0)])] = (m,multiplicity)
                                                        m += 1
                                                        fourthreebody2fivebody_indices.append([
                                                            self.fourbody_flattened_indices_dict[(n1,n2,self.l1l2l3_dict[(l1,l2,l4)])][0],self.threebody_flattened_indices_dict[(n3,l3)]
                                                            ])
                                                elif l2 == l3:
                                                    if n2 <= n3:
                                                        if n2 == n3:
                                                            multiplicity = 3
                                                        else:
                                                            multiplicity = 6
                                                        #print(n1,n2,n3,l1,l2,l3,l4,0,multiplicity)
                                                        fivebody_flattened_indices_dict[(n1,n2,n3,l1l2l3l4l5_dict[(l1,l2,l3,l4,0)])] = (m,multiplicity)
                                                        m += 1
                                                        fourthreebody2fivebody_indices.append([
                                                            self.fourbody_flattened_indices_dict[(n1,n2,self.l1l2l3_dict[(l1,l2,l4)])][0],self.threebody_flattened_indices_dict[(n3,l3)]
                                                            ])
                                                else:
                                                    multiplicity = 6
                                                    #print(n1,n2,n3,l1,l2,l3,l4,0,multiplicity)
                                                    fivebody_flattened_indices_dict[(n1,n2,n3,l1l2l3l4l5_dict[(l1,l2,l3,l4,0)])] = (m,multiplicity)
                                                    m += 1
                                                    fourthreebody2fivebody_indices.append([
                                                        self.fourbody_flattened_indices_dict[(n1,n2,self.l1l2l3_dict[(l1,l2,l4)])][0],self.threebody_flattened_indices_dict[(n3,l3)]
                                                        ])
                                        else:
                                            if l1 + l2 <= self.l_max and l1 + l4 <= self.l_max and l2 + l4 <= self.l_max:
                                                if l1 != l2 or n1 <= n2:
                                                    if n1 == n2 and l1 == l2:
                                                        multiplicity = 2
                                                    else:
                                                        multiplicity = 4
                                                    #print(n1,n2,n3,l1,l2,l3,l4,0,multiplicity)
                                                    fivebody_flattened_indices_dict[(n1,n2,n3,l1l2l3l4l5_dict[(l1,l2,l3,l4,0)])] = (m,multiplicity)
                                                    m += 1
                                                    fourthreebody2fivebody_indices.append([
                                                        self.fourbody_flattened_indices_dict[(n1,n2,self.l1l2l3_dict[(l1,l2,l4)])][0],self.threebody_flattened_indices_dict[(n3,l3)]
                                                        ])
        
        """ flattened_indices for fourbody features (n1,n2,n3,[l1,l2,l3,l4,l5]): (count, inverse_flag)"""
        self.fivebody_flattened_indices_dict = fivebody_flattened_indices_dict

        self.n_basis_dict.append(self.fivebody_flattened_indices_dict)

        self.n_angular_basis_five = len(self.fivebody_flattened_indices_dict)

        self.register_buffer("fourthreebody2fivebody_indices",torch.tensor(fourthreebody2fivebody_indices,dtype=torch.long))
        self.l1l2l3l4l5_dict = l1l2l3l4l5_dict

        #print(self.l1l2l3l4l5_dict)
                                                    
        lxlylz1_indices = []
        lxlylz2_indices = []
        lxlylz3_indices = []
        lxlylz4_indices = []
        lxlylz5_indices = []
        lxlylz123_indices = []
        lxlylz14_indices = []
        lxlylz245_indices = []
        lxlylz35_indices = []
        n1n2n3l4l5_dict = OrderedDict()
        fivebody_to_n1n2n3l4l5 = []
        n1n2n3l4l5_to_angular = []
        n = 0
        for n1,l1 in enumerate(self.l_list):
            if l1 <= self.l_max:
                for n2,l2 in enumerate(self.l_list):
                    if l2 <= self.l_max:
                        for n3,l3 in enumerate(self.l_list):
                            if l3 <= self.l_max:
                                if l1 <= l3 and l1+l2+l3 <= self.l_max:
                                    for l4 in range(self.l_max+1):
                                        if l1+l4 <= self.l_max:
                                            for l5 in range(self.l_max+1):
                                                if l2+l3+l4 <= self.l_max and l2+l4+l5 <= self.l_max and l3+l5 <= self.l_max and l4 > 0 and l5 > 0:
                                                    if l1 != l3 or l4 <= l5:
                                                        n1n2n3l4l5_dict[(n1,n2,n3,l4,l5)] = n     
                                                        n += 1

                                                        lxlylz123 = tuple([sum(x) for x in zip(self.lxlylz_list[n1],self.lxlylz_list[n2],self.lxlylz_list[n3])])
                                                        lxlylz123_indices.append(self.l_reverse_dict[lxlylz123])

                                                        n1n2n3l4l5_to_angular.append(l1l2l3l4l5_dict[(l1.item(),l2.item(),l3.item(),l4,l5)])
                                    for n4,l4 in enumerate(self.l_list):
                                        if l1+l4 <= self.l_max:
                                            for n5,l5 in enumerate(self.l_list):
                                                if l2+l3+l4 <= self.l_max and l2+l4+l5 <= self.l_max and l3+l5 <= self.l_max and l4 > 0 and l5 > 0:
                                                    if l1 != l3 or l4 <= l5:
                                                        lxlylz1_indices.append(n1)
                                                        lxlylz2_indices.append(n2)
                                                        lxlylz3_indices.append(n3)
                                                        lxlylz4_indices.append(n4)
                                                        lxlylz5_indices.append(n5)

                                                        lxlylz14 = tuple([sum(x) for x in zip(self.lxlylz_list[n1],self.lxlylz_list[n4])])
                                                        lxlylz14_indices.append(self.l_reverse_dict[lxlylz14])

                                                        lxlylz245 = tuple([sum(x) for x in zip(self.lxlylz_list[n2],self.lxlylz_list[n4],self.lxlylz_list[n5])])
                                                        lxlylz245_indices.append(self.l_reverse_dict[lxlylz245])

                                                        lxlylz35 = tuple([sum(x) for x in zip(self.lxlylz_list[n3],self.lxlylz_list[n5])])
                                                        lxlylz35_indices.append(self.l_reverse_dict[lxlylz35])

                                                        fivebody_to_n1n2n3l4l5.append(n1n2n3l4l5_dict[(n1,n2,n3,l4.item(),l5.item())])

        """ fivebody_coupling_indices: triplet indices of (l1+l4,l2+l4+l5,l3+l5) for five-body contribution """
        self.register_buffer("fivebody_coupling_indices",torch.tensor(np.array([lxlylz14_indices,lxlylz245_indices,lxlylz35_indices]).T,dtype=torch.long))

        self.register_buffer("lxlylz123_indices",torch.tensor(lxlylz123_indices,dtype=torch.long))

        """ lxlylz_fivebody_factorial_coef_list: products of trinomial coefficients C(l1)*C(l2)*C(l3) corresponding to self.fivebody_coupling_indices"""
        self.register_buffer(
            "lxlylz_fivebody_factorial_coef_list", 
            self.lxlylz_factorial_coef_list[lxlylz1_indices]*self.lxlylz_factorial_coef_list[lxlylz2_indices]*self.lxlylz_factorial_coef_list[lxlylz3_indices]*self.lxlylz_factorial_coef_list[lxlylz4_indices]*self.lxlylz_factorial_coef_list[lxlylz5_indices]
        )

        self.register_buffer("fivebody_to_n1n2n3l4l5", torch.tensor(fivebody_to_n1n2n3l4l5,dtype=torch.long))

        self.register_buffer("n1n2n3l4l5_to_angular", torch.tensor(n1n2n3l4l5_to_angular,dtype=torch.long))
        


    def forward(self, radial_basis, cutoff_rad, angular_basis, row, col, element, edge_type):

        radial_basis = radial_basis.pow(self.lamb)

        rad_basis = (1/radial_basis)*cutoff_rad.view(-1,1)

        particle_basis = torch.einsum('er,ea->era', rad_basis, angular_basis)

        P_basis = scatter(
            particle_basis, 
            row+element.shape[0]*edge_type, 
            dim_size=element.shape[0]*self.nelem*self.nelem, 
            dim=0, 
            reduce="sum").view(self.nelem*self.nelem,element.shape[0],self.n_rbf,len(self.lxlylz_list)).transpose(0, 1)

        element_embedding_chi = self.element_embedding_chi(self.dummy_edge_type)

        A_basis = torch.einsum('nkra,kr->nra',P_basis,element_embedding_chi)

        B_basis = []

        if self.body_order >= 3:

            B_basis_nu2 = self.lxlylz_threebody_factorial_coef_list*A_basis[:,:,self.threebody_coupling_indices]

            sym_basis_nu2 = (
                scatter(torch.einsum('era,ea->era', B_basis_nu2[row], angular_basis[:,self.threebody_coupling_indices]), self.threebody_to_angular, dim=-1, reduce="sum") 
            )

            xi_threebody = torch.einsum('erl,er->erl', sym_basis_nu2, radial_basis)


            B_threebody = (xi_threebody - torch.einsum('e,er->er',cutoff_rad,element_embedding_chi[edge_type]).unsqueeze(dim=-1)).flatten(start_dim=1)

            B_basis.append(B_threebody)

        if self.body_order >= 4:

            B_basis_nu3 = scatter(
                self.lxlylz_fourbody_factorial_coef_list*torch.einsum('nra,nsa->nrsa',A_basis[:,:,self.fourbody_coupling_indices[:,0]],A_basis[:,:,self.fourbody_coupling_indices[:,1]]),
                self.fourbody_to_n1n2l3,
                dim = -1,
                reduce = "sum"
            )

            sym_basis_nu3 = (
                scatter(torch.einsum('ersa,ea->ersa', B_basis_nu3[row], angular_basis[:,self.lxlylz12_indices]), self.n1n2l3_to_angular, dim=-1, reduce="sum")
            )

            xi_fourbody = torch.einsum('ersl,er,es->ersl', sym_basis_nu3, radial_basis, radial_basis).flatten(start_dim=1)[:,self.fourbody_mask]

            B_fourbody = xi_fourbody - cutoff_rad.view(-1,1)*(
                element_embedding_chi[edge_type][:,self.fourbody_to_threebody_indices[1]]*B_threebody[:,self.fourbody_to_threebody_indices[2]]
                + element_embedding_chi[edge_type][:,self.fourbody_to_threebody_indices[0]]*B_threebody[:,self.fourbody_to_threebody_indices[3]]
                ) - cutoff_rad.view(-1,1).pow(2)*element_embedding_chi[edge_type][:,self.fourbody_to_threebody_indices[1]]*element_embedding_chi[edge_type][:,self.fourbody_to_threebody_indices[0]]

            B_fourbody = torch.cat([B_fourbody,B_threebody[:,self.threebody2fourbody_indices[:,0]]*B_threebody[:,self.threebody2fourbody_indices[:,1]]],dim=-1)

            B_basis.append(B_fourbody)
        
        if self.body_order >= 5:
            B_basis_nu4 = scatter(
                self.lxlylz_fivebody_factorial_coef_list*torch.einsum('nra,nsa,nta->nrsta',
                A_basis[:,:,self.fivebody_coupling_indices[:,0]],A_basis[:,:,self.fivebody_coupling_indices[:,1]],A_basis[:,:,self.fivebody_coupling_indices[:,2]]),
                self.fivebody_to_n1n2n3l4l5,
                dim = -1,
                reduce = "sum"
            )
            sym_basis_nu4 = (
                scatter(torch.einsum('ersta,ea->ersta', B_basis_nu4[row], angular_basis[:,self.lxlylz123_indices]), self.n1n2n3l4l5_to_angular, dim=-1, reduce="sum")
            )

            xi_fivebody = torch.einsum('erstl,er,es,et->erstl', sym_basis_nu4, radial_basis, radial_basis, radial_basis).flatten(start_dim=1)[:,self.fivebody_mask]

            B_fivebody = xi_fivebody - cutoff_rad.view(-1,1)*(
                element_embedding_chi[edge_type][:,self.fivebody_to_fourthreebody_indices[2]]*B_fourbody[:,self.fivebody_to_fourthreebody_indices[3]]
                + element_embedding_chi[edge_type][:,self.fivebody_to_fourthreebody_indices[1]]*B_fourbody[:,self.fivebody_to_fourthreebody_indices[4]]
                + element_embedding_chi[edge_type][:,self.fivebody_to_fourthreebody_indices[0]]*B_fourbody[:,self.fivebody_to_fourthreebody_indices[5]]
                ) - cutoff_rad.view(-1,1).pow(2)*(
                element_embedding_chi[edge_type][:,self.fivebody_to_fourthreebody_indices[1]]*element_embedding_chi[edge_type][:,self.fivebody_to_fourthreebody_indices[2]]*B_threebody[:,self.fivebody_to_fourthreebody_indices[6]]
                + element_embedding_chi[edge_type][:,self.fivebody_to_fourthreebody_indices[0]]*element_embedding_chi[edge_type][:,self.fivebody_to_fourthreebody_indices[2]]*B_threebody[:,self.fivebody_to_fourthreebody_indices[7]]
                + element_embedding_chi[edge_type][:,self.fivebody_to_fourthreebody_indices[0]]*element_embedding_chi[edge_type][:,self.fivebody_to_fourthreebody_indices[1]]*B_threebody[:,self.fivebody_to_fourthreebody_indices[8]]
                ) - cutoff_rad.view(-1,1).pow(3)*(
                    element_embedding_chi[edge_type][:,self.fivebody_to_fourthreebody_indices[1]]
                    *element_embedding_chi[edge_type][:,self.fivebody_to_fourthreebody_indices[0]]
                    *element_embedding_chi[edge_type][:,self.fivebody_to_fourthreebody_indices[2]]
                    )

            B_fivebody = torch.cat([B_fivebody,B_fourbody[:,self.fourthreebody2fivebody_indices[:,0]]*B_threebody[:,self.fourthreebody2fivebody_indices[:,1]]],dim=-1)

            B_basis.append(B_fivebody)
        return B_basis








