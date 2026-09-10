from typing import (
    Any,
    Optional,
)


import torch
from torch import Tensor
from torch_geometric.data import Data
from ase.neighborlist import neighbor_list
import numpy as np


class AtomicData(Data):
    def __init__(
        self, 
        num_nodes: Tensor = None, #[,]
        element: Tensor = None, # [n_nodes]
        edge_index: Tensor = None, # [2, n_edges]
        edge_vector: Tensor = None, # [n_edges, 3]
        total_energy: Optional[Tensor] = None, #[,]
        forces: Optional[Tensor] = None, #[n_realatoms,3]
        cell: Optional[Tensor] = None, #[1,3,3]
        #virials: Optional[Tensor] = None, #[1,3,3]
        ):
        super().__init__()
        data = {
            "num_nodes": num_nodes,
            "element": element,
            "edge_index": edge_index,
            "edge_vector": edge_vector,
            "total_energy": total_energy,
            "forces": forces,
            "cell": cell,
            #"virials": virials,
        }
        super().__init__(**data)
    
    @classmethod
    def from_atoms(cls,atoms,rcut):

        cell = atoms.get_cell()

        try:
            total_energy = atoms.get_potential_energy()
            total_energy = torch.tensor(total_energy, dtype=torch.float).view(-1, 1)
        except:
            total_energy = None
        
        try:
            forces = atoms.get_forces()
            forces = torch.tensor(forces, dtype=torch.float)
        except:
            forces = None

        i,j,D,S = neighbor_list('ijDS', atoms, rcut, self_interaction=True)
        self_edge = (i == j)
        self_edge &= np.all(S == 0, axis=1)
        edge_mask = ~self_edge
        edge_mask &= (i <= j)
        i = i[edge_mask]
        j = j[edge_mask]
        D = D[edge_mask]
        edge_index = torch.tensor(np.array([i,j]), dtype=torch.long)
        edge_vector = torch.tensor(np.array(D), dtype=torch.float)
        element = torch.tensor(atoms.get_atomic_numbers(),dtype=torch.long)
        num_nodes = torch.tensor(len(atoms))
        
        cell = torch.tensor(np.array(atoms.get_cell()), dtype=torch.float).view(1,3,3)

        return cls(
            num_nodes = num_nodes,
            element = element,
            edge_index = edge_index,
            edge_vector = edge_vector,
            total_energy = total_energy,
            forces = forces,
            cell = cell,
        )

