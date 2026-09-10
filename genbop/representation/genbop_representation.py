import torch
from typing import Dict, List, Literal
from torch import Tensor
from genbop.utils.torch_scatter import scatter
import numpy as np

class GenBOP(torch.nn.Module):

    def __init__(
            self,
            zs: List[int],
            cutoff_distance: float,
            radial_basis: torch.nn.Module,
            angular_basis: torch.nn.Module,
            angular_coupling: torch.nn.Module,
            cutoff_twobody: torch.nn.Module,
            cutoff_radial: torch.nn.Module,
            twobody: torch.nn.Module,
            output: torch.nn.Module,
            reference_mode: Literal['atom', 'structure'] = 'atom' ,
            reference_atomic_energy: List[float] = None,
            reference_structure: torch.nn.Module = None,
        ):
        super(GenBOP, self).__init__()

        self.cutoff_distance = cutoff_distance

        self.cutoff_twobody = cutoff_twobody
        self.cutoff_radial = cutoff_radial
        self.radial_basis = radial_basis

        self.angular_basis = angular_basis

        self.reference_mode = reference_mode
            
        self.register_buffer("index_map", torch.tensor([zs.index(z) if z in zs else -1 for z in range(max(zs) + 1)], dtype=torch.int64))

        nelem = len(zs)

        elem_node2undirectedge = torch.zeros((nelem,nelem),dtype=torch.long)
        count = 0
        for i in range(nelem):
            for j in range(i,nelem):

                elem_node2undirectedge[i][j] = count
                elem_node2undirectedge[j][i] = count
                count += 1

        self.register_buffer("elem_node2undirectedge",elem_node2undirectedge)
        self.register_buffer("elem_node2directedge",torch.arange(nelem*nelem).view(nelem,nelem))

        if reference_mode == 'atom':
            if reference_atomic_energy is None:
                raise ValueError("reference_atomic_energy is not defined")
            self.register_buffer("atomic_energy",torch.tensor(reference_atomic_energy, dtype=torch.float))
        else:
            if reference_structure is None:
                raise ValueError("reference_structure is not defined")
            self.register_buffer("atomic_energy",torch.tensor(np.zeros(nelem), dtype=torch.float))
            self.ref = reference_structure
            self.ref.preprocess(self.index_map,self.elem_node2directedge,self.elem_node2undirectedge)

        self.angular_coupling = angular_coupling

        self.morseenergy = twobody
        self.morseenergy.set_random_parameters()

        self.bij = output

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)

        device = None
        for p in self.parameters():
            device = p.device
            break
        if device is None:
            for b in self.buffers():
                device = b.device
                break
        if device is None:
            device = torch.device("cpu")

        if hasattr(self, "ref") and self.ref is not None:
            self.ref.to(device)

        return self
    
    def update_isolated_atom_energy(self):

        if not self.training or self.reference_mode == 'atom':
            return self.atomic_energy
        
        else:
            energy,_ = self.calc_energy(
                self.ref.element, 
                self.ref.edge_vec, 
                self.ref.row, 
                self.ref.col, 
                self.ref.direct_edge_type, 
                self.ref.undirect_edge_type,
                self.ref.structures["batch"],
                self.ref.dim_size
                )

            ene_diff = self.ref.structures['total_energy'].squeeze() - energy

            if self.ref.solvable:
                self.atomic_energy = torch.linalg.solve(self.ref.elem_counts, ene_diff)
            else:
                results = torch.linalg.lstsq(self.ref.elem_counts, ene_diff)
                self.atomic_energy = results.solution

            return self.atomic_energy

    def calc_energy(self, element, edge_vec, row, col, direct_edge_type, undirect_edge_type, batch, dim_size):

        edge_dist = torch.linalg.vector_norm(edge_vec, dim=-1)

        cutoff_rad = self.cutoff_radial(edge_dist)

        repulsive_energy, attractive_energy = self.morseenergy(edge_dist,undirect_edge_type)

        edge_vec_scaled = torch.div(edge_vec, edge_dist.view(-1,1))

        radial_basis = self.radial_basis(edge_dist)

        angular_basis = self.angular_basis(edge_vec_scaled)

        xi = self.angular_coupling(
            radial_basis,
            cutoff_rad,
            angular_basis,
            row,
            col,
            element,
            direct_edge_type
        )

        bij = self.bij(xi, direct_edge_type)

        bond_energy = self.cutoff_twobody(edge_dist)*(repulsive_energy-bij*attractive_energy)

        energy = scatter(bond_energy, batch[row], dim_size=dim_size, dim=0, reduce="sum")/2

        return energy,bij

    def forward(self, inputs: Dict[str, Tensor], compute_stress=False):

        element = self.index_map[inputs["element"]]

        batch = inputs["batch"]
        edge_vec_undirec = inputs["edge_vector"]

        row_undirec, col_undirec = torch.tensor_split(inputs["edge_index"], 2 ,dim=0)
        row_undirec = row_undirec.view(-1)
        col_undirec = col_undirec.view(-1)
        edge_dist_undirec = torch.linalg.vector_norm(edge_vec_undirec, dim=-1)
        edge_mask = edge_dist_undirec<self.cutoff_distance

        row_undirec = row_undirec[edge_mask]
        col_undirec = col_undirec[edge_mask]
        edge_vec_undirec = edge_vec_undirec[edge_mask]

        nedge = edge_vec_undirec.size(0)

        row = torch.cat((row_undirec, col_undirec), 0)
        col = torch.cat((col_undirec, row_undirec), 0)
        edge_vec = torch.cat((edge_vec_undirec, -edge_vec_undirec), 0)

        direct_edge_type = self.elem_node2directedge[element[row],element[col]]
        undirect_edge_type = self.elem_node2undirectedge[element[row],element[col]]

        edge_vec.requires_grad_(True)

        out_unique, counts  = torch.unique(batch, return_counts=True)

        dim_size = out_unique.shape[0]

        if nedge > 0:

            energy, bij = self.calc_energy(element, edge_vec, row, col, direct_edge_type, undirect_edge_type, batch, dim_size)

            pair_forces = torch.autograd.grad([energy.sum(),], [edge_vec,], create_graph=self.training)[0]

            fi = scatter(pair_forces, row, dim_size=element.shape[0], dim=0, reduce="sum")
            fj = scatter(pair_forces, col, dim_size=element.shape[0], dim=0, reduce="sum")

            forces = fi - fj

            isolated_atomic_energy = self.update_isolated_atom_energy()

            total_energy = energy + scatter(isolated_atomic_energy[element], batch, dim_size=dim_size, dim=0, reduce="sum")

            if compute_stress:
                cell = inputs["cell"]
                virial = scatter(torch.einsum("ex,ey->exy",edge_vec,pair_forces), batch[row], dim_size=dim_size, dim=0, reduce="sum")
                volume = torch.abs(torch.linalg.det(cell))
                stress = virial/volume.view(-1,1,1)
            else:
                virial = None
                stress = None

        else:

            isolated_atomic_energy = self.update_isolated_atom_energy()
            total_energy = scatter(isolated_atomic_energy[element], batch, dim_size=dim_size, dim=0, reduce="sum")

            bij = None

            forces = torch.zeros((element.shape[0],3),device=element.device)

            if compute_stress:

                virial = torch.zeros((dim_size,3,3),device=element.device)
                stress = torch.zeros((dim_size,3,3),device=element.device)

            else:
                virial = None
                stress = None
   
        return {"total_energy": total_energy, "forces" : forces, "virial": virial, "stress": stress, "bij": bij}
