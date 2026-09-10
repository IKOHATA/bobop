import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from genbop.utils.atomicdataset import AtomicInMemoryDataset
from genbop.utils.torch_scatter import scatter

class ReferenceStructureContainer(torch.nn.Module):
    def __init__(self,structures,nelem,cutoff_distance):
        super(ReferenceStructureContainer, self).__init__()

        self.structures = structures
        self.nelem = nelem

        out_unique, counts  = torch.unique(self.structures["batch"], return_counts=True)
        self.dim_size = out_unique.shape[0]

        edge_index = self.structures["edge_index"]
        
        edge_vec = self.structures["edge_vector"]

        row, col = torch.tensor_split(self.structures["edge_index"], 2 ,dim=0)
        row = torch.squeeze(row)
        col = torch.squeeze(col)
        edge_dist = torch.linalg.vector_norm(edge_vec, dim=-1)
        edge_mask = edge_dist<cutoff_distance

        self.register_buffer("row", row[edge_mask])
        self.register_buffer("col", col[edge_mask])
        self.register_buffer("edge_vec", edge_vec[edge_mask])
    
    def preprocess(self,index_map,elem_node2directedge,elem_node2undirectedge):

        self.element = index_map[self.structures["element"]]

        self.direct_edge_type = elem_node2directedge[self.element[self.row],self.element[self.col]]
        self.undirect_edge_type = elem_node2undirectedge[self.element[self.row],self.element[self.col]]

        batch = self.structures["batch"]

        num_structures = int(batch.max().item() + 1)

        one_hot = torch.nn.functional.one_hot(self.element, num_classes=self.nelem).float()

        self.elem_counts = scatter(one_hot, batch, dim=0, dim_size=num_structures, reduce='sum')

        if self.nelem == torch.linalg.matrix_rank(self.elem_counts):
            self.solvable = True
        elif self.nelem > torch.linalg.matrix_rank(self.elem_counts):
            self.solvable = False
        else:
            raise ValueError("nelem < rank")
    
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device = None
        
        for p in self.buffers():
            device = p.device
            break
        if device is None:
            device = torch.device("cpu")

        for k, v in self.structures.items():
            if isinstance(v, torch.Tensor):
                self.structures[k] = v.to(device)
        
        for attr in ["element", "row", "col", "edge_vec", "direct_edge_type", "undirect_edge_type", "elem_counts"]:
            if hasattr(self, attr):
                val = getattr(self, attr)
                if isinstance(val, torch.Tensor):
                    setattr(self, attr, val.to(device))
        return self

def get_refstructure(dataset_filename, zs, mode = 'number'):
    index_map = torch.tensor([zs.index(z) if z in zs else -1 for z in range(max(zs) + 1)])
    dataset = AtomicInMemoryDataset(dataset_filename)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    element_count = torch.zeros((len(dataset),len(zs)),dtype=torch.long)
    energy_per_atom_array = []
    for n,batch in enumerate(loader):
        element = index_map[batch['element']]
        out_unique, counts  = torch.unique(element, return_counts=True)
        element_count[n,out_unique] = counts
        energy_per_atom = batch['total_energy']/len(batch['element'])
        energy_per_atom_array.append(energy_per_atom.squeeze())
    
    energy_per_atom_array = torch.tensor(energy_per_atom_array)
    
    element_count = element_count.float()

    if mode == 'number':

        norms = torch.norm(element_count, dim=1)
        sorted_indices = torch.argsort(norms.abs())
    
    elif mode == 'energy':
        sorted_indices = torch.argsort(energy_per_atom_array)

    rank = torch.linalg.matrix_rank(element_count)

    independent = []
    independent_indices = []

    for index in sorted_indices:
        e = element_count[index]
        if len(independent) == 0:
            independent.append(e)
            independent_indices.append(index)
            continue

        stacked = torch.stack(independent + [e])
        rank1 = torch.linalg.matrix_rank(torch.stack(independent))
        rank2 = torch.linalg.matrix_rank(stacked)
        if rank2 > rank1:
            independent.append(e)
            independent_indices.append(index)
            if rank2 == len(zs):
                break
    

    subset = Subset(dataset, independent_indices)
    loader_ref = DataLoader(subset, batch_size=len(subset), shuffle=False)
    structures = next(iter(loader_ref))

    return structures
