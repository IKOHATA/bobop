import numpy as np 
import torch
from ase.calculators.calculator import Calculator, all_changes, PropertyNotImplementedError
from ase.stress import full_3x3_to_voigt_6_stress
from genbop.utils.atomicdata import AtomicData
from torch_geometric.loader import DataLoader
from ase.neighborlist import neighbor_list

class GenBOPCalculator(Calculator):

    def __init__(
        self,
        model: torch.nn.Module,
        model_path: str,
        cutoff: float,
        device: str,
        compute_stress = False,
        **kwargs,
        ):

        Calculator.__init__(self, **kwargs)
        self.implemented_properties = [
            "energy",
            "forces",
            "stress",
        ]

        self.device = device

        self.model = model
        self.model.to(device)
        if model_path is not None:
            checkpoint = torch.load(model_path,map_location=device,weights_only=True)
            model.load_state_dict(checkpoint)

        self.model.eval()

        self.cutoff = cutoff
        self.compute_stress = compute_stress
    
    def update(self, atoms):
        if not hasattr(self,'atoms') or self.atoms != atoms:
            self.calculate(atoms)
    
    def get_potential_energy(self, atoms, force_consistent=False):
        self.update(atoms)
        if force_consistent:
            if 'free_energy' not in self.results:
                name = self.__class__.__name__
                raise PropertyNotImplementedError(
                    'Force consistent/free energy ("free_energy") '
                    'not provided by {0} calculator'.format(name))
            return self.results['free_energy']
        else:
            return self.results['energy']
    
    def get_forces(self, atoms):
        self.update(atoms)
        return self.results['forces']
    
    def get_stress(self, atoms):
        self.update(atoms)
        return self.results['stress']

    
    def calculate(self, atoms=None, properties=None):
        """
        atoms: Atoms object
            Contains positions, unit-cell, ...
        properties: list of str
            List of what needs to be calculated.  Can be any combination
            of 'energy', 'forces', 'stress'
        """

        Calculator.calculate(self, atoms)

        loader = DataLoader([AtomicData.from_atoms(atoms,self.cutoff)], batch_size=1, shuffle=False, drop_last=False)
        atomicdata_base = next(iter(loader)).to(self.device)
        atomicdata = atomicdata_base.clone()

        output = self.model(atomicdata,compute_stress=self.compute_stress)
        

        self.results['energy'] = output['total_energy'].to('cpu').detach().numpy().copy()[0]
        self.results['forces'] = output["forces"].to('cpu').detach().numpy()
        self.results['free_energy'] = self.results['energy']
        
        if self.compute_stress:
            self.results['stress'] = output["stress"].to('cpu').detach().numpy()[0]
            self.results["stress"] = full_3x3_to_voigt_6_stress(self.results["stress"])
        
        
        return self.results





