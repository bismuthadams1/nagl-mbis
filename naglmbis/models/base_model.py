# models for the nagl run

import torch
from nagl.molecules import DGLMolecule
from nagl.training import DGLMoleculeLightningModel
from rdkit import Chem

DIALETRIC_CONSTANT_WATER = 78.4

class MBISGraphModel(DGLMoleculeLightningModel):
    "A wrapper to make it easy to load and evaluate models"

    def compute_properties(self, molecule: Chem.Mol) -> dict[str, torch.Tensor]:
        dgl_molecule = DGLMolecule.from_rdkit(
            molecule, self.config.model.atom_features, self.config.model.bond_features
        )

        return self.forward(dgl_molecule)

    def return_dgl_molecule(self, molecule: Chem.Mol) -> DGLMolecule:

        return DGLMolecule.from_rdkit(
            molecule, self.config.model.atom_features, self.config.model.bond_features
        )
class ComputePartialPolarised:
    "Compute the partially polarized properties based on a supplied dialetric constant"
    def __init__(self,
                 model_gas: MBISGraphModel,
                 model_water: MBISGraphModel,
                 dialetric_constant: float):
        
        self.model_gas = model_gas
        self.model_water = model_water
        self.dialetric_constant = dialetric_constant
        
    def compute_polarised_charges(self, molecule: Chem.Mol) -> torch.Tensor:
        
        gas_charges = self.model_gas.compute_properties(
            molecule=molecule
        )["mbis-charges"]
        
        water_charges = self.model_water.compute_properties(
            molecule=molecule
        )["mbis-charges"]
        
        return (
            (water_charges - gas_charges)/
            DIALETRIC_CONSTANT_WATER
            )*self.dialetric_constant + gas_charges  
        
        