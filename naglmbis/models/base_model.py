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
<<<<<<< HEAD
        )
class ComputePartialPolarised:
    "Compute the partially polarized properties based on a supplied dialetric constant"
    def __init__(self,
                 model_gas: MBISGraphModel,
                 model_water: MBISGraphModel,
                 alpha: float =  0.5):
        """
        Parameters
        ----------
        model_gas: MBISGraphModel
            loaded graph model for the gas phase charges
        model_water: MBISGraphModel
            loaded graph model for the water based charges
        alpha: float
            weighting constant to weight each model
        """
        
        self.model_gas = model_gas
        self.model_water = model_water
        self.alpha = alpha
        
    def compute_polarised_charges(self, molecule: Chem.Mol) -> torch.Tensor:
        """Compute polarized charges based on an openff molecule input
        
        Parameters
        ----------
        molecule: Chem.Mol
            openff molecule to calculate the charges for
        
        Returns
        -------
        torch.Tensor
            weighted average partial charges
        """
        gas_charges = self.model_gas.compute_properties(
            molecule=molecule
        )["mbis-charges"]
        
        water_charges = self.model_water.compute_properties(
            molecule=molecule
        )["mbis-charges"]
        
        return self.alpha * gas_charges + (1-self.alpha) * water_charges
        
        
=======
        )
>>>>>>> 99f689272f18e4cbc592c3319fa961c34fd4079e
