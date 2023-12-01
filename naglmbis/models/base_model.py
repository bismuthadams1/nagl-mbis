# models for the nagl run

import torch
from nagl.molecules import DGLMolecule
from nagl.training import DGLMoleculeLightningModel
from rdkit import Chem


class MBISGraphModel(DGLMoleculeLightningModel):
    "A wrapper to make it easy to load and evaluate models"

    def compute_properties(self, molecule: Chem.Mol) -> dict[str, torch.Tensor]:
        dgl_molecule = DGLMolecule.from_rdkit(
            molecule, self.config.model.atom_features, self.config.model.bond_features
        )

        return self.forward(dgl_molecule)
