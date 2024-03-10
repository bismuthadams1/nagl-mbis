import os

from pkg_resources import resource_filename
from typing_extensions import Literal
import rdkit
from naglmbis.models import load_charge_model
from openff.toolkit.topology import Molecule
import numpy as np
import pandas as pd
import torch
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
from scipy.spatial.distance import cdist
import seaborn as sns
import matplotlib.pyplot as plt

# load two pre-trained charge models
charge_model = load_charge_model(charge_model="nagl-v1-mbis")
# load a model trained to scf dipole and mbis charges
charge_model_2 = load_charge_model(charge_model="nagl-v1-mbis-dipole")

def get_model_weights(model_type: Literal["charge", "volume"], model_name: str) -> str:
    """
    Get the model weights from the naglmbis package.

    """

    fn = resource_filename(
        "naglmbis", os.path.join("data", "models", model_type, model_name)
    )
    if not os.path.exists(fn):
        raise ValueError(
            f"{model_name} does not exist. If you have just added it, you'll need to re-install."
        )
    return fn


def get_latent_embedding(smiles: str) -> tuple[torch.Tensor,rdkit.Chem]:
    """Returns the latent embeddings given a smiles string
    """
    test_mol  = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    test_molrd = test_mol.to_rdkit()
    dgl_mol_2 = charge_model.return_dgl_molecule(test_molrd)
    charge_model.forward(dgl_mol_2)
    #this gives us our latent vector
    return dgl_mol_2.graph.ndata['h'], test_molrd

def total_latent_embeddings(smiles_list: list[str]) -> torch.Tensor:
    atom_index = 0
    tensors = []
    for smiles in smiles_list:
        latent_embedding, test_molrd = get_latent_embedding(smiles)
        tensors.append(latent_embedding)
        for i, atom in enumerate(test_molrd.GetAtoms()):
            atom_type = atom.GetSymbol()  # Get the atom type (C, N, O, etc.)
            embedding = latent_embedding[i].unsqueeze(1)  # Get the embedding for the atom

            # Check if this atom type has been encountered before
            if atom_type not in unique_embeddings:
                unique_embeddings[atom_type] = [embedding]
                atom_labels[atom_index] = f"{atom_type}1"
            else:
                # Compare with existing embeddings for this atom type
                is_unique = True
                for j, unique_embedding in enumerate(unique_embeddings[atom_type]):
                    #atol controls the distance
                    if torch.all(torch.isclose(embedding, unique_embedding, atol=50)):
                        atom_labels[atom_index] = f"{atom_type}{j+1}"
                        is_unique = False
                        break
                
                if is_unique:
                    unique_embeddings[atom_type].append(embedding)
                    atom_labels[atom_index] = f"{atom_type}{len(unique_embeddings[atom_type])}"

            # Set the atom label property
            atom.SetProp("atomLabel", atom_labels[atom_index])
            atom_index += 1 

        rdkit_mols.append(test_molrd)    

    total_embedings = torch.cat(tensors)