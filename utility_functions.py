from naglmbis.models.base_model import MBISGraphModel
import torch
from openff.toolkit.topology import Molecule
import rdkit
from rdkit import Chem
# from pyCheckmol import CheckMol

def get_latent_embedding(smiles: str,
                         charge_model: MBISGraphModel) -> tuple[torch.Tensor,rdkit.Chem]:
    """Returns the latent embeddings given a smiles string.

    Parameters
    ----------
    smiles: str
        Smiles string in which to grab the unique latent embedding vector
    charge_model: MBISGraphModel
        charge model to get the atom features for the latent embeddings
    
    Returns
    -------
    tensor: torch.Tensor
        Tensor of shape 128 containing the atom embeddings
    openff_mol: Molecule
        RDKit molecule of inputted SMILES
    """
    openff_mol  = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    rdkit_mol = openff_mol.to_rdkit()
    dgl_mol_2 = charge_model.return_dgl_molecule(rdkit_mol)
    charge_model.forward(dgl_mol_2)
    #this gives us our latent vector
    return dgl_mol_2.graph.ndata['h'], rdkit_mol

def total_latent_embeddings(smiles_list: list[str],
                            charge_model: MBISGraphModel, 
                            tolerence: int = 50) -> tuple[list[torch.tensor, dict[str,torch.tensor]], dict[int,str],Chem.Mol,torch.Tensor]:
    """Use this on a group of SMILES to get the total latent embeddings.
    
    Parameters
    ----------
    smiles_list: list[str]
        list of molecules to obtain a total latent embedding tensor
    charge_model: MBISGraphModel
        charge model to get the atom features for the latent embeddings
    tolerance: int
        tolerance in which to apply to the grouping procedure of the atoms

    Returns
    -------
    tensors: list[]
        
    """
    atom_index = 0
    tensors = []
    unique_embeddings = {}
    atom_labels = {}
    rdkit_mols = []

    for smiles in smiles_list:
        latent_embedding, rdkit_mol = get_latent_embedding(smiles, charge_model)
        tensors.append(latent_embedding)
        for i, atom in enumerate(rdkit_mol.GetAtoms()):
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
                    if torch.all(torch.isclose(embedding, unique_embedding, atol=tolerence)):
                        atom_labels[atom_index] = f"{atom_type}{j+1}"
                        is_unique = False
                        break
                
                if is_unique:
                    unique_embeddings[atom_type].append(embedding)
                    atom_labels[atom_index] = f"{atom_type}{len(unique_embeddings[atom_type])}"

            # Set the atom label property
            atom.SetProp("atomLabel", atom_labels[atom_index])
            atom_index += 1 

        rdkit_mols.append(rdkit_mol)    

    total_embedings = torch.cat(tensors)

    return tensors, unique_embeddings, atom_labels, rdkit_mols, total_embedings

def find_functional_groups(smiles: str) -> dict[str,tuple[tuple[int]]]:
    """Based on ths smiles and the data in the pyCheckmol csv, produce a list of functional groups
    and a tuple of atom locations of these groups. 

    Parameters
    ----------
    smiles: str
        smiles of inputed species
    Returns
    --------
    dictionary: list[str]
        list of functional groups

    """


"""
from rdkit import Chem

mol =  Chem.MolFromSmiles(smi)

# Iterate over the atoms
for i, atom in enumerate(mol.GetAtoms()):
    # For each atom, set the property "molAtomMapNumber" to a custom number, let's say, the index of the atom in the molecule
    atom.SetProp("molAtomMapNumber", str(atom.GetIdx()+1))
mol

#we know there is a thiohemiaminal
substructure_smiles = 'CC1NCCS1'
substructure = Chem.MolFromSmarts(substructure_smiles)
#get the substructure locations
mol.GetSubstructMatches(substructure)

Chem.MolFromSmiles('CC(=NN)C')

Chem.MolFromSmiles('CC(=NN((C=O)N(C)C)C)C')


"""