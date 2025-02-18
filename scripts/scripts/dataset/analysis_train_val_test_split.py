# collect stats on the number of molecules, range of charges and occurances of elements in each train, val and test split of the dataset
from rdkit import Chem
from rdkit.Chem import Descriptors
import deepchem as dc
import numpy as np

ps = Chem.SmilesParserParams()
ps.removeHs = False


def calculate_stats(dataset_name: str):
    formal_charges = {}
    molecular_weights = []
    elements = {}
    heavy_atom_count = []

    # load the dataset
    dataset = dc.data.DiskDataset(dataset_name)

    for smiles in dataset.ids:
        mol = Chem.MolFromSmiles(smiles, ps)
        charges = []
        for atom in mol.GetAtoms():
            charges.append(atom.GetFormalCharge())
            atomic_number = atom.GetAtomicNum()
            if atomic_number in elements:
                elements[atomic_number] += 1
            else:
                elements[atomic_number] = 1

        total_charge = sum(charges)
        if total_charge in formal_charges:
            formal_charges[total_charge] += 1
        else:
            formal_charges[total_charge] = 1

        molecular_weights.append(Descriptors.MolWt(mol))
        heavy_atom_count.append(Descriptors.HeavyAtomCount(mol))

    return formal_charges, molecular_weights, elements, heavy_atom_count


for dataset in ["maxmin-train", "maxmin-valid", "maxmin-test"]:
    charges, weights, atoms, heavy_atoms = calculate_stats(dataset_name=dataset)
    print(f"Running {dataset} number of molecules {len(weights)}")
    print("Total formal charges ", charges)
    print("Total elements", atoms)
    print(f"Average mol weight {np.mean(weights)} and std {np.std(weights)}")
    print(
        f"Average number of heavy atoms {np.mean(heavy_atoms)} and std {np.std(heavy_atoms)}"
    )
