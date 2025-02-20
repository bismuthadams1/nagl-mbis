"""The purpose of this script is to 


"""
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
from openff.units import Unit, unit
from openff.recharge.esp.storage import MoleculeESPStore
from openff.toolkit.topology import Molecule, FrozenMolecule

esp_store = MoleculeESPStore("./esp-store.sqlite", cache_size=20000000)
smiles_list = esp_store.list()

# List of parquet file names
parquet_files = ['training.parquet', 'testing.parquet', 'validation.parquet']

def canonical_smiles(smiles: str, conformer: list) -> str:
    """return canonical smiles with same toolkit"""
    conformer = np.array(conformer).reshape(-1, 3)
    smiles_mol = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    smiles_mol.add_conformer(coordinates=conformer * unit.angstrom)
    return smiles_mol.to_smiles(explicit_hydrogens=False)

def canonical(smiles: str) -> str:
    """return canonical smiles with same toolkit"""
    smiles_mol = Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True)
    return smiles_mol.to_smiles(explicit_hydrogens=False)

def openff_mol(smiles: str, conformer: list) -> Molecule:
    """make an openff_mol"""
    conformer = np.array(conformer).reshape(-1, 3)
    smiles_mol = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    smiles_mol.add_conformer(coordinates=conformer * unit.angstrom)
    return smiles_mol
    
def inchi(smiles: str, conformer: list) -> str:
    """return an inchi from a smiles"""
    openffmol =  openff_mol(smiles, conformer)
    return openffmol.to_inchi(fixed_hydrogens=True)

def remap(mol1: Molecule, mol2: Molecule) -> Molecule:
    """remap mol1 to mol2 if they are isomorphic"""
    
    if (isomap := FrozenMolecule.are_isomorphic(mol1=mol1, mol2=mol2, return_atom_map=True)[0]):
        map_dictionary = isomap[1]

    mol1.remap(mapping_dict=map_dictionary)
    
    return mol1

def isclose(conf1: list, conf2: list) -> bool:
    """test if two lists are close to one another"""
    np_conf1, np_conf2 = np.array(conf1), np.array(conf2)
    return np.isclose(np_conf1, np_conf2)


# Loop through each parquet file
for parquet_file in parquet_files:
    # Read the parquet file into a pandas DataFrame
    table = pq.read_table(parquet_file)
    pandas_parquet = table.to_pandas()

    # Add the two new empty columns
    pandas_parquet['esp_column'] = pd.Series(dtype='object')
    pandas_parquet['inv_distance_column'] = pd.Series(dtype='object')
    pandas_parquet['inchi'] = pandas_parquet.apply(
        lambda row: inchi(row['smiles'], row['conformation']), axis = 1
    )    

    # Loop through the smiles_list and update the DataFrame
    for smile in smiles_list:
        record = esp_store.retrieve(smiles=smile)
        tagged_smiles = record[0].tagged_smiles
        conformer = record[0].conformer
        grid = record[0].grid_coordinates_quantity
        data_base_inchi = inchi(tagged_smiles, conformer)

        if data_base_inchi in pandas_parquet['smiles'].values:
            pandas_parquet.loc[pandas_parquet['smiles'] == tagged_smiles, 'esp_column'] = record[0].esp.tolist()
            conformer_data = pandas_parquet.loc[pandas_parquet['smiles'] == tagged_smiles, 'conformation'].values[0]
            inv_dist_data = inv_distance(conformer=conformer_data, grid=grid)
            pandas_parquet.loc[pandas_parquet['smiles'] == tagged_smiles, 'inv_distance_column'] = inv_dist_data.flatten().tolist()

    updated_table = pa.Table.from_pandas(pandas_parquet)

    # Write the updated table back to a parquet file with a new name
    output_file = parquet_file.replace(".parquet", "_esp.parquet")
    pq.write_table(updated_table, output_file)

