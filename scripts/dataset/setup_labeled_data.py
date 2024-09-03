import h5py
import pyarrow
import pyarrow.parquet
from openff.units import unit
from collections import defaultdict, OrderedDict
import deepchem as dc
import numpy as np
import typing
from openff.recharge.grids import GridGenerator, GridSettingsType, LatticeGridSettings
from build_multipoles import ESPCalculator
from qcportal import PortalClient
from qcportal.dataset_models import BaseDataset
from qcportal.singlepoint import SinglepointRecord
from openff.toolkit.topology import Molecule
import pandas as pd
import logging
from tqdm import tqdm
# setup the parquet datasets using the splits generated by deepchem

esp_calculator = ESPCalculator()

client = PortalClient("api.qcarchive.molssi.org")
data_set = client.get_dataset(dataset_type='singlepoint',dataset_name='MLPepper RECAP Optimized Fragments v1.0')
    
#grid settings assigned here
grid_settings =  LatticeGridSettings(
    type="fcc", spacing=0.5, inner_vdw_scale=1.4, outer_vdw_scale=2.0
)
def build_grid(molecule: Molecule, conformer: unit.Quantity, grid_settings: GridSettingsType) -> unit.Quantity:
    grid = GridGenerator.generate(molecule, conformer, grid_settings)
    return grid

#create the parquet dataset
def create_parquet_dataset(
    parquet_name: str,
    deep_chem_dataset: dc.data.DiskDataset,
    reference_dataset: dict[int,SinglepointRecord],
):
    dataset_keys = deep_chem_dataset.X
    dataset_smiles = deep_chem_dataset.ids
    coloumn_names = ["smiles", "conformation", "dipole", "mbis-charges", "mbis-dipoles", "mbis-quadrupoles","inv_distance","esp"]
    #dictionary to store the results
    results = defaultdict(list)
    #dictionary to store info from the same conformers
    recordscache = defaultdict(list)    
    cached_smiles = None
    for index,(key, smiles) in tqdm(enumerate(zip(dataset_keys, dataset_smiles)),total=len(dataset_smiles)):
        #write in batches
            if index > 0:
                prev_smiles = dataset_smiles[index - 1]
                if smiles != prev_smiles:
                    # Finalize the cache for the previous smiles
                    if cached_smiles is not None:
                        results['smiles'].append(cached_smiles)
                        for column, values in recordscache.items():
                            # print(values)
                            results[column].append(values)
                        # total_records += 1
                    # Reset cache for the new smiles
                    recordscache = defaultdict(list)
            if (singlepoint_record := client.get_records(record_ids=key)):

                group_smiles = singlepoint_record.molecule.identifiers.canonical_isomeric_explicit_hydrogen_mapped_smiles
                assert group_smiles == smiles
                cached_smiles = group_smiles
                
                dipoles = singlepoint_record.properties['scf dipole']
                recordscache["dipole"].append(np.array(dipoles).flatten().tolist())
                
                charges = singlepoint_record.properties['mbis charges']
                recordscache["charges"].append(charges)
                
                conformation = singlepoint_record.molecule.geometry * unit.angstrom
                conformation_store = conformation.m_as(unit.bohr).flatten()
                # print(conformation)
                recordscache["conformation"].append(conformation_store.flatten().tolist())
                
                mbis_dipoles = singlepoint_record.properties['mbis dipoles']
                recordscache["mbis-dipoles"].append(np.array(mbis_dipoles).flatten().tolist())
                
                mbis_quadrupoles = singlepoint_record.properties['mbis quadrupoles']
                recordscache["mbis-quadrupoles"].append(np.array(mbis_quadrupoles).flatten().tolist())
                #build the grid and inv the distance between the grid coords and points
                
                openff_mol: Molecule = Molecule.from_mapped_smiles(group_smiles, allow_undefined_stereo=True)
                openff_mol.add_conformer(conformation)
                grid_coords = build_grid(molecule = openff_mol,
                        conformer= conformation,
                        grid_settings=grid_settings)
                
                #find the inv displacment between grid coords and esp
                grid_coordinates = grid_coords.reshape((-1, 3)).to(unit.bohr)
                atom_coordinates = conformation.reshape((-1, 3)).to(unit.bohr)
                displacement = grid_coordinates[:, None, :] - atom_coordinates[None, :, :]
                inv_displacement = 1/displacement
                recordscache["inv_distance"].append(inv_displacement.m.flatten().tolist())
                
                esp, _ = esp_calculator.assign_esp(
                    monopoles= np.array(charges),
                    dipoles=np.array(mbis_dipoles).reshape(-1,3),
                    quadropules=np.array(mbis_quadrupoles).reshape(-1,3,3),
                    grid = grid_coords,
                    coordinates= conformation
                )
                recordscache["esp"].append(esp)

    # for key, values in results.items():
    #     assert len(values) == total_records, print(key)
    # print(results)
    # results_df = pd.DataFrame(results)
    # results_df = pd.DataFrame.from_dict(results, orient='index')
    # results_df = results_df.transpose()
    # results_df.to_csv('results.csv')

    columns = [results[label] for label in coloumn_names]
    table = pyarrow.table(columns, coloumn_names)
    pyarrow.parquet.write_table(table, parquet_name)


for file_name, dataset_name in [
    ("training.parquet", "maxmin-train"),
    ("validation.parquet", "maxmin-valid"),
    ("testing.parquet", "maxmin-test"),
]:
    print("creating parquet for ", dataset_name)
    dc_dataset = dc.data.DiskDataset(dataset_name)
    create_parquet_dataset(
        parquet_name=file_name,
        deep_chem_dataset=dc_dataset,
        reference_dataset=data_set,
    )



