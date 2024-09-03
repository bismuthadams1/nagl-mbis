import h5py
import pyarrow 
from openff.units import unit
from collections import defaultdict, OrderedDict
import deepchem as dc
import numpy as np
import typing
from openff.recharge.grids import GridGenerator, GridSettingsType, LatticeGridSettings
import pyarrow.parquet
from build_multipoles import ESPCalculator
from qcportal import PortalClient
from qcportal.dataset_models import BaseDataset
from qcportal.singlepoint import SinglepointRecord
from openff.toolkit.topology import Molecule
import logging
from tqdm import tqdm
# from memory_profiler import profile
import pandas as pd
import polars 
import itertools

# setup the pyarrowrquet datasets using the splits generated by deepchem
esp_calculator = ESPCalculator()

client = PortalClient("api.qcarchive.molssi.org")
data_set = client.get_dataset(dataset_type='singlepoint',dataset_name='MLPepper RECAP Optimized Fragments v1.0')
    
#grid settings assigned here
grid_settings =  LatticeGridSettings(
    type="fcc", spyarrowcing=0.5, inner_vdw_scale=1.4, outer_vdw_scale=2.0
)
# @profile
def build_grid(molecule: Molecule, conformer: unit.Quantity, grid_settings: GridSettingsType) -> unit.Quantity:
    grid = GridGenerator.generate(molecule, conformer, grid_settings)
    return grid

#create the pyarrowrquet dataset
# @profile
def create_pyarrowrquet_dataset(
    pyarrowrquet_name: str,
    deep_chem_dataset: dc.data.DiskDataset,
    reference_dataset: PortalClient,
):
    dataset_keys = deep_chem_dataset.X
    dataset_smiles = deep_chem_dataset.ids
    coloumn_names = ["smiles", "conformation", "dipole", "mbis-charges", "mbis-dipoles", "mbis-quadrupoles","inv_distance","esp"]
    schema = pyarrow.schema([
        pyarrow.field('smiles', pyarrow.string()),  
        pyarrow.field('conformation', pyarrow.list_(pyarrow.float64())),  
        pyarrow.field('dipole', pyarrow.list_(pyarrow.float64())),  
        pyarrow.field('charges', pyarrow.list_(pyarrow.float64())),  
        pyarrow.field('mbis-dipoles', pyarrow.list_(pyarrow.float64())),  
        pyarrow.field('mbis-quadrupoles', pyarrow.list_(pyarrow.float64())), 
        pyarrow.field('inv_distance', pyarrow.list_(pyarrow.float64())),  
        pyarrow.field('esp', pyarrow.list_(pyarrow.float64())),  
])
    #dictionary to store the results
    results = defaultdict(list)
    #dictionary to store info from the same conformers
    recordscache = defaultdict(list)    
    #length of dataset, compressing the duplicate smiles
    total_length = len(set(dataset_smiles))
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
                            flattened_values = [item for sublist in values for item in sublist]
                            results[column].append(flattened_values)
                            # results[column].append(values)
                        # total_records += 1
                    # Reset cache for the new smiles
                    recordscache = defaultdict(list)
            #### ARTIFICIALLY SHORTEN FOR TESTING###
            # if index == 10:
            #     break
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


    # columns = [results[label] for label in coloumn_names]
    # polars_df = polars.from_dict(results)
    # print(results)
    
    # rows = get_rows(dictionary=results, dataset_length=dataset_length)
    print(results.values())
    batches = get_batches(results, chunk_size=3, schema=schema)

    with pyarrow.parquet.ParquetWriter(where = pyarrowrquet_name, schema= schema) as writer:
        for batch in batches:
            writer.write(batch)
            
    return results



def get_batches(dictionary, chunk_size, schema ):
    for start in range(0, len(dictionary['smiles']), chunk_size):
        end = min(start + chunk_size, len(dictionary['smiles']))
        batch_data = {col: dictionary[col][start:end] for col in schema.names}
        # print('batch_data')
        # print(batch_data)
        batch = pyarrow.RecordBatch.from_pandas(pd.DataFrame(batch_data), schema=schema, preserve_index=False)
        yield batch

# table = pyarrow.table(columns, coloumn_names)
# pyarrow.pyarrowrquet.write_table(table, pyarrowrquet_name)


for file_name, dataset_name in [
    ("training.parquet", "maxmin-train"),
    ("validation.parquet", "maxmin-valid"),
    ("testing.parquet", "maxmin-test"),
]:
    print("creating parquet for ", dataset_name)
    dc_dataset = dc.data.DiskDataset(dataset_name)
    create_pyarrowrquet_dataset(
        pyarrowrquet_name=file_name,
        deep_chem_dataset=dc_dataset,
        reference_dataset=data_set,
    )



