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
from tqdm import tqdm
# from memory_profiler import profile
import pandas as pd

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

    recordscache = defaultdict(list)    
    #length of dataset, compressing the duplicate smiles
    cached_smiles = None
    #WARNING PATCHED THE START TO LAST FAILURE
    for index,(key, smiles) in tqdm(enumerate(zip(dataset_keys, dataset_smiles), start= 0),total=len(dataset_smiles)):
        #write in batches
            if index > 0:
                #this is to test to see if we have more than 1 conformer
                prev_smiles = dataset_smiles[index - 1]
                if smiles != prev_smiles:
                    # Finalize the cache for the previous smiles
                    if cached_smiles is not None:
                        flat_dict = defaultdict(list)

                        for column, values in recordscache.items():
                            flattened_values = [item for sublist in values for item in sublist]
                            flat_dict[column].append(flattened_values)
                        flat_dict['smiles'].append(str(cached_smiles))
                        batch = get_batch(dictionary = flat_dict, schema= schema)
                        try:
                            table_original_file = pyarrow.parquet.read_table(source=pyarrowrquet_name,  pre_buffer=False, use_threads=True, memory_map=True)  # Use memory map for speed.
                            table_to_append = batch.cast(table_original_file.schema)  # Attempt to cast new schema to existing, e.g. datetime64[ns] to datetime64[us] (may throw otherwise).
                        except Exception as e:
                            print('need to create parquet first')
                            table_original_file = None
                      
                        with pyarrow.parquet.ParquetWriter(where = pyarrowrquet_name, schema= schema) as writer:
                            if table_original_file:
                                writer.write(table_original_file)
                                writer.write(table_to_append)
                            else:
                                writer.write(batch)
                
                    recordscache = defaultdict(list)
            #### ARTIFICIALLY SHORTEN FOR TESTING###
            # if index == 10:
            #     break
            if (singlepoint_record := reference_dataset.get_records(record_ids=key)):

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
                distance = np.linalg.norm(displacement.m, axis=-1) * unit.bohr
                inv_displacement = 1/distance
                recordscache["inv_distance"].append(inv_displacement.m.flatten().tolist())
                
                esp, _ = esp_calculator.assign_esp(
                    monopoles= np.array(charges),
                    dipoles=np.array(mbis_dipoles).reshape(-1,3),
                    quadropules=np.array(mbis_quadrupoles).reshape(-1,3,3),
                    grid = grid_coords,
                    coordinates= conformation
                )
                recordscache["esp"].append(esp)

def get_batch(dictionary, schema):
    batch = pyarrow.RecordBatch.from_pandas(pd.DataFrame(dictionary), schema=schema, preserve_index=False)
    return batch

for file_name, dataset_name in [  #("training.parquet", "maxmin-train")
    ("training.parquet", "maxmin-train"),
    ("validation.parquet", "maxmin-valid"),
    ("testing.parquet", "maxmin-test"),
]:
    print("creating parquet for ", dataset_name)
    dc_dataset = dc.data.DiskDataset(dataset_name)
    create_pyarrowrquet_dataset(
        pyarrowrquet_name=file_name,
        deep_chem_dataset=dc_dataset,
        reference_dataset=client,
    )



