
"""
Create a single pyarrow dataset from a collection of qcarchive datasets which can later be split into training and testing.

Notes:
    - conformers for molecules are grouped together first
    - try and batch calls to qcarchive for faster processing
    - we use a processpool to speed up calls to qca and builidng the grids.

Two datasets can currently be accessed:

               specification    complete
----------------------------  ----------
          wb97x-d/def2-tzvpp       68966
wb97x-d/def2-tzvpp/ddx-water       68966
"""
import pyarrow 
from openff.units import unit
from collections import defaultdict
import numpy as np
from openff.recharge.grids import GridGenerator, GridSettingsType, LatticeGridSettings
import pyarrow.parquet
from build_multipoles import ESPCalculator
from qcportal import PortalClient
from openff.toolkit.topology import Molecule
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from more_itertools import batched
import polars as pl
import os

pl.Config.set_verbose(True)  
os.environ["POLARS_VERBOSE"] = "1"


def build_grid(molecule: Molecule, conformer: unit.Quantity, grid_settings: GridSettingsType) -> unit.Quantity:
        grid = GridGenerator.generate(molecule, conformer, grid_settings)
        return grid

def process_record(record, molecule, grid_settings: GridSettingsType) -> dict:
    """Process the qca records into the data needed for the pyarrow table."""
    record_data = {
         "smiles": molecule.identifiers.canonical_isomeric_explicit_hydrogen_mapped_smiles,
         "dipole": record["properties"]['scf dipole'],
         "mbis-charges": record["properties"]["mbis charges"],
         # extract the conformation in bohr
         "conformation": molecule.geometry.flatten().tolist(),
         "mbis-dipoles": record["properties"]["mbis dipoles"],
         "mbis-quadrupoles": record["properties"]["mbis quadrupoles"],
         "record_id": record["id"]
    }
    #build the grid and inv the distance between the grid coords and points
    # build the molecule with its conformer attached
    openff_mol: Molecule = Molecule.from_qcschema(molecule, allow_undefined_stereo=True)
    grid_coords = build_grid(molecule = openff_mol,
            conformer=openff_mol.conformers[0],
            grid_settings=grid_settings)
    
    #find the inv displacment between grid coords and esp
    grid_coordinates = grid_coords.reshape((-1, 3)).to(unit.bohr)
    record_data["grid"] = grid_coordinates.m.flatten().tolist()
    atom_coordinates = openff_mol.conformers[0].reshape((-1, 3)).to(unit.bohr)
    displacement = grid_coordinates[:, None, :] - atom_coordinates[None, :, :]
    distance = np.linalg.norm(displacement.m, axis=-1) * unit.bohr
    inv_displacement = 1/distance
    record_data["inv_distance"] = inv_displacement.m.flatten().tolist()
    record_data["esp_length"] = len(inv_displacement)
    
    esp = ESPCalculator().assign_esp(
        monopoles=np.array(record_data["mbis-charges"]),
        dipoles=np.array(record_data["mbis-dipoles"]).reshape(-1,3),
        quadropules=np.array(record_data["mbis-quadrupoles"]).reshape(-1,3,3),
        grid=grid_coords,
        coordinates=openff_mol.conformers[0],
    )
    record_data["esp"] = esp.tolist()
    return record_data

def main(output: str):

    client = PortalClient("api.qcarchive.molssi.org", cache_dir=".")
    # allowing passing of datasets and features in a yaml file
    data_set = client.get_dataset(
        dataset_type='singlepoint',
        dataset_name='MLPepper RECAP Optimized Fragments v1.0'
    )

    # allow grid settings to be defined in a yaml file
    grid_settings =  LatticeGridSettings(
        type="fcc", spacing=0.5, inner_vdw_scale=1.4, outer_vdw_scale=2.0
    )
 
    rec_fn = partial(process_record, grid_settings=grid_settings)

    # set up the arrow table which we will write to
    schema = pyarrow.schema([
            pyarrow.field('smiles', pyarrow.string()),  
            pyarrow.field('dipole', pyarrow.list_(pyarrow.float64())), 
            pyarrow.field('conformation', pyarrow.list_(pyarrow.float64())),   
            pyarrow.field('mbis-charges', pyarrow.list_(pyarrow.float64())),  
            pyarrow.field('mbis-dipoles', pyarrow.list_(pyarrow.float64())),  
            pyarrow.field('mbis-quadrupoles', pyarrow.list_(pyarrow.float64())), 
            pyarrow.field('inv_distance', pyarrow.list_(pyarrow.float64())),  
            pyarrow.field('esp', pyarrow.list_(pyarrow.float64())),
            pyarrow.field("record_id", pyarrow.int32()),
            pyarrow.field('esp_length', pyarrow.int32()),
            pyarrow.field('grid', pyarrow.list_(pyarrow.float64())),
        ])
    entries = data_set.entry_names
    with pyarrow.parquet.ParquetWriter(where=output, schema=schema) as writer:
         
        with ProcessPoolExecutor(max_workers=16) as pool:
            # process in 1000 batch chunks
            for batch in batched(entries, 1000):
                jobs = [
                    pool.submit(rec_fn,
                        record.dict(),
                        record.molecule) for _, _, record in tqdm(
                        data_set.iterate_records(
                        specification_names=["wb97x-d/def2-tzvpp/ddx-water"],
                        status="complete",
                        entry_names=batch),
                        desc="Building Job list",
                        total=len(batch)) 
                ]
                for result in tqdm(as_completed(jobs), desc="Building local dataset ..."):
                    rec_data = result.result()
                    rec_batch = pyarrow.RecordBatch.from_pylist([rec_data,], schema=schema)
                    writer.write_batch(rec_batch)

if __name__ == "__main__":
    #an example output.
    main(output="mlpepper_water_grid_esp.parquet")


