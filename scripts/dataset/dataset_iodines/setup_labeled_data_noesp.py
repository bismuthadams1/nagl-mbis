
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


def process_record(record, molecule) -> dict:
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
    return record_data

def main(output: str):

    client = PortalClient("api.qcarchive.molssi.org", cache_dir=".")
    # allowing passing of datasets and features in a yaml file
    data_set = client.get_dataset(
        dataset_type='singlepoint',
        dataset_name='MLPepper RECAP Optimized Fragments v1.0'
    )

    iodine_data_set = client.get_dataset(
        dataset_type='singlepoint',
        dataset_name='MLPepper RECAP Optimized Fragments v1.0 Add Iodines'
    )
    rec_fn = partial(process_record)

    # set up the arrow table which we will write to
    schema = pyarrow.schema([
            pyarrow.field('smiles', pyarrow.string()),  
            pyarrow.field('dipole', pyarrow.list_(pyarrow.float64())), 
            pyarrow.field('conformation', pyarrow.list_(pyarrow.float64())),   
            pyarrow.field('mbis-charges', pyarrow.list_(pyarrow.float64())),  
            pyarrow.field('mbis-dipoles', pyarrow.list_(pyarrow.float64())),  
            pyarrow.field('mbis-quadrupoles', pyarrow.list_(pyarrow.float64())), 
            pyarrow.field("record_id", pyarrow.int32()),
        ])
    entries = data_set.entry_names
    entries_iodine = iodine_data_set.entry_names
    with pyarrow.parquet.ParquetWriter(where=output, schema=schema) as writer:
         
        with ProcessPoolExecutor(max_workers=16) as pool:
            # process in 1000 batch chunks
            for batch in batched(entries, 1000):
                jobs = [
                    pool.submit(rec_fn, record.dict(), record.molecule) for _, _, record in tqdm(data_set.iterate_records(specification_names=["wb97x-d/def2-tzvpp/ddx-water"], status="complete", entry_names=batch), desc="Building Job list", total=len(batch)) 
                ]
                for result in tqdm(as_completed(jobs), desc="Building local dataset ..."):
                    rec_data = result.result()
                    rec_batch = pyarrow.RecordBatch.from_pylist([rec_data,], schema=schema)
                    writer.write_batch(rec_batch)

        with ProcessPoolExecutor(max_workers=16) as pool:
                # process in 1000 batch chunks
                for batch in batched(entries_iodine, 1000):
                    jobs = [
                        pool.submit(rec_fn, record.dict(), record.molecule) for _, _, record in tqdm(iodine_data_set.iterate_records(specification_names=["wb97x-d/def2-tzvpp/ddx-water"], status="complete", entry_names=batch), desc="Building Job list", total=len(batch)) 
                    ]
                    for result in tqdm(as_completed(jobs), desc="Building local dataset ..."):
                        rec_data = result.result()
                        rec_batch = pyarrow.RecordBatch.from_pylist([rec_data,], schema=schema)
                        writer.write_batch(rec_batch)

if __name__ == "__main__":
    main(output="/tank/home/charlie2/nagl-mbis-training/scripts/dataset_iodines/charge_dipole_dbs/charge_dipole_set_iodines.parquet")


