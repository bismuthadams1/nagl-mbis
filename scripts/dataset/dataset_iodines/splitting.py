import deepchem as dc
import pyarrow.parquet
import pyarrow


def create_parquet(parquet_dataset: pyarrow.Table, deep_chem_dataset: dc.data.DiskDataset, dataset_name: str):

    # get the target smiles
    dataset_smiles = deep_chem_dataset.ids
    with pyarrow.parquet.ParquetWriter(dataset_name, schema=parquet_dataset.schema) as output:
        for data in parquet_dataset.to_pylist():
            if data["smiles"] in dataset_smiles not in data["smiles"]:
                batch = pyarrow.RecordBatch.from_pylist([data ], schema=parquet_dataset.schema)
                output.write_batch(batch)

if __name__ == "__main__":
    # load the ref dataset to split
    ref_dataset = pyarrow.parquet.read_table("/home/mlpepper/bismuthadams.mlpepper/repos/nagl-mbis-release-for-training/nagl-mbis/scripts/dataset_iodines/esp_set/default_grid_iodines_water.parquet")

    for deep_dataset, file_name in [
        ("maxmin-train", "training_water_charge_dipole.parquet"),
        ("maxmin-valid", "validation_water_charge_dipole.parquet"),
        ("maxmin-test", "testing_water_charge_dipole.parquet")
    ]:
        print(f"creating parquet for ", file_name)
        dc_dataset = dc.data.DiskDataset(deep_dataset)
        create_parquet(parquet_dataset=ref_dataset, deep_chem_dataset=dc_dataset, dataset_name=file_name)