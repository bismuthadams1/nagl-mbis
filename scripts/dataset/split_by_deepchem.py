# try spliting the entire collection of data using deepchem spliters
import deepchem as dc
import numpy as np
from qcportal import PortalClient

client = PortalClient("api.qcarchive.molssi.org")
print('getting dataset')
data_set = client.get_dataset(dataset_type='singlepoint',dataset_name='MLPepper RECAP Optimized Fragments v1.0')
print('getting finished records')
# finished_records = [record for record in data_set.iterate_records(status='complete')]

dataset_keys = []
smiles_ids = []

for _, _, singlepoint in data_set.iterate_records(status='complete'):
    if 'ddx' in singlepoint.specification.keywords:
        continue
    smiles = singlepoint.molecule.identifiers.canonical_isomeric_explicit_hydrogen_mapped_smiles
    print(smiles)
    smiles_ids.append(smiles)
    # use the key to quickly split the datasets later
    dataset_keys.append(singlepoint.id)

print(f"The total number of unique molecules {len(smiles_ids)}")
print("Running MaxMin Splitter ...")
print('smiles id list')
print(smiles_ids)
print('smiles set')
print(set(smiles_ids))
xs = np.array(dataset_keys)
print(xs)
total_dataset = dc.data.DiskDataset.from_numpy(X=xs, ids=smiles_ids)

max_min_split = dc.splits.MaxMinSplitter()
train, validation, test = max_min_split.train_valid_test_split(
    total_dataset,
    train_dir="maxmin-train",
    valid_dir="maxmin-valid",
    test_dir="maxmin-test",
)

# train.to_csv('train_smiles.csv')
# validation.to_csv('validation_smiles.csv')
# test.to_csv('test_smiles.csv')
