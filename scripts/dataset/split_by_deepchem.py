# try spliting the entire collection of data using deepchem spliters
<<<<<<< HEAD
import h5py
import deepchem as dc
import numpy as np

dataset_keys = []
smiles_ids = []
training_set = h5py.File("TrainingSet-v1.hdf5", "r")
for key, group in training_set.items():
    smiles_ids.append(group["smiles"].asstr()[0])
    # use the key to quickly split the datasets later
    dataset_keys.append(key)
training_set.close()

# val_set = h5py.File('ValSet-v1.hdf5', 'r')
# for key, group in val_set.items():
#     smiles_ids.append(group['smiles'].asstr()[0])
#     dataset_keys.append(key)

# val_set.close()


print(f"The total number of unique molecules {len(smiles_ids)}")
print("Running MaxMin Splitter ...")

xs = np.array(dataset_keys)

=======
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
>>>>>>> 2469f10361fc24ac1bf64a5da1aa84db5e853498
total_dataset = dc.data.DiskDataset.from_numpy(X=xs, ids=smiles_ids)

max_min_split = dc.splits.MaxMinSplitter()
train, validation, test = max_min_split.train_valid_test_split(
    total_dataset,
    train_dir="maxmin-train",
    valid_dir="maxmin-valid",
    test_dir="maxmin-test",
)
<<<<<<< HEAD
=======

# train.to_csv('train_smiles.csv')
# validation.to_csv('validation_smiles.csv')
# test.to_csv('test_smiles.csv')
>>>>>>> 2469f10361fc24ac1bf64a5da1aa84db5e853498
