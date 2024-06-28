# try spliting the entire collection of data using deepchem spliters
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

total_dataset = dc.data.DiskDataset.from_numpy(X=xs, ids=smiles_ids)

max_min_split = dc.splits.MaxMinSplitter()
train, validation, test = max_min_split.train_valid_test_split(
    total_dataset,
    train_dir="maxmin-train",
    valid_dir="maxmin-valid",
    test_dir="maxmin-test",
)
