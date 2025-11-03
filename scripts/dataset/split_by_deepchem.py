# try spliting the entire collection of data using deepchem spliters
import deepchem as dc
import numpy as np
from collections import defaultdict
from qcportal import PortalClient
from rdkit import Chem, DataStructs, SimDivFilters
import json

RANDOM_SEED = 42

maxminpicker = SimDivFilters.MaxMinPicker()

client = PortalClient("api.qcarchive.molssi.org")
print('getting dataset')
data_set = client.get_dataset(dataset_type='singlepoint',dataset_name='MLPepper RECAP Optimized Fragments v1.0')
print('getting finished records')
# finished_records = [record for record in data_set.iterate_records(status='complete')]

dataset_keys = []
smiles_ids = []
smiles_id_dict = defaultdict(list)

for _, _, singlepoint in data_set.iterate_records(status='complete'):
    if 'ddx' in singlepoint.specification.keywords:
        continue
    smiles = singlepoint.molecule.identifiers.canonical_isomeric_explicit_hydrogen_mapped_smiles
    print(smiles)
    smiles_id_dict[smiles].append(singlepoint.id)
    if smiles not in smiles_ids:
        smiles_ids.append(smiles)
    # smiles_ids.append(smiles)

    # use the key to quickly split the datasets later
    dataset_keys.append(singlepoint.id)

print('making fingeprints')

fingerprints = []
for smi in smiles_ids:
    mol = Chem.MolFromSmiles(smi)
    fp = SimDivFilters.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    arr = np.zeros((1,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    fingerprints.append(arr)

indices = np.arange(len(fingerprints))
#https://www.rdkit.org/docs/source/rdkit.SimDivFilters.rdSimDivPickers.html#rdkit.SimDivFilters.rdSimDivPickers.MaxMinPicker.LazyBitVectorPick
test_indices = maxminpicker.LazyBitVectorPick(
    fingerprints, 
    len(fingerprints), 
    int(0.2*len(fingerprints)), seed=RANDOM_SEED
)  
remaining_indices = list(set(indices) - set(test_indices))
validation_indices = maxminpicker.LazyBitVectorPick(
    [fingerprints[i] for i in remaining_indices],
    len(remaining_indices),
    int(0.1*len(fingerprints)),
    seed=RANDOM_SEED,
)
train_indices = list(set(remaining_indices) - set(validation_indices))

print(f"Total molecules: {len(smiles_ids)}")

test_dict = {smiles_ids[i]: smiles_id_dict[smiles_ids[i]] for i in test_indices}
validation_dict = {smiles_ids[i]: smiles_id_dict[smiles_ids[i]] for i in validation_indices}
train_dict = {smiles_ids[i]: smiles_id_dict[smiles_ids[i]] for i in train_indices}

# save the splits
with open('maxmin-test.json', 'w') as f:
    json.dump(test_dict, f, indent=4)
with open('maxmin-valid.json', 'w') as f:           
    json.dump(validation_dict, f, indent=4)
with open('maxmin-train.json', 'w') as f:           
    json.dump(train_dict, f, indent=4)




# print(f"The total number of unique molecules {len(smiles_ids)}")
# print("Running MaxMin Splitter ...")
# print('smiles id list')
# print(smiles_ids)
# print('smiles set')
# print(set(smiles_ids))
# xs = np.array(dataset_keys)
# print(xs)
# total_dataset = dc.data.DiskDataset.from_numpy(X=xs, ids=smiles_ids)

# max_min_split = dc.splits.MaxMinSplitter()
# train, validation, test = max_min_split.train_valid_test_split(
#     total_dataset,
#     train_dir="maxmin-train",
#     valid_dir="maxmin-valid",
#     test_dir="maxmin-test",
# )

# # train.to_csv('train_smiles.csv')
# # validation.to_csv('validation_smiles.csv')
# # test.to_csv('test_smiles.csv')
