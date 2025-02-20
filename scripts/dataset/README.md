This folder contains all the scripts pertaining to the training process.
The should be used in the following order:

1. **split_by_deepchem.py**: split the remote qcarchive in to test:train:validation. A set of three folders prefixed with maxmin- will be created, these will be used later to split the local dataset and containing metadata about which molecules should exist in each split.

2. **setup_labaled_data.py**: this builds a parquet based on the remote qcarchive data. Additionally, with the help of the class in `build_multipoles.py`, it rebuilds the ESP from the MBIS multipoles.

3. **spltting.py**: this takes the locally built parquet and splits it into three parquets: training.parquet, test.parquet, and validation.parquet. These can now be used in training. 

You can use the `analysis_train_val_test_split.py` to analyse the splits in terms of number of elements and heavy atoms. 

The main folder contains all the scripts to create the main dataset - [MLPepper RECAP Optimized Fragments v1.0
](https://github.com/openforcefield/qca-dataset-submission/tree/master/submissions/2024-07-26-MLPepper-RECAP-Optimized-Fragments-v1.0). The `add_iodines/` subfolder contains the scripts which added the iodine containing compounds ([MLPepper-RECAP-Optimized-Fragments-Add-Iodines-v1.0
](https://github.com/openforcefield/qca-dataset-submission/tree/master/submissions/2024-10-11-MLPepper-RECAP-Optimized-Fragments-Add-Iodines-v1.0)). 