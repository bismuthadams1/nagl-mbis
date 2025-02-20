This folder contains all the scripts pertaining to the training process.
The should be used in the following order:

1. **split_by_deepchem.py**: split the remote qcarchive in to test:train:validation. A set of three folders prefixed with maxmin- will be created, these will be used later to split the local dataset and containing metadata about which molecules should exist in each split.

2. **setup_labaled_data.py**: this builds a parquet based on the remote qcarchive data. Additionally, with the help of the class in `build_multipoles.py`, it rebuilds the ESP from the MBIS multipoles.

3. **spltting.py**: this takes the locally built parquet and splits it into three parquets: training.parquet, test.parquet, and validation.parquet. These can now be used in training. 