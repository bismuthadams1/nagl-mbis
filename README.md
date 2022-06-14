# NAGL-MBIS

A collection of models to predict conformation independent MBIS charges and volumes of molecules, built on the [NAGL](https://github.com/SimonBoothroyd/na)
package by SimonBoothroyd.

## Installation

The required dependencies to run these models can be installed using ``conda``:

```bash
conda install -c conda-forge -c dglteam nagl "gdl >=0.7" openff-toolkit pytorch-lightning
```

You will then need to install this package from source, first clone the repository from github:

```bash
git clone https://github.com/jthorton/nagl-mbis.git
cd nagl-mbis
```

With the nagl environment activate install the models via:

```bash
python setup.py install
```

## Quick start
NAGL-MBIS offers some pre-trained models to compute conformation independent MBIS charges and volumes, these can be loaded
using the following code in a script

```python
from naglmbis.models import load_volume_model, load_charge_model

# load two pre-trained models
charge_model = load_charge_model(charge_model=1)
volume_model = load_volume_model(volume_model=1)
```

we can then use these models to predict the corresponding properties for a given [openff-toolkit](https://github.com/openforcefield/openff-toolkit) [Molecule object](https://docs.openforcefield.org/projects/toolkit/en/stable/users/molecule_cookbook.html#cookbook-every-way-to-make-a-molecule).

```python
from openff.toolkit.topology import Molecule

# create ethanol
ethanol = Molecule.from_smiles("CCO")
# predict the charges (in e) and atomic volumes in (bohr ^3)
charges = charge_model.compute_properties()["mbis-charges"]
volumes = volume_model.compute_properties()["mbis-volumes"]
```

# Models

## MBISGraphModelV1

This model uses a minimal set of basic atomic features including

- one hot encoded element
- the number of bonds
- ring membership of size 3-6
- n_gcn_layers 5
- n_gcn_hidden_features 128
- n_mbis_layers 2
- n_mbis_hidden_features 64
- learning_rate 0.001
- n_epochs 100

These models were trained on the [OpenFF ESP Fragment Conformers v1.0](https://github.com/openforcefield/qca-dataset-submission/tree/master/submissions/2022-01-16-OpenFF-ESP-Fragment-Conformers-v1.0) dataset
which is on QCArchive. The dataset was computed using HF/6-31G* with PSI4.  


