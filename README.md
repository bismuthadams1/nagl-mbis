# NAGL-MBIS
[![CI](https://github.com/jthorton/nagl-mbis/actions/workflows/CI.yaml/badge.svg)](https://github.com/jthorton/nagl-mbis/actions/workflows/CI.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/jthorton/nagl-mbis/branch/main/graph/badge.svg?token=LI1hLoCxZK)](https://codecov.io/gh/jthorton/nagl-mbis)

A collection of models to predict conformation independent MBIS charges and volumes of molecules, built on the [NAGL](https://github.com/SimonBoothroyd/nagl)
package by SimonBoothroyd.

## Installation

The required dependencies to run these models can be installed using ``mamba`` and the provided environment file:

```bash
mamba env create -f devtools/conda-envs/env.yaml
```

You will then need to install this package from source, first clone the repository from github:

```bash
git clone https://github.com/bismuthadams1/nagl-mbis.git
cd nagl-mbis
```

With the nagl environment activate install the models via:

```bash
pip install -e . --no-build-isolation 
```

## Quick start
NAGL-MBIS offers a large number of pre-trained models to compute conformation independent MBIS charges, these can be loaded
using the following code in a script

```python
from naglmbis.models import load_charge_model

# load two pre-trained charge models
charge_model = load_charge_model(charge_model="nagl-gas-charge-wb")
# load a model trained to scf dipole and mbis charges
charge_model_2 = load_charge_model(charge_model="nagl-gas-charge-dipole-wb")
```

A list of the available models can be found in naglmbis/modls/models.py

we can then use these models to predict the corresponding properties for a given [openff-toolkit](https://github.com/openforcefield/openff-toolkit) [Molecule object](https://docs.openforcefield.org/projects/toolkit/en/stable/users/molecule_cookbook.html#cookbook-every-way-to-make-a-molecule) or rdkit `Chem.Mol`.

```python
from openff.toolkit.topology import Molecule

# create ethanol
ethanol = Molecule.from_smiles("CCO")
# predict the charges (in e) and atomic volumes in (bohr ^3)
charges = charge_model.compute_properties(ethanol.to_rdkit())["mbis-charges"]
volumes = charge_model_2.compute_properties(ethanol.to_rdkit())["mbis-volumes"]
```

For computing partially polarised charges, we can use the class ComputePartialPolarised

```python
from openff.toolkit.topology import Molecule
from naglbmis.models import ComputePartialPolarised
from naglmbis.models import load_charge_model

gas_model = load_charge_model(charge_model="nagl-gas-charge-dipole-wb")
water_model = load_charge_model(charge_model="nagl-gas-charge-dipole-wb")

polarised_model = ComputePartialPolarised(
   model_gas = gas_model,
   model_water = water_model,
   alpha = 0.5 #scaling parameter which can be adjusted
)

polarised_model.compute_properties(ethanol.to_rdkit())
```


# This is currently broken, due to plugins changing in the openff stack!
Alternatively we provide an openff-toolkit parameter handler plugin which allows you to create an openmm system
using the normal python pathway with a modified force field which requests that the ``NAGMBIS`` model be used to 
predict charges and LJ parameters. We provide a function which can modify any offxml to add the custom handler

```python
from naglmbis.plugins import modify_force_field
from openff.toolkit.topology import Molecule

nagl_sage = modify_force_field(force_field="openff_unconstrained-2.0.0.offxml")
# write out the force field to file
nagl_sage.to_file("nagl_sage.offxml")
# or use it to create an openmm system
methanol = Molecule.from_smiles("CO")
openmm_system = nagl_sage.create_openmm_system(topology=methanol.to_topology())
```

# Models

## MBISGraphMode

This model uses a minimal set of basic atomic features including

- one hot encoded element
- the number of bonds
- ring membership of size 3-8
- n_gcn_layers 5
- n_gcn_hidden_features 128
- n_mbis_layers 2
- n_mbis_hidden_features 64
- learning_rate 0.001
- n_epochs 1000

The models in this repo were trained from two QM datasets. 

1. The models starting with `nagl-v1`:

These models were trained on the [OpenFF ESP Fragment Conformers v1.0](https://github.com/openforcefield/qca-dataset-submission/tree/master/submissions/2022-01-16-OpenFF-ESP-Fragment-Conformers-v1.0) dataset
which is on QCArchive. 

These models were computed using HF/6-31G* with PSI4 and was split 80:10:10 using the deepchem maxmin spliter.  

2. The rest of the models:

These models were trained on the [MLPepper RECAP Optimized Fragments v1.0
](https://github.com/openforcefield/qca-dataset-submission/tree/master/submissions/2024-07-26-MLPepper-RECAP-Optimized-Fragments-v1.0) and [MLPepper-RECAP-Optimized-Fragments-Add-Iodines-v1.0
](https://github.com/openforcefield/qca-dataset-submission/tree/master/submissions/2024-10-11-MLPepper-RECAP-Optimized-Fragments-Add-Iodines-v1.0) datasets.

These models were computed using wB79X-d/def2-TZPP with PSI4 and was split 80:10:10 using the deepchem maxmin spliter.   

## Training

The training scripts are localed in the scripts subfolder in this repo. This is split into further subfolders.

1. **dataset** -  this subfolder contains all the scripts to pull down the QM data from qcarchive, 