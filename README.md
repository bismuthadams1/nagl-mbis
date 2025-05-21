# NAGL-MBIS
[![CI](https://github.com/jthorton/nagl-mbis/actions/workflows/CI.yaml/badge.svg)](https://github.com/jthorton/nagl-mbis/actions/workflows/CI.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/jthorton/nagl-mbis/branch/main/graph/badge.svg?token=LI1hLoCxZK)](https://codecov.io/gh/jthorton/nagl-mbis)

A collection of models to predict conformation independent MBIS atom-centred charges for molecules, built on the [NAGL](https://github.com/SimonBoothroyd/nagl)
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

With the nagl environment activated install the models via:

```bash
pip install -e . --no-build-isolation 
```

## Quick start
NAGL-MBIS offers a number of pre-trained models to compute conformation-independent MBIS charges, these can be loaded
using the following code in a script

```python
from naglmbis.models import load_charge_model

# load two pre-trained charge models
charge_model = load_charge_model(charge_model="nagl-gas-charge-wb")
# load a model trained to scf dipole and mbis charges
charge_model_2 = load_charge_model(charge_model="nagl-gas-charge-dipole-wb")
```

A list of the available models can be found in naglmbis/models/models.py

We can then use these models to predict the corresponding properties for a given [openff-toolkit](https://github.com/openforcefield/openff-toolkit) [Molecule object](https://docs.openforcefield.org/projects/toolkit/en/stable/users/molecule_cookbook.html#cookbook-every-way-to-make-a-molecule) or rdkit `Chem.Mol`.

```python
from openff.toolkit.topology import Molecule

# create ethanol
ethanol = Molecule.from_smiles("CCO")
# predict the charges (in e)
charges = charge_model.compute_properties(ethanol.to_rdkit())["mbis-charges"]
```

For computing partially polarised charges, we can use the class ComputePartialPolarised

```python
from openff.toolkit.topology import Molecule
from naglmbis.models.base_model import ComputePartialPolarised
from naglmbis.models import load_charge_model

gas_model = load_charge_model(charge_model="nagl-gas-charge-dipole-esp-wb-default")
water_model = load_charge_model(charge_model="nagl-water-charge-dipole-esp-wb-default")

polarised_model = ComputePartialPolarised(
   model_gas = gas_model,
   model_water = water_model,
   alpha = 0.5 #scaling parameter which can be adjusted
)

partial_charges = polarised_model.compute_polarised_charges(ethanol.to_rdkit())
print(partial_charges)
```

## Using the charges in a simulation

To use the charges in a simulation, we first create an Interchange object (following on from above):
```
from openff.toolkit import Quantity, unit

charges = polarised_model.compute_polarised_charges(ethanol.to_rdkit())

# Convert the charges to a 1D numpy array
charges = charges.detach().numpy().astype(float).squeeze()

# Assign the charges to the molecule and normalise them
ethanol.partial_charges = Quantity(
            charges,
            unit.elementary_charge,
        )
ethanol._normalize_partial_charges()
```
Now, create the interchange object. Note that the charge_from_molecules argument is critical, otherwise we'll end up with AM1-BCC charges. Also note that you will need to install [openff-interchange](https://github.com/openforcefield/openff-interchange) e.g. `mamba install -c conda-forge openff-interchange`.
```
from openff.toolkit import ForceField
from openff.interchange import Interchange

force_field = ForceField("openff-2.2.1.offxml")
interchange = Interchange.from_smirnoff(force_field=force_field, topology=[ethanol], charge_from_molecules=[ethanol])
print(ethanol.partial_charges)
```
You can then run a simulation with your engine of chioce, for example with OpenMM as shown [here](https://docs.openforcefield.org/en/latest/examples/openforcefield/openff-interchange/ligand_in_water/ligand_in_water.html).

# Models

## Summary of Models

## Available Models

The available charges are, for brievity Q = on-atom charges, $\mu$ = dipole, and V = electrostatic potential:

| Model                          | Objective          | Level of Theory of Training | Phase |
|--------------------------------|--------------------|--------------------------|-------------|
| nagl-v1-mbis                   | Q                  | HF/6-31G* - MBIS Charges |      gas
| nagl-v1-mbis-dipole            | Q, $\mu$           | HF/6-31G* - MBIS Charges | gas    |
| nagl-gas-charge-wb             | Q                  | $\omega$ B79X-d/def2-TZVPP - MBIS Charges| gas    |
| nagl-gas-charge-dipole-wb      | Q, $\mu$          | $\omega$ B79X-d/def2-TZVPP - MBIS Charges, QM Dipoles     | gas    |
| nagl-gas-charge-dipole-esp-wb-default   | Q, $\mu$, V | $\omega$~B79X-d/def2-TZVPP - MBIS Charges, QM Dipoles, ESP rebuilt to 1.4-2.0$\times$VdW with 0.5$\Angstrom$ spacing| naglmbis    |
| MBIS WB Water Charge           | on-atom charges    | MBIS_WB_WATER_CHARGE     | naglmbis    |
| MBIS WB Water Charge + Dipole  | on-atom charges    | MBIS_WB_WATER_CHARGE_DIPOLE| naglmbis  |
| MBIS WB Water Charge + Dipole + ESP | on-atom charges | MBIS_WB_WATER_CHARGE_DIPOLE_ESP| naglmbis |
| MBIS WB Gas ESP 2A             |on-atom charges  ESP | MBIS_WB_GAS_ESP_2A       | naglmbis    |
| MBIS WB Gas ESP 15A            | on-atom charges  | MBIS_WB_GAS_ESP_15A      | naglmbis    |
| MBIS WB Gas ESP Default        | on-atom charges  | MBIS_WB_GAS_ESP_DEFAULT  | naglmbis    |

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

These models were computed using  $\omega$B79X-d/def2-TZVPP with PSI4 and was split 80:10:10 using the deepchem maxmin spliter.   

## Training

The training scripts are located in the scripts subfolder in this repo. This is split into further subfolders.

1. **dataset** -  this subfolder contains all the scripts to pull down the QM data from qcarchive.
