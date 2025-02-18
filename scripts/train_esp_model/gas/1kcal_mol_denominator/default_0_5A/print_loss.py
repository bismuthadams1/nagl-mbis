# Test training script to make sure dipole prediction works
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from pytorch_lightning.loggers import MLFlowLogger
import numpy

from nagl.config import Config, DataConfig, ModelConfig, OptimizerConfig
from nagl.config.data import Dataset, DipoleTarget, ReadoutTarget, ESPTarget
from nagl.config.model import GCNConvolutionModule, ReadoutModule, Sequential
from nagl.features import (
    AtomConnectivity,
    AtomFeature,
    AtomicElement,
    BondFeature,
    AtomFeature,
    register_atom_feature,
    _CUSTOM_ATOM_FEATURES,
)
from nagl.training import DGLMoleculeDataModule, DGLMoleculeLightningModel
# from naglmbis.models.models import load_charge_model
import typing
import logging
import pathlib
import pydantic
from rdkit import Chem
import dataclasses

DEFAULT_RING_SIZES = [3, 4, 5, 6, 7, 8]

def configure_model(
    atom_features: typing.List[AtomFeature],
    bond_features: typing.List[BondFeature],
    n_gcn_layers: int,
    n_gcn_hidden_features: int,
    n_am1_layers: int,
    n_am1_hidden_features: int,
) -> ModelConfig:
    return ModelConfig(
        atom_features=atom_features,
        bond_features=bond_features,
        convolution=GCNConvolutionModule(
            type="SAGEConv",
            hidden_feats=[n_gcn_hidden_features] * n_gcn_layers,
            activation=["ReLU"] * n_gcn_layers,
        ),
        readouts={
            "mbis-charges": ReadoutModule(
                pooling="atom",
                forward=Sequential(
                    hidden_feats=[n_am1_hidden_features] * n_am1_layers + [2],
                    activation=["ReLU"] * n_am1_layers + ["Identity"],
                ),
                postprocess="charges",
            )
        },
    )

def configure_data() -> DataConfig:
    from openff.units import unit
    KE = 1

    return DataConfig(
        training=Dataset(
            sources=["/projects/public/mlpepper/nagl_datasets/default_esp_grids/gas/training_esp.parquet"],
            # The 'column' must match one of the label columns in the parquet
            # table that was create during stage 000.
            # The 'readout' column should correspond to one our or model readout
            # keys.
            # denom for charge in e and dipole in e*bohr 0.1D~
            targets=[
                ReadoutTarget(
                    column="mbis-charges",
                    readout="mbis-charges",
                    metric="rmse",
                    denominator=0.02,
                ),
                DipoleTarget(
                    metric="rmse",
                    dipole_column="dipole",
                    conformation_column="conformation",
                    charge_label="mbis-charges",
                    denominator=0.04,
                ),
                ESPTarget(
                    esp_column="esp",
                    charge_label="mbis-charges",
                    inv_distance_column="inv_distance",
                    metric="rmse",
                    esp_length_column="esp_length",
                    ke = KE,
                    denominator = 0.00159362,  #1hartree =  627.503 kcal/mol
                ),
            ],
            batch_size=250,
        ),
        validation=Dataset(
            sources=["/projects/public/mlpepper/nagl_datasets/default_esp_grids/gas/validation_esp.parquet"],
            targets=[
                ReadoutTarget(
                    column="mbis-charges",
                    readout="mbis-charges",
                    metric="rmse",
                    denominator=0.02,
                ),
                DipoleTarget(
                    metric="rmse",
                    dipole_column="dipole",
                    conformation_column="conformation",
                    charge_label="mbis-charges",
                    denominator=0.04,
                ),
                ESPTarget(
                    esp_column="esp",
                    charge_label="mbis-charges",
                    inv_distance_column="inv_distance",
                    metric="rmse",
                    esp_length_column="esp_length",
                    ke = KE,
                    denominator = 0.00159362,  #1hartree =  627.503 kcal/mol
                ),
            ],
        ),
        test=Dataset(
            sources=["/projects/public/mlpepper/nagl_datasets/default_esp_grids/gas/testing_esp.parquet"],
            targets=[
                ReadoutTarget(
                    column="mbis-charges",
                    readout="mbis-charges",
                    metric="rmse",
                    denominator=0.02,
                ),
                DipoleTarget(
                    metric="rmse",
                    dipole_column="dipole",
                    conformation_column="conformation",
                    charge_label="mbis-charges",
                    denominator=0.04,
                ),
                ESPTarget(
                    esp_column="esp",
                    charge_label="mbis-charges",
                    inv_distance_column="inv_distance",
                    metric="rmse",
                    esp_length_column="esp_length",
                    ke = KE,
                    denominator = 0.00159362,  #1hartree =  627.503 kcal/mol
                ),
            ],
        ),
    )


def configure_optimizer(lr: float) -> OptimizerConfig:
    return OptimizerConfig(type="Adam", lr=lr)




def main():
    logging.basicConfig(level=logging.INFO)
    output_dir = pathlib.Path("001-train-charge-model-small-mols")

    # Define your custom feature
    @pydantic.dataclasses.dataclass(config={"extra": pydantic.Extra.forbid})
    class AtomInRingOfSize(AtomFeature):
        type: typing.Literal["ringofsize"] = "ringofsize"
        ring_sizes: typing.List[pydantic.PositiveInt] = pydantic.Field(
            DEFAULT_RING_SIZES,
            description="The size of the ring we want to check membership of",
        )

        def __len__(self):
            return len(self.ring_sizes)

        def __call__(self, molecule: Chem.Mol) -> torch.Tensor:
            ring_info: Chem.RingInfo = molecule.GetRingInfo()

            return torch.vstack(
                [
                    torch.Tensor(
                        [
                            int(ring_info.IsAtomInRingOfSize(atom.GetIdx(), ring_size))
                            for ring_size in self.ring_sizes
                        ]
                    )
                    for atom in molecule.GetAtoms()
                ]
            )

    # Register the custom feature
    register_atom_feature(AtomInRingOfSize)

    # Configure your model
    model_config = configure_model(
        atom_features=[
            AtomicElement(values=["H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br"]),
            AtomConnectivity(),
            dataclasses.asdict(AtomInRingOfSize()),
        ],
        bond_features=[],
        n_gcn_layers=5,
        n_gcn_hidden_features=128,
        n_am1_layers=2,
        n_am1_hidden_features=64,
    )

    data_config = configure_data()

    optimizer_config = configure_optimizer(0.001)

    # Configure data and optimizer

    config = Config(model=model_config, data=data_config, optimizer=optimizer_config)

    # Initialize the model
    model = DGLMoleculeLightningModel(config)

    # Load the checkpoint
    checkpoint_path = "/home/mlpepper/bismuthadams.mlpepper/repos/nagl-mbis-release-for-training/nagl-mbis/scripts/train_esp_model/gas/1kcal_mol_denominator/default_0_5A/001-train-charge-model-small-mols/epoch=764-step=169065.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])

    # Proceed with setting up your data module
    data = DGLMoleculeDataModule(config, cache_dir=output_dir / "feature-cache")

    # Create the trainer
    trainer = pl.Trainer(accelerator="cpu")

    # Run the test set and collect results
    test_results = trainer.test(model, datamodule=data)

    # Print the test loss on all loss functions
    print("Test Results:")
    for key, value in test_results[0].items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()