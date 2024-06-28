# Test training script to make sure dipole prediction works
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from pytorch_lightning.loggers import MLFlowLogger

from nagl.config import Config, DataConfig, ModelConfig, OptimizerConfig
from nagl.config.data import Dataset, DipoleTarget, ReadoutTarget
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
import typing
import logging
import pathlib
import pydantic
from rdkit import Chem
import dataclasses

DEFAULT_RING_SIZES = [3, 4, 5, 6, 7, 8]


# define our ring membership feature
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
    return DataConfig(
        training=Dataset(
            sources=["../datasets/training.parquet"],
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
            ],
            batch_size=250,
        ),
        validation=Dataset(
            sources=["../datasets/validation.parquet"],
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
            ],
        ),
        test=Dataset(
            sources=["../datasets/testing.parquet"],
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
            ],
        ),
    )


def configure_optimizer(lr: float) -> OptimizerConfig:
    return OptimizerConfig(type="Adam", lr=lr)


def main():
    logging.basicConfig(level=logging.INFO)
    output_dir = pathlib.Path("001-train-charge-model-small-mols")

    register_atom_feature(AtomInRingOfSize)
    print(_CUSTOM_ATOM_FEATURES)
    # Configure our model, data sets, and optimizer.
    model_config = configure_model(
        atom_features=[
            AtomicElement(values=["H", "C", "N", "O", "F", "P", "S", "Cl", "Br"]),
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

    # Define the model and lightning data module that will contain the train, val,
    # and test dataloaders if specified in ``data_config``.
    config = Config(model=model_config, data=data_config, optimizer=optimizer_config)

    model = DGLMoleculeLightningModel(config)
    model.to_yaml("charge-dipole-v1.yaml")
    print("Model", model)

    # The 'cache_dir' will store the fully featurized molecules so we don't need to
    # re-compute these each to we adjust a hyperparameter for example.
    data = DGLMoleculeDataModule(config, cache_dir=output_dir / "feature-cache")

    # Define an MLFlow experiment to store the outputs of training this model. This
    # Will include the usual statistics as well as useful artifacts highlighting
    # the models weak spots.
    logger = MLFlowLogger(
        experiment_name="mbis-charge-dipole-model-small-mols-1000",
        save_dir=str(output_dir / "mlruns"),
        log_model="all",
    )

    # The MLFlow UI can be opened by running:
    #
    #    mlflow ui --backend-store-uri     ./001-train-charge-model/mlruns \
    #              --default-artifact-root ./001-train-charge-model/mlruns
    #

    # Train the model
    n_epochs = 1000

    n_gpus = 0 if not torch.cuda.is_available() else 1
    print(f"Using {n_gpus} GPUs")

    model_checkpoint = ModelCheckpoint(
        monitor="val/loss", dirpath=output_dir.joinpath("")
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        # devices=n_gpus,
        min_epochs=n_epochs,
        max_epochs=n_epochs,
        logger=logger,
        log_every_n_steps=50,
        callbacks=[model_checkpoint],
    )

    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data)

    print(model_checkpoint.best_model_path)


if __name__ == "__main__":
    main()
