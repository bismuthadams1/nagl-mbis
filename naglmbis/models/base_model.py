# models for the nagl run

import abc
from typing import Dict, List, Literal

import torch
from nagl.lightning import DGLMoleculeLightningModel
from nagl.molecules import DGLMolecule
from nagl.nn import SequentialLayers
from nagl.nn.modules import ConvolutionModule, ReadoutModule
from nagl.nn.pooling import PoolAtomFeatures
from nagl.nn.postprocess import ComputePartialCharges
from openff.toolkit.topology import Molecule


class MBISGraphModel(DGLMoleculeLightningModel):
    "A wrapper to make it easy to load and evaluate models"

    @abc.abstractmethod
    def features(self):
        ...

    def __init__(
        self,
        n_gcn_hidden_features: int,
        n_gcn_layers: int,
        n_mbis_hidden_features: int,
        n_mbis_layers: int,
        readout_modules: List[Literal["charge", "volume"]],
        learning_rate: float,
    ):
        self.n_gcn_hidden_features = n_gcn_hidden_features
        self.n_gcn_layers = n_gcn_layers
        self.n_mbis_hidden_features = n_mbis_hidden_features
        self.n_mbis_layers = n_mbis_layers
        n_atom_features = sum(len(feature) for feature in self.features()[0])
        readout = {}
        if "charge" in readout_modules:
            readout["mbis-charges"] = ReadoutModule(
                pooling_layer=PoolAtomFeatures(),
                readout_layers=SequentialLayers(
                    in_feats=n_gcn_hidden_features,
                    hidden_feats=[n_mbis_hidden_features] * n_mbis_layers + [2],
                    activation=["ReLU"] * n_mbis_layers + ["Identity"],
                ),
                postprocess_layer=ComputePartialCharges(),
            )
        if "volume" in readout_modules:
            readout["mbis-volumes"] = ReadoutModule(
                pooling_layer=PoolAtomFeatures(),
                readout_layers=SequentialLayers(
                    in_feats=n_gcn_hidden_features,
                    hidden_feats=[n_mbis_hidden_features] * n_mbis_layers + [1],
                    activation=["ReLU"] * n_mbis_layers + ["Identity"],
                ),
            )

        super().__init__(
            convolution_module=ConvolutionModule(
                architecture="SAGEConv",
                in_feats=n_atom_features,
                hidden_feats=[n_gcn_hidden_features] * n_gcn_layers,
            ),
            readout_modules=readout,
            learning_rate=learning_rate,
        )

        self.save_hyperparameters()

    def compute_properties(self, molecule: Molecule) -> Dict[str, torch.Tensor]:
        atom_features, bond_features = self.features()
        dgl_molecule = DGLMolecule.from_openff(molecule, atom_features, bond_features)

        return self.forward(dgl_molecule)
