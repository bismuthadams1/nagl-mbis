from typing import Literal

import torch
from nagl.features import AtomFeature, one_hot_encode, register_atom_feature
from pydantic import Extra, Field, dataclasses
from rdkit import Chem


@dataclasses.dataclass(config={"extra": Extra.forbid})
class HydrogenAtoms(AtomFeature):
    """One hot encode the number of bonded hydrogen atoms"""

    type: Literal["hydrogenatoms"] = "hydrogenatoms"
    hydrogens: list[int] = Field(
        [0, 1, 2, 3, 4],
        description="The options for the number of bonded hydrogens to one hot encode.",
    )

    def __call__(self, molecule: Chem.Mol) -> torch.Tensor:
        return torch.vstack(
            [
                one_hot_encode(
                    atom.GetTotalNumHs(includeNeighbors=True),
                    self.hydrogens,
                )
                for atom in molecule.GetAtoms()
            ]
        )

    def __len__(self):
        return len(self.hydrogens)


@dataclasses.dataclass(config={"extra": Extra.forbid})
class AtomInRingOfSize(AtomFeature):
    type: Literal["ringofsize"] = "ringofsize"
    ring_sizes: list[int] = Field(
        [3, 4, 5, 6, 7, 8],
        description="The ring of size we want to check membership of.",
    )

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

    def __len__(self):
        return len(self.ring_sizes)


@dataclasses.dataclass(config={"extra": Extra.forbid})
class LipinskiDonor(AtomFeature):
    """
    Return if the atom is a Lipinski h-bond donor.
    """

    type: Literal["lipinskidonor"] = "lipinskidonor"

    def __len__(self):
        return 1

    def __call__(self, molecule: Chem.Mol) -> torch.Tensor:
        from rdkit.Chem import Lipinski

        donors = Lipinski._HDonors(molecule)
        # squash the lists
        donors = [d for donor in donors for d in donor]
        return torch.tensor(
            [int(atom.GetIdx() in donors) for atom in molecule.GetAtoms()]
        ).reshape(-1, 1)


@dataclasses.dataclass(config={"extra": Extra.forbid})
class LipinskiAcceptor(AtomFeature):
    """
    Return if the atom is a Lipinski h-bond acceptor.
    """

    type: Literal["lipinskiacceptor"] = "lipinskiacceptor"

    def __len__(self):
        return 1

    def __call__(self, molecule: Chem.Mol) -> torch.Tensor:
        from rdkit.Chem import Lipinski

        acceptors = Lipinski._HAcceptors(molecule)
        # squash the lists
        acceptors = [a for acceptor in acceptors for a in acceptor]
        return torch.tensor(
            [int(atom.GetIdx() in acceptors) for atom in molecule.GetAtoms()]
        ).reshape(-1, 1)


@dataclasses.dataclass(config={"extra": Extra.forbid})
class PaulingElectronegativity(AtomFeature):
    """
    Return the pauling electronegativity of each of the atoms.
    """

    type: Literal["paulingelectronegativity"] = "paulingelectronegativity"
    # values taken from <https://github.com/AstexUK/ESP_DNN/blob/master/esp_dnn/data/atom_data.csv>
    negativities: dict[int, float] = Field(
        {
            1: 2.2,
            5: 2.04,
            6: 2.55,
            7: 3.04,
            8: 3.44,
            9: 3.98,
            14: 1.9,
            15: 2.19,
            16: 2.58,
            17: 3.16,
            35: 2.96,
            53: 2.66,
        },
        description="The reference negativities for each element.",
    )

    def __len__(self):
        return 1

    def __call__(self, molecule: Chem.Mol) -> torch.Tensor:
        return torch.tensor(
            [self.negativities[atom.GetAtomicNum()] for atom in molecule.GetAtoms()]
        ).reshape(-1, 1)


@dataclasses.dataclass(config={"extra": Extra.forbid})
class SandersonElectronegativity(AtomFeature):
    """
    Return the Sanderson electronegativity of each of the atoms.

    Values taken from <https://github.com/AstexUK/ESP_DNN/blob/master/esp_dnn/data/atom_data.csv>
    """

    type: Literal["sandersonelectronegativity"] = "sandersonelectronegativity"
    negativities: dict[int, float] = Field(
        {
            1: 2.59,
            5: 2.28,
            6: 2.75,
            7: 3.19,
            8: 3.65,
            9: 4.0,
            14: 2.14,
            15: 2.52,
            16: 2.96,
            17: 3.48,
            35: 3.22,
            53: 2.78,
        },
        description="The reference negativities for each element.",
    )

    def __len__(self):
        return 1

    def __call__(self, molecule: Chem.Mol) -> torch.Tensor:
        return torch.Tensor(
            [self.negativities[atom.GetAtomicNum()] for atom in molecule.GetAtoms()]
        ).reshape(-1, 1)


@dataclasses.dataclass(config={"extra": Extra.forbid})
class VDWRadius(AtomFeature):
    """
    Return the vdW radius of the atom.

    Values taken from <https://github.com/AstexUK/ESP_DNN/blob/master/esp_dnn/data/atom_data.csv>
    """

    type: Literal["vdwradius"] = "vdwradius"
    radii: dict[int, float] = Field(
        {
            1: 1.17,
            5: 1.62,
            6: 1.75,
            7: 1.55,
            8: 1.4,
            9: 1.3,
            14: 1.97,
            15: 1.85,
            16: 1.8,
            17: 1.75,
            35: 1.95,
            53: 2.1,
        },
        description="The reference vdW radii in angstroms for each element.",
    )

    def __len__(self):
        return 1

    def __call__(self, molecule: Chem.Mol) -> torch.Tensor:
        return torch.Tensor(
            [self.radii[atom.GetAtomicNum()] for atom in molecule.GetAtoms()]
        ).reshape(-1, 1)


@dataclasses.dataclass(config={"extra": Extra.forbid})
class AtomicPolarisability(AtomFeature):
    """Assign the atomic polarisability for each atom.
    values from <https://github.com/AstexUK/ESP_DNN/blob/master/esp_dnn/data/atom_data.csv>
    """

    type: Literal["atomicpolarisability"] = "atomicpolarisability"
    polarisability: dict[int, float] = Field(
        {
            1: 0.67,
            5: 3.03,
            6: 1.76,
            7: 1.1,
            8: 1.1,
            9: 0.56,
            14: 5.38,
            15: 3.63,
            16: 2.9,
            17: 2.18,
            35: 3.05,
            53: 5.35,
        },
        description="The atomic polarisability in atomic units for each element.",
    )

    def __len__(self):
        return 1

    def __call__(self, molecule: Chem.Mol) -> torch.Tensor:
        return torch.Tensor(
            [self.polarisability[atom.GetAtomicNum()] for atom in molecule.GetAtoms()]
        ).reshape(-1, 1)


@dataclasses.dataclass(config={"extra": Extra.forbid})
class Hybridization(AtomFeature):
    """
    one hot encode the rdkit hybridization of the atom.
    """

    type: Literal["hybridization"] = "hybridization"
    hybridization: list[Chem.rdchem.HybridizationType] = Field(
        [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            Chem.rdchem.HybridizationType.S,
        ],
        description="The list of hybridization types which we can one hot encode",
    )

    def __len__(self):
        return len(self.hybridization)

    def __call__(self, molecule: Chem.Mol) -> torch.Tensor:
        return torch.vstack(
            [
                one_hot_encode(atom.GetHybridization(), self.hybridization)
                for atom in molecule.GetAtoms()
            ]
        )


@dataclasses.dataclass(config={"extra": Extra.forbid})
class TotalValence(AtomFeature):
    type: Literal["totalvalence"] = "totalvalence"

    def __len__(self):
        return 1

    def __call__(self, molecule: Chem.Mol) -> torch.Tensor:
        return torch.Tensor([[atom.GetTotalValence()] for atom in molecule.GetAtoms()])


@dataclasses.dataclass(config={"extra": Extra.forbid})
class ExplicitValence(AtomFeature):
    type: Literal["explicitvalence"] = "explicitvalence"

    def __len__(self):
        return 1

    def __call__(self, molecule: Chem.Mol) -> torch.Tensor:
        return torch.Tensor(
            [[atom.GetExplicitValence()] for atom in molecule.GetAtoms()]
        )


@dataclasses.dataclass(config={"extra": Extra.forbid})
class AtomicMass(AtomFeature):
    type: Literal["atomicmass"] = "atomicmass"

    def __len__(self):
        return 1

    def __call__(self, molecule: Chem.Mol) -> torch.Tensor:
        return torch.Tensor([[atom.GetMass()] for atom in molecule.GetAtoms()])


@dataclasses.dataclass(config={"extra": Extra.forbid})
class TotalDegree(AtomFeature):
    type: Literal["totaldegree"] = "totaldegree"

    def __len__(self):
        return 1

    def __call__(self, molecule: Chem.Mol) -> torch.Tensor:
        return torch.Tensor([[atom.GetTotalDegree()] for atom in molecule.GetAtoms()])


# Register all new features
register_atom_feature(HydrogenAtoms)
register_atom_feature(AtomInRingOfSize)
register_atom_feature(LipinskiDonor)
register_atom_feature(LipinskiAcceptor)
register_atom_feature(PaulingElectronegativity)
register_atom_feature(SandersonElectronegativity)
register_atom_feature(VDWRadius)
register_atom_feature(AtomicPolarisability)
register_atom_feature(Hybridization)
register_atom_feature(TotalValence)
register_atom_feature(ExplicitValence)
register_atom_feature(AtomicMass)
register_atom_feature(TotalDegree)
