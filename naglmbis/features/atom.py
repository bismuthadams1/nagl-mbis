from typing import List, Optional

import torch
from nagl.features import AtomFeature, one_hot_encode
from openff.toolkit.topology import Molecule

# from nagl.resonance import enumerate_resonance_forms
# from nagl.utilities.toolkits import normalize_molecule
from rdkit import Chem

# class AtomAverageFormalCharge(AtomFeature):
#     def __call__(self, molecule: "Molecule") -> torch.Tensor:
#         try:
#             molecule = normalize_molecule(molecule)
#         except AssertionError:
#             pass
#
#         resonance_forms = enumerate_resonance_forms(
#             molecule=molecule,
#             lowest_energy_only=True,
#             as_dicts=True,
#             include_all_transfer_pathways=True,
#         )
#         formal_charges = [
#             [
#                 atom["formal_charge"]
#                 for resonance_form in resonance_forms
#                 if i in resonance_forms["atoms"]
#                 for atom in resonance_form["atoms"][i]
#             ]
#             for i in range(molecule.n_atoms)
#         ]
#         feature_tensor = torch.tensor(
#             [
#                 [
#                     sum(formal_charges[i]) / len(formal_charges[i])
#                     if len(formal_charges[i]) > 0
#                     else 0.0
#                 ]
#                 for i in range(molecule.n_atoms)
#             ]
#         )
#         return feature_tensor
#
#     def __len__(self):
#         return 1


class HydrogenAtoms(AtomFeature):
    """One hot encode the number of bonded hydrogen atoms"""

    _HYDROGENS = [0, 1, 2, 3, 4]

    def __init__(self, hydrogens: Optional[List[int]] = None) -> None:
        self.hydrogens = hydrogens if hydrogens is not None else [*self._HYDROGENS]

    def __call__(self, molecule: "Molecule") -> torch.Tensor:
        return torch.vstack(
            [
                one_hot_encode(
                    sum(
                        [
                            n.atomic_number
                            for n in atom.bonded_atoms
                            if n.atomic_number == 1
                        ]
                    ),
                    self.hydrogens,
                )
                for atom in molecule.atoms
            ]
        )

    def __len__(self):
        return len(self.hydrogens)


class AtomInRingOfSize(AtomFeature):
    def __init__(self, ring_size: int) -> None:
        assert ring_size >= 3
        self.ring_size = ring_size

    def __call__(self, molecule: "Molecule") -> torch.Tensor:
        rd_molecule: Chem.Mol = molecule.to_rdkit()
        ring_info: Chem.RingInfo = rd_molecule.GetRingInfo()

        return torch.tensor(
            [
                int(ring_info.IsAtomInRingOfSize(atom.GetIdx(), self.ring_size))
                for atom in rd_molecule.GetAtoms()
            ]
        ).reshape(-1, 1)

    def __len__(self):
        return 1


class BondInRingOfSize(AtomFeature):
    def __init__(self, ring_size: int):
        assert ring_size >= 3
        self.ring_size = ring_size

    def __call__(self, molecule: "Molecule") -> torch.Tensor:
        rd_molecule: Chem.Mol = molecule.to_rdkit()
        ring_info: Chem.RingInfo = rd_molecule.GetRingInfo()

        rd_bond_by_index = {
            tuple(sorted((rd_bond.GetBgnIdx(), rd_bond.GetEndIdx()))): rd_bond
            for rd_bond in rd_molecule.GetBonds()
        }

        rd_bonds = [
            rd_bond_by_index[tuple(sorted((bond.atom1_index, bond.atom2_index)))]
            for bond in molecule.bonds
        ]

        return torch.tensor(
            [
                int(ring_info.IsBondInRingOfSize(rd_bond, self.ring_size))
                for rd_bond in rd_bonds
            ]
        ).reshape(-1, 1)

    def __len__(self):
        return 1


class LipinskiDonor(AtomFeature):
    """
    Return if the atom is a Lipinski h-bond donor.
    """

    def __len__(self):
        return 1

    def __call__(self, molecule: Molecule, *args, **kwargs) -> torch.Tensor:
        from rdkit.Chem import Lipinski

        rd_molecule: Chem.Mol = molecule.to_rdkit()
        donors = Lipinski._HDonors(rd_molecule)
        # squash the lists
        donors = [d for donor in donors for d in donor]
        return torch.tensor(
            [int(atom.GetIdx() in donors) for atom in rd_molecule.GetAtoms()]
        ).reshape(-1, 1)


class LipinskiAcceptor(AtomFeature):
    """
    Return if the atom is a Lipinski h-bond acceptor.
    """

    def __len__(self):
        return 1

    def __call__(self, molecule: Molecule, *args, **kwargs):
        from rdkit.Chem import Lipinski

        rd_molecule: Chem.Mol = molecule.to_rdkit()
        acceptors = Lipinski._HAcceptors(rd_molecule)
        # squash the lists
        acceptors = [a for acceptor in acceptors for a in acceptor]
        return torch.tensor(
            [int(atom.GetIdx() in acceptors) for atom in rd_molecule.GetAtoms()]
        ).reshape(-1, 1)


class PaulingElectronegativity(AtomFeature):
    """
    Return the pauling electronegativity of each of the atoms.
    """

    # values taken from <https://github.com/AstexUK/ESP_DNN/blob/master/esp_dnn/data/atom_data.csv>
    _negativities = {
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
    }

    def __len__(self):
        return 1

    def __call__(self, molecule: Molecule, *args, **kwargs):
        return torch.tensor(
            [self._negativities[atom.atomic_number] for atom in molecule.atoms]
        ).reshape(-1, 1)


class SandersonElectronegativity(AtomFeature):
    """
    Return the Sanderson electronegativity of each of the atoms.

    Values taken from <https://github.com/AstexUK/ESP_DNN/blob/master/esp_dnn/data/atom_data.csv>
    """

    _negativities = {
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
    }

    def __len__(self):
        return 1

    def __call__(self, molecule: Molecule, *args, **kwargs):
        return torch.Tensor(
            [self._negativities[atom.atomic_number] for atom in molecule.atoms]
        ).reshape(-1, 1)


class vdWRadius(AtomFeature):
    """
    Return the vdW radius of the atom.

    Values taken from <https://github.com/AstexUK/ESP_DNN/blob/master/esp_dnn/data/atom_data.csv>
    """

    _radii = {
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
    }

    def __len__(self):
        return 1

    def __call__(self, molecule: Molecule, *args, **kwargs):
        return torch.Tensor(
            [self._radii[atom.atomic_number] for atom in molecule.atoms]
        ).reshape(-1, 1)


class AtomicPolarisability(AtomFeature):
    """Assign the atomic polarisability for each atom.
    values from <https://github.com/AstexUK/ESP_DNN/blob/master/esp_dnn/data/atom_data.csv>
    """

    _polarisability = {
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
    }

    def __len__(self):
        return 1

    def __call__(self, molecule: Molecule, *args, **kwargs):
        return torch.Tensor(
            [self._polarisability[atom.atomic_number] for atom in molecule.atoms]
        ).reshape(-1, 1)
