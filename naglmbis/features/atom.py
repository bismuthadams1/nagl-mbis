from typing import List, Optional

import torch
from nagl.features import AtomFeature, one_hot_encode

# from nagl.resonance import enumerate_resonance_forms
# from nagl.utilities.toolkits import normalize_molecule
from openeye import oechem
from openff.toolkit.topology import Molecule

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
        oe_molecule: oechem.OEMol = molecule.to_openeye()
        oechem.OEFindRingAtomsAndBonds(oe_molecule)

        return torch.tensor(
            [
                int(oechem.OEAtomIsInRingSize(oe_atom, self.ring_size))
                for oe_atom in oe_molecule.GetAtoms()
            ]
        ).reshape(-1, 1)

    def __len__(self):
        return 1


class BondInRingOfSize(AtomFeature):
    def __init__(self, ring_size: int):
        assert ring_size >= 3
        self.ring_size = ring_size

    def __call__(self, molecule: "Molecule") -> torch.Tensor:
        oe_molecule: oechem.OEMol = molecule.to_openeye()
        oechem.OEFindRingAtomsAndBonds(oe_molecule)

        oe_bond_by_index = {
            tuple(sorted((oe_bond.GetBgnIdx(), oe_bond.GetEndIdx()))): oe_bond
            for oe_bond in oe_molecule.GetBonds()
        }

        oe_bonds = [
            oe_bond_by_index[tuple(sorted((bond.atom1_index, bond.atom2_index)))]
            for bond in molecule.bonds
        ]

        return torch.tensor(
            [
                int(oechem.OEBondIsInRingSize(oe_bond, self.ring_size))
                for oe_bond in oe_bonds
            ]
        ).reshape(-1, 1)

    def __len__(self):
        return 1
