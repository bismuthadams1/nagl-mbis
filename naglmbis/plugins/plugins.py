from openff.toolkit.topology import TopologyAtom, TopologyVirtualSite
from openff.toolkit.typing.engines.smirnoff import (
    ElectrostaticsHandler,
    LibraryChargeHandler,
    ParameterAttribute,
    vdWHandler,
)
from openff.toolkit.typing.engines.smirnoff.parameters import (
    _allow_only,
    _NonbondedHandler,
)
from qubekit.charges import MBISCharges
from qubekit.molecules import Ligand

from naglmbis.models import load_charge_model, load_volume_model
from naglmbis.plugins.trained_models import trained_models


class NAGLMBISHandler(_NonbondedHandler):
    """
    A custom handler to allow the use of the pretrained nagl-mbis models with a smirnoff force field.
    """

    _TAGNAME = "NAGLMBIS"
    _DEPENDENCIES = [
        vdWHandler,
        ElectrostaticsHandler,
        LibraryChargeHandler,
    ]  # we need to let waters be handled by libray charges first
    _KWARGS = []
    charge_model = ParameterAttribute(
        default="nagl-v1", converter=_allow_only(["nagl-v1", "espaloma-v1"])
    )
    volume_model = ParameterAttribute(
        default="nagl-v1", converter=_allow_only(["nagl-v1"])
    )
    rfree_model = ParameterAttribute(
        default=1, converter=_allow_only(list(trained_models.keys()))
    )

    def check_handler_compatibility(self, handler_kwargs):
        """We do not want to be mixed with AM1 handler as this is not compatible."""
        pass

    def create_force(self, system, topology, **kwargs):
        if "nagl" in self.charge_model:
            charge_model = load_charge_model(charge_model=self.charge_model)
        elif "espaloma" in self.charge_model:
            from espaloma_charge import charge

            charge_model = charge
        else:
            raise NotImplementedError(
                "Only NAGL and Esaploma type models are supported!"
            )

        volume_model = load_volume_model(volume_model=self.volume_model)
        # the volume and charge models are tied to the trained model
        lj = trained_models[self.rfree_model]

        force = super().create_force(system, topology, **kwargs)

        for ref_mol in topology.reference_molecules:
            # If the molecule has charges then we should skip the molecule
            # this should let us skip water as it has lib charges
            if self.check_charges_assigned(ref_mol, topology):
                continue

            # qubekit assumes conformers so make one if there are non
            if ref_mol.n_conformers == 0:
                ref_mol.generate_conformers(n_conformers=1)

            qb_mol = Ligand.from_rdkit(ref_mol.to_rdkit())
            # before we run make sure we have enough Rfree terms to run
            lj.check_element_coverage(molecule=qb_mol)

            # predict the mbis charges and volumes
            if "nagl" in self.charge_model:
                mbis_charges = charge_model.compute_properties(molecule=ref_mol)[
                    "mbis-charges"
                ].detach()
            elif "espaloma" in self.charge_model:
                mbis_charges = charge_model(ref_mol.to_rdkit())
                mbis_charges = mbis_charges.reshape((ref_mol.n_atoms, 1))
            mbis_volumes = volume_model.compute_properties(molecule=ref_mol)[
                "mbis-volumes"
            ].detach()

            # fix the charges and volumes
            for i in range(qb_mol.n_atoms):
                qb_mol.atoms[i].aim.charge = float(mbis_charges[i][0])
                qb_mol.atoms[i].aim.volume = float(mbis_volumes[i][0])
            # apply charge and volume symmetry
            MBISCharges.apply_symmetrisation(qb_mol)
            # update nonbonded params with aim values and fix net charge
            for i in range(qb_mol.n_atoms):
                atom = qb_mol.atoms[i]
                qb_mol.NonbondedForce.create_parameter(
                    atoms=(i,), charge=atom.aim.charge, sigma=0, epsilon=0
                )

            qb_mol.fix_net_charge()

            # calculate the LJ terms for the model
            lj.run(qb_mol)

            # now assign the parameters in the openmm system
            for topology_molecule in topology._reference_molecule_to_topology_molecules[
                ref_mol
            ]:
                for topology_particle in topology_molecule.atoms:
                    if type(topology_particle) is TopologyAtom:
                        ref_mol_particle_index = (
                            topology_particle.atom.molecule_particle_index
                        )
                    elif type(topology_particle) is TopologyVirtualSite:
                        ref_mol_particle_index = (
                            topology_particle.virtual_site.molecule_particle_index
                        )
                    else:
                        raise ValueError(
                            f"Particles of type {type(topology_particle)} are not supported"
                        )

                    topology_particle_index = topology_particle.topology_particle_index
                    particle_parameters = qb_mol.NonbondedForce[
                        (ref_mol_particle_index,)
                    ]
                    # Set the nonbonded force parameters
                    force.setParticleParameters(
                        topology_particle_index,
                        particle_parameters.charge,
                        particle_parameters.sigma,
                        particle_parameters.epsilon,
                    )

            # Mark that we have assigned the parameters
            self.mark_charges_assigned(ref_mol, topology)
