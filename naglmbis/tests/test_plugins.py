import pytest
from openmm import unit

from naglmbis.plugins import modify_force_field


def test_modify_force_field():
    """Make sure we can correctly modify a force field with a NAGLMBIS tag, this ensures the plugin is picked
    up by the toolkit.
    """
    nagl_sage = modify_force_field(force_field="openff_unconstrained-2.0.0.offxml")
    handlers = nagl_sage.registered_parameter_handlers
    assert "NAGLMBIS" in handlers
    assert "ToolkitAM1BCC" not in handlers


def test_plugin_water(water):
    """Make sure that the default TIP3P parameters are applied to water when using a NAGLMBIS force field."""

    nagl_sage = modify_force_field(force_field="openff_unconstrained-2.0.0.offxml")
    water_system = nagl_sage.create_openmm_system(topology=water.to_topology())
    water_forces = {
        water_system.getForce(index).__class__.__name__: water_system.getForce(index)
        for index in range(water_system.getNumForces())
    }
    # get the oxygen parameters, these should match the library charge
    charge, sigma, epsilon = water_forces["NonbondedForce"].getParticleParameters(0)
    assert charge / unit.elementary_charge == -0.834
    assert sigma / unit.nanometer == 0.31507
    assert epsilon / unit.kilojoule_per_mole == 0.6363864000000001
    # now check the hydrogen
    for i in [1, 2]:
        charge, sigma, epsilon = water_forces["NonbondedForce"].getParticleParameters(i)
        assert charge / unit.elementary_charge == 0.417
        assert sigma / unit.nanometer == 0.1
        assert epsilon / unit.kilojoule_per_mole == 0.0


def test_plugin_methanol(methanol):
    """Make sure the correct parameters are asigned to methanol when using a NAGLMBIS force field."""

    nagl_sage = modify_force_field(force_field="openff_unconstrained-2.0.0.offxml")
    methanol_system = nagl_sage.create_openmm_system(topology=methanol.to_topology())
    methanol_forces = {
        methanol_system.getForce(index).__class__.__name__: methanol_system.getForce(
            index
        )
        for index in range(methanol_system.getNumForces())
    }
    # check the system parameters
    ref_parameters = {
        # index: [charge, sigma, epsilon]
        0: [0.08475, 0.3506905398376649, 0.29246740950730743],
        1: [-0.67143, 0.30824716826324094, 0.42874612945325397],
        2: [0.049413, 0.23126852234757847, 0.07259909774155628],
        3: [0.049413, 0.23126852234757847, 0.07259909774155628],
        4: [0.049413, 0.23126852234757847, 0.07259909774155628],
        5: [0.438441, 0.11098246898497655, 0.41631660852994784],
    }
    for particle_index, refs in ref_parameters.items():
        charge, sigma, epsilon = methanol_forces[
            "NonbondedForce"
        ].getParticleParameters(particle_index)
        assert charge / unit.elementary_charge == refs[0]
        assert sigma / unit.nanometers == pytest.approx(refs[1])
        assert epsilon / unit.kilojoule_per_mole == pytest.approx(refs[2])


def test_plugin_missing_element(iodobezene):
    """Make sure an error is raised when we try to parameterize a molecule with an element not covered by model 1."""
    from qubekit.utils.exceptions import MissingRfreeError

    nagl_sage = modify_force_field(force_field="openff_unconstrained-2.0.0.offxml")
    with pytest.raises(MissingRfreeError):
        _ = nagl_sage.create_openmm_system(topology=iodobezene.to_topology())


def test_plugin_no_conformer(methane_no_conf):
    """Make sure the system can still be made if the refernce molecule has no conformer"""

    nagl_sage = modify_force_field(force_field="openff_unconstrained-2.0.0.offxml")
    _ = nagl_sage.create_openmm_system(topology=methane_no_conf.to_topology())
