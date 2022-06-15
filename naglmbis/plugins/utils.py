from openff.toolkit.typing.engines.smirnoff import ForceField


def modify_force_field(force_field: str) -> ForceField:
    """
    Modify a base OpenFF force field by adding a NAGLMBIS handler and removing the AM1BCC handler.
    """
    ff = ForceField(force_field, load_plugins=True)
    ff.deregister_parameter_handler("ToolkitAM1BCC")
    ff.get_parameter_handler("NAGLMBIS")
    return ff
