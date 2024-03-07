# a file to track bcc models
from typing_extensions import Literal
from openff.toolkit.typing.engines.smirnoff import ForceField

# Model fit with nagl-v1 charges and nagl-v1 volumes with no polar h Rfree
# list of smirks and charge corrections
bcc_model_v1 = """
<SMIRNOFF version="0.3" aromaticity_model="OEAroModel_MDL">
    <BondChargeCorrection version="0.3">
        <BCC smirks="[#1:1]-[#6a:2]" charge_correction="1.918287613579e-04 * elementary_charge" id="bcc-0" />
        <BCC smirks="[#6a:1]:[#7X2a:2]" charge_correction="-6.515627555745e-02 * elementary_charge" id="bcc-1" />
        <BCC smirks="[#6a:1]-[$([NX3](=O)=O),$([NX3+](=O)[O-]):2]" charge_correction="0 * elementary_charge" id="bcc-2"/>
        <BCC smirks="[#6a:1]-[NX3;H2;!$(NC=O):2]" charge_correction="-3.287530301247e-03 * elementary_charge" id="bcc-3" />
        <BCC smirks="[#1:1]-[NX3;H2;!$(NC=O):2]-[#6a]" charge_correction="-2.385134696800e-02 * elementary_charge" id="bcc-4" />
        <BCC smirks="[#6a:1]-[#9:2]" charge_correction="4.111091863251e-02 * elementary_charge" id="bcc-5" />
        <BCC smirks="[#6a:1]-[#35:2]" charge_correction="0 * elementary_charge" id="bcc-6"/>
        <BCC smirks="[#6a]-[#6X2:1]#[#7X1:2]" charge_correction="0 * elementary_charge" id="bcc-7"/>
        <BCC smirks="[#6a:1]-[#8X2H1:2]" charge_correction="0 * elementary_charge" id="bcc-8"/>
        <BCC smirks="[#8X1H0:1]=[#16:2]" charge_correction="1.194201883302e-01 * elementary_charge" id="bcc-9" />
        <BCC smirks="[#6!a:1]-[#8X2H1:2]" charge_correction="-4.327420532297e-02 * elementary_charge" id="bcc-10" />
        <BCC smirks="[$([#6X3!a](C)(C)),$([#6X3H1!a](C)):1]=[#8X1H0:2]" charge_correction="-1.226504648139e-01 * elementary_charge" id="bcc-11" />
        <BCC smirks="[#6!a:1]-[#16X2H0:2]" charge_correction="-7.065983295162e-02 * elementary_charge" id="bcc-12" />
        <BCC smirks="[#6!a:1]-[#16X2H1:2]" charge_correction="0 * elementary_charge" id="bcc-13"/>
        <BCC smirks="[$([NX3](=O)=O),$([NX3+](=O)[O-]):1]-,=[#8X1:2]" charge_correction="0 * elementary_charge" id="bcc-14"/>
        <BCC smirks="[#6!a:1]-[#17:2]" charge_correction="7.067734595593e-03 * elementary_charge" id="bcc-15" />
        <BCC smirks="[#6!a:1]-[#35:2]" charge_correction="-3.473056879281e-02 * elementary_charge" id="bcc-16" />
        <BCC smirks="[#6!a:1]-[NX3;H2;!$(NC=O):2]" charge_correction="0 * elementary_charge" id="bcc-17"/>
        <BCC smirks="[*!a]-[#6X2:1]#[#7X1:2]" charge_correction="0 * elementary_charge" id="bcc-18"/>
    </BondChargeCorrection>
</SMIRNOFF>
"""

BCC_MODELS = Literal["nagl-v1"]
bcc_force_fields = {"nagl-v1": bcc_model_v1}


def load_bcc_model(bcc_model: BCC_MODELS):
    """
    Load the BCC parameter handler with the requested model.

    Return:
        The BCC plugin parameter handler from qubekit which can be used to get matches
    """
    ff = ForceField(bcc_force_fields[bcc_model], load_plugins=True)
    return ff.get_parameter_handler("BondChargeCorrection")
