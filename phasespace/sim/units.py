"""Unit conversions.

The engine works internally in field/customary units (psia, inches, miles,
scf/day, degrees Rankine) because that is what the correlations are
published in and what the target users read off a datasheet. This module
is the boundary: convert at input and output, never mid-calculation.
"""
from __future__ import annotations

# Length
FT_PER_MI = 5280.0
IN_PER_FT = 12.0
M_PER_FT = 0.3048

# Pressure
PSI_PER_BAR = 14.503773773
PSI_PER_KPA = 0.1450377377
ATM_PSIA = 14.6959
STD_PRESSURE_PSIA = 14.65   # contract standard condition used by Weymouth's 433.5 constant
STD_TEMPERATURE_R = 520.0   # 60 degF

# Temperature
RANKINE_OFFSET = 459.67


def f_to_r(t_f: float) -> float:
    return t_f + RANKINE_OFFSET


def r_to_f(t_r: float) -> float:
    return t_r - RANKINE_OFFSET


def c_to_r(t_c: float) -> float:
    return (t_c * 9.0 / 5.0) + 32.0 + RANKINE_OFFSET


def psia_to_bar(p: float) -> float:
    return p / PSI_PER_BAR


def bar_to_psia(p: float) -> float:
    return p * PSI_PER_BAR


def psig_to_psia(p: float, atmospheric_psia: float = ATM_PSIA) -> float:
    return p + atmospheric_psia


def psia_to_psig(p: float, atmospheric_psia: float = ATM_PSIA) -> float:
    return p - atmospheric_psia


# Gas volumetric flow
def scfd_to_mmscfd(q: float) -> float:
    return q / 1e6


def mmscfd_to_scfd(q: float) -> float:
    return q * 1e6


def scfd_to_sm3d(q: float) -> float:
    """scf/day -> standard m3/day (at the same standard conditions)."""
    return q * (M_PER_FT ** 3)


def sm3d_to_scfd(q: float) -> float:
    return q / (M_PER_FT ** 3)


# Liquid volumetric flow
BBL_PER_FT3 = 0.1781076
FT3_PER_BBL = 5.614583


def bpd_to_ft3d(q: float) -> float:
    return q * FT3_PER_BBL


def ft3d_to_bpd(q: float) -> float:
    return q * BBL_PER_FT3


def mi_to_ft(length_mi: float) -> float:
    return length_mi * FT_PER_MI


def ft_to_mi(length_ft: float) -> float:
    return length_ft / FT_PER_MI


# --------------------------------------------------------------- metric ----
#
# The engine works in field units because that is what the correlations are
# published in. Most of the world's operators do not. An Indian, Gulf or
# European asset hands you kg/cm2, cubic metres per day and millimetres, and a
# tool that silently reads those as psi, scf/day and inches produces a
# confident answer that is wrong by orders of magnitude -- 5 ksc read as 5 psia
# is a factor of fourteen, and nothing downstream will complain.
#
# So the conversions live here, at the boundary, and the units a number arrived
# in travel with it until it crosses.

# kg/cm2, universally written "ksc" on Indian and Gulf datasheets and almost
# always gauge rather than absolute.
PSI_PER_KSC = 14.223343307
SCF_PER_M3 = 35.314666721
M_PER_MI = 1609.344
MM_PER_IN = 25.4


def ksc_to_psia(p_ksc: float, atmospheric_psia: float = ATM_PSIA) -> float:
    """Gauge kg/cm2 to absolute psi.

    Gauge by default because that is what a GGS datasheet means by "5 ksc",
    and treating it as absolute understates every pressure by an atmosphere --
    which on a 5 ksc separator is a 20% error, large enough to change an
    answer and small enough to look plausible.
    """
    return p_ksc * PSI_PER_KSC + atmospheric_psia


def psia_to_ksc(p_psia: float, atmospheric_psia: float = ATM_PSIA) -> float:
    return (p_psia - atmospheric_psia) / PSI_PER_KSC


def ksca_to_psia(p_ksca: float) -> float:
    """Absolute kg/cm2 to absolute psi, for the datasheets that say ksca."""
    return p_ksca * PSI_PER_KSC


def m3d_to_scfd(q_m3d: float) -> float:
    return q_m3d * SCF_PER_M3


def scfd_to_m3d(q_scfd: float) -> float:
    return q_scfd / SCF_PER_M3


def m_to_mi(length_m: float) -> float:
    return length_m / M_PER_MI


def mi_to_m(length_mi: float) -> float:
    return length_mi * M_PER_MI


def mm_to_in(d_mm: float) -> float:
    return d_mm / MM_PER_IN


def in_to_mm(d_in: float) -> float:
    return d_in * MM_PER_IN


def c_to_f(t_c: float) -> float:
    return t_c * 9.0 / 5.0 + 32.0


def f_to_c(t_f: float) -> float:
    return (t_f - 32.0) * 5.0 / 9.0
