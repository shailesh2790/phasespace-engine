"""Steady-state isothermal gas-flow equations for a single pipe segment.

Ground truth for both (a) synthetic training-data generation and (b) the
always-shown deterministic audit companion in the UI (PRD Section 5.1, 6).

Units: field/customary units common in Indian gas-distribution engineering —
pressures in psia, diameter in inches, length in miles, flow (Qh) in
standard cubic feet per day (scf/day, i.e. SCFD — the native output of the
433.5-constant Weymouth equation), temperature in Rankine. Gas gravity
(air = 1) and compressibility factor are dimensionless.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Standard conditions
T_SC = 520.0  # deg R (60 deg F)
P_SC = 14.65  # psia


@dataclass(frozen=True)
class GasProperties:
    gravity: float = 0.65  # gas specific gravity, air = 1
    z_factor: float = 0.95  # average compressibility factor
    temperature_r: float = 520.0  # flowing temperature, deg R


def weymouth_flow(
    p1: float,
    p2: float,
    diameter_in: float,
    length_mi: float,
    gas: GasProperties = GasProperties(),
    efficiency: float = 0.92,
) -> float:
    """Weymouth equation: steady, isothermal, horizontal gas flow (scf/day).

    Q = 433.5 * E * (T_sc/P_sc) * sqrt((P1^2 - P2^2) / (G * T * L * Z)) * D^2.667

    p1, p2: upstream/downstream pressure (psia). Sign of (p1^2 - p2^2) sets
    flow direction; magnitude uses |p1^2 - p2^2|.

    dp2 is computed as (p1-p2)*(p1+p2) rather than p1*p1-p2*p2: pipe loops
    routinely have near-equal end pressures, and squaring-then-subtracting
    two large nearly-equal numbers loses precision by catastrophic
    cancellation, which otherwise stalls the network solver's Newton
    iteration right where it matters most (loop ties).
    """
    dp2 = (p1 - p2) * (p1 + p2)
    sign = 1.0 if dp2 >= 0 else -1.0
    dp2 = abs(dp2)
    q = (
        433.5
        * efficiency
        * (T_SC / P_SC)
        * math.sqrt(dp2 / (gas.gravity * gas.temperature_r * length_mi * gas.z_factor))
        * diameter_in ** 2.667
    )
    return sign * q


def weymouth_flow_and_grad(
    p1: float,
    p2: float,
    diameter_in: float,
    length_mi: float,
    gas: GasProperties = GasProperties(),
    efficiency: float = 0.92,
) -> tuple[float, float, float]:
    """Weymouth flow plus its exact partials (dQ/dp1, dQ/dp2).

    Used by the network solver's Newton-Raphson Jacobian in place of a
    finite-difference approximation: near a loop tie's near-zero flow, no
    single finite-difference step size is both fine enough to resolve the
    true local slope and coarse enough to avoid floating-point noise, which
    stalls Newton in a limit cycle. The analytic partial has no such
    step-size tradeoff — d(sqrt(dp2))/dp1 = p1/sqrt(dp2) is exact regardless
    of how small dp2 is (a small epsilon floor only guards the division).
    """
    dp2 = (p1 - p2) * (p1 + p2)
    sign = 1.0 if dp2 >= 0 else -1.0
    # Epsilon only needs to guard against literal division-by-zero (a
    # perfectly balanced tie), so it must be far smaller than any physically
    # meaningful dp2 (a real near-zero loop-tie flow can have |dp2| as small
    # as 1e-3 psia^2) — a floor scaled to p^2 would bias exactly those cases.
    dp2_abs = max(abs(dp2), 1e-12)
    const = (
        433.5
        * efficiency
        * (T_SC / P_SC)
        / math.sqrt(gas.gravity * gas.temperature_r * length_mi * gas.z_factor)
        * diameter_in ** 2.667
    )
    q = sign * const * math.sqrt(dp2_abs)
    inv_sqrt = 1.0 / math.sqrt(dp2_abs)
    dq_dp1 = const * p1 * inv_sqrt
    dq_dp2 = -const * p2 * inv_sqrt
    return q, dq_dp1, dq_dp2


def general_pressure_squared_flow(
    p1: float,
    p2: float,
    diameter_in: float,
    length_mi: float,
    friction_factor: float,
    gas: GasProperties = GasProperties(),
    efficiency: float = 1.0,
) -> float:
    """General pressure-squared (fully-turbulent / AGA-style) flow equation.

    Q = 77.54 * E * (T_sc/P_sc) * sqrt((P1^2-P2^2) / (G*T*L*Z*f)) * D^2.5

    `friction_factor` is the Darcy friction factor. The constant was 38.77
    here originally, which is the constant for the *transmission-factor*
    form of the same equation (Q = 38.77 * F * ... with F = 2/sqrt(f)) —
    pairing it with a Darcy f understated flow by exactly a factor of two.

    Kept alongside Weymouth as a second deterministic anchor — the PRD names
    both correlations explicitly (Section 3, Appendix).
    """
    dp2 = (p1 - p2) * (p1 + p2)
    sign = 1.0 if dp2 >= 0 else -1.0
    dp2 = abs(dp2)
    q = (
        77.54
        * efficiency
        * (T_SC / P_SC)
        * math.sqrt(dp2 / (gas.gravity * gas.temperature_r * length_mi * gas.z_factor * friction_factor))
        * diameter_in ** 2.5
    )
    return sign * q
