"""Pressure and temperature solved against each other along a line.

The flow assurance screen took the pressure profile as a straight line between
the inlet and the outlet you typed. That is defensible for screening and wrong
in a way worth fixing, because the two fields are coupled in both directions:

  * Temperature sets Z, density and viscosity, which set the pressure drop.
  * The pressure drop sets the Joule-Thomson cooling, which sets the
    temperature.

Neither can be computed first. So this iterates: guess a temperature profile,
solve the pressures through it, re-solve the temperature against those
pressures, and repeat until the temperature stops moving. It converges quickly
because the coupling is weak in one direction -- a few degrees changes the
pressure drop by a percent or two -- and strong in the other, which is exactly
the structure fixed-point iteration likes.

What changes versus the straight line is not usually the arrival temperature.
It is *where along the line* the pressure falls, and therefore where the
hydrate curve is crossed: a real profile drops pressure fastest at the cold,
dense end rather than uniformly, so the crossing moves.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .components import GasPipe
from .fluids import GasProperties
from .thermal import ThermalPath, ThermalStream, temperature_profile
from .units import r_to_f

# The temperature profile is converged when no station moves more than this
# between sweeps. A hundredth of a degree is far below the uncertainty in the
# heat transfer coefficient, so tightening it further buys nothing real.
TEMPERATURE_TOLERANCE_R = 0.01
MAX_SWEEPS = 40


@dataclass
class CoupledProfile:
    distances_ft: list
    pressures_psia: list
    temperatures_r: list
    sweeps: int
    converged: bool
    max_change_r: float

    @property
    def outlet_psia(self) -> float:
        return self.pressures_psia[-1]

    @property
    def outlet_f(self) -> float:
        return r_to_f(self.temperatures_r[-1])


def _outlet_pressure(pipe: GasPipe, gas: GasProperties, inlet_psia: float,
                     flow_scfd: float) -> float:
    """Invert the flow equation for the downstream pressure of one segment.

    The correlation runs pressure -> flow, and this needs flow -> pressure.
    Flow is monotonically decreasing in the outlet pressure, so bisection is
    both valid and robust -- Newton on this is faster and can step to a
    negative pressure on a segment near its capacity, where the derivative is
    steep.

    Returns 0.0 when the segment cannot pass the requested rate at any outlet
    pressure, which the caller must treat as the line being undersized rather
    than as a converged answer.
    """
    if flow_scfd <= 0.0:
        return inlet_psia

    # Can the segment pass this rate even with a perfect vacuum downstream?
    if pipe.flow(inlet_psia, 1e-6, gas) < flow_scfd:
        return 0.0

    low, high = 1e-6, inlet_psia
    for _ in range(80):
        mid = 0.5 * (low + high)
        if pipe.flow(inlet_psia, mid, gas) > flow_scfd:
            low = mid          # still passing too much: raise the back pressure
        else:
            high = mid
    return 0.5 * (low + high)


def coupled_profile(
    inlet_psia: float,
    inlet_temperature_r: float,
    flow_scfd: float,
    gas: GasProperties,
    path: ThermalPath,
    stream: ThermalStream,
    length_ft: float,
    segments: int = 40,
    joule_thomson_f_per_psi: float = 0.0,
    roughness_in: float = 0.0007,
    efficiency: float = 0.92,
) -> CoupledProfile:
    """Solve pressure and temperature together.

    ``gas`` supplies the composition; its temperature is overridden per segment
    from the temperature profile, which is the entire point of coupling. The
    Joule-Thomson coefficient is passed in rather than re-derived each sweep,
    because re-deriving it inside the loop makes the iteration chase its own
    tail for a change smaller than the uncertainty in U.
    """
    if segments < 1:
        raise ValueError("need at least one segment, got %r" % (segments,))
    if inlet_psia <= 0.0:
        raise ValueError("inlet pressure must be positive, got %r" % (inlet_psia,))
    if flow_scfd <= 0.0:
        raise ValueError("flow must be positive, got %r" % (flow_scfd,))

    dx = length_ft / segments
    distances = [i * dx for i in range(segments + 1)]
    segment_mi = dx / 5280.0

    # Start from an isothermal guess. Any starting profile converges to the
    # same fixed point; this one is simply the closest cheap answer.
    temperatures = [inlet_temperature_r] * (segments + 1)
    pressures = [inlet_psia] * (segments + 1)

    converged = False
    change = math.inf
    sweeps = 0

    for sweeps in range(1, MAX_SWEEPS + 1):
        # Pressures, through the current temperature profile.
        pressures = [inlet_psia]
        for i in range(segments):
            mid_t = 0.5 * (temperatures[i] + temperatures[i + 1])
            local = GasProperties(gravity=gas.gravity, temperature_r=mid_t,
                                  y_co2=gas.y_co2, y_h2s=gas.y_h2s, y_n2=gas.y_n2)
            pipe = GasPipe(a=0, b=1, diameter_in=path.diameter_ft * 12.0,
                           length_mi=segment_mi, roughness_in=roughness_in,
                           efficiency=efficiency)
            nxt = _outlet_pressure(pipe, local, pressures[-1], flow_scfd)
            if nxt <= 0.0:
                # The line cannot pass this rate. Reporting the collapse
                # honestly beats returning a profile through negative
                # pressures that looks like an answer.
                pressures.extend([0.0] * (segments - i))
                break
            pressures.append(nxt)

        if pressures[-1] <= 0.0:
            return CoupledProfile(distances_ft=distances, pressures_psia=pressures,
                                  temperatures_r=temperatures, sweeps=sweeps,
                                  converged=False, max_change_r=math.inf)

        # Temperature, through the pressures just computed.
        profile = temperature_profile(
            inlet_temperature_r, path, stream, length_ft,
            pressures_psia=pressures, segments=segments,
            joule_thomson_f_per_psi=joule_thomson_f_per_psi)

        change = max(abs(a - b) for a, b in zip(profile.temperatures_r, temperatures))
        temperatures = profile.temperatures_r
        if change <= TEMPERATURE_TOLERANCE_R:
            converged = True
            break

    return CoupledProfile(distances_ft=distances, pressures_psia=pressures,
                          temperatures_r=temperatures, sweeps=sweeps,
                          converged=converged, max_change_r=change)
