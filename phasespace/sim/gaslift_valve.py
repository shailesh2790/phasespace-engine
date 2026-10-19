"""Whether the gas can physically get into the well.

The allocator answers "how should the gas be shared". It does not ask whether
the gas can cross into the tubing at all, and that is a separate question with
a separate answer.

Lift gas travels down the casing-tubing annulus and enters the tubing through
a valve port at depth. How much can cross depends on the pressure difference
across that port, and the upstream side of it is set by the compressor. A
compressor at 1,200 psi cannot push 2 MMscf/d through a quarter-inch port
against 900 psi of tubing, whatever the allocation says is optimal.

So an allocation computed without this can return a setpoint nobody is able to
execute. That is the same class of failure as allocating a well below its
minimum stable injection -- a number that looks like an answer and is not
one -- arriving by a different route.

Three pieces, each a published correlation:

  Thornhill-Craver        port capacity for a given pressure drop
  static gas column       casing pressure at the injection depth
  critical pressure ratio where the port chokes and stops caring about tubing

The useful consequence for an operator is not the capacity number itself. It
is which constraint binds. A well held back by its valve port needs a bigger
port, a few hours of wireline. A well held back by compressor discharge
pressure needs a different machine. Those are different projects with
different budgets, and a tool that reports only "this well is at its limit"
sends someone to price the wrong one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Ratio of specific heats for natural gas. 1.27 is the usual working value,
# and the critical pressure ratio it implies is 0.551 -- which is where the
# familiar "choked below about 0.55" comes from.
GAS_HEAT_CAPACITY_RATIO = 1.27

# Thornhill-Craver discharge coefficient for a square-edged orifice. The
# equation is known to over-predict real valve capacity -- there is a stem in
# the way and the port is not a clean orifice -- so `derating` is exposed for
# an operator's own factor rather than this being presented as the truth.
THORNHILL_CRAVER_CD = 0.865

# Field-unit constant giving Mscf/day from area in in^2 and pressure in psia.
THORNHILL_CRAVER_CONSTANT = 155.5

GRAVITY_FT_S2 = 32.174

# Mscf/day into the units an allocation is working in. Stated explicitly
# because a gas-lift calculation mixing Mscf and sm3 is wrong by a factor of
# about 35 and looks entirely plausible on the page.
SCF_PER_MSCF = 1000.0
SM3_PER_MSCF = 28.317


def critical_pressure_ratio(k: float = GAS_HEAT_CAPACITY_RATIO) -> float:
    """Downstream/upstream ratio below which the port is choked.

    Below it the port passes everything it can, and lowering the tubing
    pressure further buys nothing. That is why a gas-lift valve's capacity is
    set by the casing side rather than the tubing side over most of its
    working range, and why raising compressor pressure helps where lowering
    separator pressure does not.
    """
    if k <= 1.0:
        raise ValueError("ratio of specific heats must exceed 1, got %r" % (k,))
    return (2.0 / (k + 1.0)) ** (k / (k - 1.0))


def thornhill_craver_rate(
    port_area_in2: float,
    upstream_psia: float,
    downstream_psia: float,
    temperature_r: float,
    gas_gravity: float = 0.65,
    z_factor: float = 0.9,
    discharge_coefficient: float = THORNHILL_CRAVER_CD,
    k: float = GAS_HEAT_CAPACITY_RATIO,
    derating: float = 1.0,
) -> float:
    """Gas rate through a valve port, in Mscf/day.

    The industry's standard estimate of gas-lift valve port capacity, and one
    that is known to run high against measured valve performance. Taking it at
    face value sizes a system optimistically; `derating` is there so a user can
    apply their own factor, and it defaults to 1.0 so the raw correlation is
    what you get unless you ask for something else.
    """
    if port_area_in2 <= 0.0:
        raise ValueError("port area must be positive, got %r" % (port_area_in2,))
    if upstream_psia <= 0.0:
        raise ValueError("upstream pressure must be positive, got %r" % (upstream_psia,))
    if downstream_psia < 0.0:
        raise ValueError("downstream pressure cannot be negative, got %r"
                         % (downstream_psia,))
    if temperature_r <= 0.0 or gas_gravity <= 0.0 or z_factor <= 0.0:
        raise ValueError("temperature, gas gravity and Z must all be positive")
    if not 0.0 < derating <= 1.0:
        raise ValueError("derating must be in (0, 1], got %r" % (derating,))

    if downstream_psia >= upstream_psia:
        # No driving force. The tubing is at or above casing pressure, so the
        # valve passes nothing -- returned as zero rather than as the negative
        # root the expression would otherwise take.
        return 0.0

    ratio = downstream_psia / upstream_psia
    # Choked: below the critical ratio the port cannot pass more however far
    # the downstream pressure falls, so the expansion term is evaluated at the
    # critical ratio instead of the actual one.
    ratio = max(ratio, critical_pressure_ratio(k))

    expansion = (ratio ** (2.0 / k)) - (ratio ** ((k + 1.0) / k))
    group = ((2.0 * GRAVITY_FT_S2 * k / (k - 1.0)) * expansion
             / (gas_gravity * temperature_r * z_factor))
    if group <= 0.0:
        return 0.0

    return (THORNHILL_CRAVER_CONSTANT * discharge_coefficient * port_area_in2
            * upstream_psia * math.sqrt(group) * derating)


def casing_pressure_at_depth(
    surface_psia: float,
    depth_ft: float,
    temperature_r: float,
    gas_gravity: float = 0.65,
    z_factor: float = 0.9,
) -> float:
    """Static gas column: casing pressure at the injection point.

        p(depth) = p_surface * exp(0.01877 * gravity * depth / (T * Z))

    The gas in the annulus weighs something. On a 6,000 ft well that is 15 to
    20 percent above surface pressure, so ignoring it under-predicts what the
    compressor delivers downhole -- conservative, but by enough to call a well
    pressure-limited when it is not.
    """
    if surface_psia <= 0.0:
        raise ValueError("surface pressure must be positive, got %r" % (surface_psia,))
    if depth_ft < 0.0:
        raise ValueError("depth cannot be negative, got %r" % (depth_ft,))
    if temperature_r <= 0.0 or z_factor <= 0.0:
        raise ValueError("temperature and Z must be positive")
    return surface_psia * math.exp(
        0.01877 * gas_gravity * depth_ft / (temperature_r * z_factor))


@dataclass
class InjectionLimit:
    """The most gas a compressor can actually put into one well."""

    well: str
    rate: float
    casing_at_depth_psia: float
    tubing_at_depth_psia: float
    choked: bool
    port_diameter_in: float = 0.0
    binding: bool = False

    @property
    def differential_psi(self) -> float:
        return self.casing_at_depth_psia - self.tubing_at_depth_psia

    def describe(self) -> str:
        if self.rate <= 0.0:
            return ("%s cannot take gas: tubing at %.0f psia is at or above the "
                    "%.0f psia the compressor delivers at depth"
                    % (self.well, self.tubing_at_depth_psia,
                       self.casing_at_depth_psia))
        return ("%s can take %.0f at a %.0f psi differential%s"
                % (self.well, self.rate, self.differential_psi,
                   " (choked)" if self.choked else ""))


def injection_capacity(
    name: str,
    compressor_discharge_psia: float,
    injection_depth_ft: float,
    tubing_at_depth_psia: float,
    port_diameter_in: float,
    temperature_r: float,
    gas_gravity: float = 0.65,
    z_factor: float = 0.9,
    derating: float = 1.0,
    units_per_mscf: float = SCF_PER_MSCF,
) -> InjectionLimit:
    """How much gas this well can take, given the compressor and the valve."""
    if port_diameter_in <= 0.0:
        raise ValueError("port diameter must be positive, got %r" % (port_diameter_in,))

    casing_at_depth = casing_pressure_at_depth(
        compressor_discharge_psia, injection_depth_ft, temperature_r,
        gas_gravity, z_factor)

    area = math.pi * (port_diameter_in ** 2) / 4.0
    rate_mscfd = thornhill_craver_rate(
        port_area_in2=area, upstream_psia=casing_at_depth,
        downstream_psia=tubing_at_depth_psia, temperature_r=temperature_r,
        gas_gravity=gas_gravity, z_factor=z_factor, derating=derating)

    return InjectionLimit(
        well=name, rate=rate_mscfd * units_per_mscf,
        casing_at_depth_psia=casing_at_depth,
        tubing_at_depth_psia=tubing_at_depth_psia,
        choked=(tubing_at_depth_psia / casing_at_depth) <= critical_pressure_ratio(),
        port_diameter_in=port_diameter_in)


def apply_injection_limits(wells: list, limits: list) -> list:
    """Cap each well's valve limit at what the compressor can deliver.

    Lowers `q_max` so the allocator cannot return a rate nobody can execute,
    and marks which limits actually bind.

    A binding injection-pressure limit is a finding rather than a nuisance:
    the well wants more gas, the economics want to give it more gas, and the
    compressor discharge pressure is what stops it. Fixing that is a different
    project from buying throughput, and a tool that says only "at limit" sends
    someone to price the wrong machine.
    """
    by_name = {w.name: w for w in wells}
    for limit in limits:
        well = by_name.get(limit.well)
        if well is None:
            raise ValueError("injection limit given for unknown well %r"
                             % (limit.well,))
        limit.binding = limit.rate < well.q_max
        if limit.binding:
            well.q_max = max(limit.rate, 0.0)
            if well.q_max < well.q_min:
                # The compressor cannot even reach this well's minimum stable
                # injection, so it is not a candidate for gas at all. Setting
                # q_min above q_max would be an invalid well; zeroing both is
                # the honest encoding of "this well cannot be lifted today".
                well.q_min = 0.0
                well.q_max = 0.0
    return limits
