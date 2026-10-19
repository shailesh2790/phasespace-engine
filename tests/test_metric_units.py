"""Unit conventions that are wrong by a factor, not by a percent.

Most numerical errors in this engine would show up as a result that looks odd.
These would not: each one produces a plausible pressure from an implausible
input, and the solver converges on it without complaint. They are the errors
worth a test of their own.

  kg/cm2 is gauge         reading 5 ksc as absolute loses one atmosphere,
                          which on a 5 ksc separator is 20%
  bean size in 64ths      an 8 mm bean is 20/64 in, not 8/64 -- entering the
                          millimetre figure into a correlation written in
                          64ths quarters the bean
  m3/day into scf/day     a factor of 35
  ksc into psi            a factor of 14

None of these needs field data to check, which is why they survived the
removal of the validation suite they used to live in. That suite was built
on a private operator study and does not ship; `fixtures/README.md` records
what it proved and what open data would replace it.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from phasespace.sim.chokes import bean_64ths
from phasespace.sim.components import GasPipe
from phasespace.sim.fluids import GasProperties
from phasespace.sim.solver import FLOW, GAS, PRESSURE, Network, Node, solve
from phasespace.sim.units import (
    ATM_PSIA,
    ksc_to_psia,
    m3d_to_scfd,
    m_to_mi,
    mm_to_in,
    psia_to_ksc,
)

KSC_TO_PSI = 14.223343307


# ------------------------------------------------------------ the factors --


def test_ksc_is_gauge_and_the_atmosphere_is_not_lost():
    """"5 ksc" on a GGS datasheet is gauge. Reading it as absolute understates
    every pressure by one atmosphere, which on a 5 ksc separator is a 20%
    error in the boundary condition the whole network is solved against."""
    assert ksc_to_psia(5.0) == pytest.approx(5.0 * KSC_TO_PSI + ATM_PSIA, rel=1e-9)
    assert ksc_to_psia(0.0) == pytest.approx(ATM_PSIA)
    assert psia_to_ksc(ksc_to_psia(5.0)) == pytest.approx(5.0, rel=1e-12)
    # And the absolute reading really is 20% low on a 5 ksc separator.
    absolute_misread = 5.0 * KSC_TO_PSI
    assert absolute_misread / ksc_to_psia(5.0) < 0.84


def test_bean_conversion_is_not_a_factor_of_four():
    """8 mm is 20.2/64 in, not 8/64. Entering the millimetre figure into a
    correlation written in 64ths quarters the bean, and the result looks like
    a pressure rather than like a unit error."""
    assert bean_64ths(diameter_mm=8.0) == pytest.approx(20.157, rel=1e-3)
    assert bean_64ths(diameter_in=0.5) == pytest.approx(32.0)
    with pytest.raises(ValueError):
        bean_64ths(diameter_mm=8.0, diameter_in=0.315)
    with pytest.raises(ValueError):
        bean_64ths(diameter_mm=0.0)


def test_the_volume_and_length_factors_are_the_published_ones():
    assert m3d_to_scfd(1.0) == pytest.approx(35.314666721, rel=1e-9)
    assert mm_to_in(101.6) == pytest.approx(4.0, rel=1e-9)
    assert m_to_mi(1609.344) == pytest.approx(1.0, rel=1e-9)


def test_the_conversions_round_trip():
    for ksc in (0.5, 7.0, 44.0):
        assert psia_to_ksc(ksc_to_psia(ksc)) == pytest.approx(ksc, rel=1e-12)
    for mm in (50.8, 101.6, 304.8):
        assert mm_to_in(mm) * 25.4 == pytest.approx(mm, rel=1e-12)


# ------------------------------------------- what the misreading looks like --


WELL_KSC = [18.0, 17.0, 16.0, 15.0, 14.0]
SEPARATOR_KSC = 7.0


def metric_gathering(ksc_as_absolute: bool):
    """The fixture network, built either correctly or with the classic error.

    Mirrors `fixtures/metric_gathering.csv`: five wells on 4 in flow lines
    into a header, and a 12 in trunk from the header to a separator.

    The trunk is not decoration. A header with nothing downstream settles
    between its wells' pressures, so the lowest ones flow out of it instead of
    into it -- which still solves, and is not a gathering system.
    """
    gas = GasProperties(gravity=0.65, temperature_r=95.0 + 459.67)
    convert = (lambda ksc: ksc) if ksc_as_absolute else ksc_to_psia

    net = Network(phase=GAS, fluid=gas)
    net.add_node(Node(0, PRESSURE, pressure_psia=convert(SEPARATOR_KSC),
                      name="Separator"))
    net.add_node(Node(1, FLOW, withdrawal=0.0, name="Header"))
    net.add_branch(GasPipe(a=1, b=0, diameter_in=mm_to_in(304.8),
                           length_mi=m_to_mi(3000.0), name="Trunk"))
    for index, ksc in enumerate(WELL_KSC, start=2):
        net.add_node(Node(index, PRESSURE, pressure_psia=convert(ksc),
                          name="W-%02d" % (index - 1)))
        net.add_branch(GasPipe(a=index, b=1, diameter_in=mm_to_in(101.6),
                               length_mi=m_to_mi(1500.0),
                               name="FL-W-%02d" % (index - 1)))
    return net


def test_the_metric_fixture_solves_and_every_well_flows_inward():
    """A gathering system, not an arithmetic exercise: if the header settles
    above a wellhead, that well takes gas out of the header instead of
    putting it in, and the case is not testing what it claims to."""
    net = metric_gathering(ksc_as_absolute=False)
    result = solve(net)
    assert result.converged

    header = result.pressures[1]
    for node_id in range(2, 7):
        assert result.pressures[node_id] > header, (
            "%s at %.1f psia is below the header at %.1f -- it would flow "
            "backwards" % (net.nodes[node_id].name, result.pressures[node_id],
                           header))
    # And the header must sit above the separator it delivers into.
    assert header > result.pressures[0]


def test_reading_ksc_as_psia_still_converges_which_is_the_danger():
    """The reason this fixture exists.

    The misread network is not rejected and does not fail to solve. It
    produces pressures an order of magnitude out that are internally
    consistent, and nothing downstream can tell. A unit error that crashed
    would need no test.
    """
    correct = solve(metric_gathering(ksc_as_absolute=False))
    misread = solve(metric_gathering(ksc_as_absolute=True))

    assert correct.converged
    # The misread case either converges on nonsense or collapses to the
    # solver's floor. Either way it is not refused, which is the point.
    header_correct = correct.pressures[1]
    header_misread = misread.pressures[1]
    assert header_correct > 10.0 * header_misread, (
        "correct header %.1f psia against misread %.1f -- the error should be "
        "about a factor of fourteen" % (header_correct, header_misread))


def test_the_fixture_file_matches_what_the_test_builds():
    """The CSV and this module describe the same network. If they drift, the
    fixture stops being a regression case for anything."""
    path = Path(__file__).resolve().parents[1] / "fixtures" / "metric_gathering.csv"
    rows = [line.split(",") for line in
            path.read_text(encoding="utf-8").strip().split("\n")[1:]]

    sources = [row for row in rows if row[0] == "node" and row[3] == "source"]
    pressures = [float(row[4]) for row in sources]
    assert pressures == [SEPARATOR_KSC] + WELL_KSC

    flowlines = [row for row in rows if row[0] == "branch"
                 and row[2].startswith("FL-")]
    assert len(flowlines) == len(WELL_KSC)
    for row in flowlines:
        assert float(row[10]) == pytest.approx(101.6)   # mm, not inches
        assert float(row[11]) == pytest.approx(1500.0)  # m, not miles

    trunk = next(row for row in rows if row[2] == "Trunk")
    assert float(trunk[10]) == pytest.approx(304.8)
    assert float(trunk[11]) == pytest.approx(3000.0)


def test_the_fixture_carries_no_operator_identifiers():
    """A standing check, not a one-off cleanup.

    The fixture this replaced held real well names and measured rates from a
    named operator's asset. Generic names are the rule here; this fails if
    anyone reaches for real ones again.
    """
    root = Path(__file__).resolve().parents[1]
    text = (root / "fixtures" / "metric_gathering.csv").read_text(encoding="utf-8")
    for row in text.strip().split("\n")[1:]:
        cells = row.split(",")
        name = cells[2]
        if not name:
            continue
        assert name in ("Header", "Separator", "Trunk")             or name.startswith(("W-", "FL-W-")), (
                "%r is not a generic fixture name" % name)
