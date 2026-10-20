# PhaseSpace Engine

A production-engineering physics kernel you can read, check against the
papers it cites, and reproduce.

Nodal analysis, multiphase pressure traverses, gas-network hydraulics,
decline and material balance, gas lift, rod pumping, flow assurance and
well integrity. The simulation core runs on numpy and the standard
library alone, every correlation is named and attributed, and every
accuracy claim below is measured rather than asserted.

```python
from phasespace.sim.wells import (Tubing, ProductionStream,
                                  pressure_traverse, HAGEDORN_BROWN)
from phasespace.sim.fluids import GasProperties

tubing = Tubing(depth_ft=8000.0, inner_diameter_in=2.441)
stream = ProductionStream(gas_rate_scf_d=1.5e6,
                          gas=GasProperties(gravity=0.60))

result = pressure_traverse(wellhead_pressure_psia=400.0, stream=stream,
                           tubing=tubing, correlation=HAGEDORN_BROWN)

print(result.bottomhole_pressure_psia)   # 510.8
print(result.elevation_fraction)         # 0.76 -- gravity's share
print(result.friction_fraction)          # 0.24
```

That last pair is the point. The answer arrives with its own decomposition,
so you can see *why* the column holds the pressure it does, and the full
`result.pressures_psia`, `result.holdups` and `result.regimes` arrays are
there at every depth rather than hidden behind a single number.

---

## Why this exists

Production engineers run simulators they cannot inspect. The SPE auditing
standard already warns against relying on "proprietary or commercial
software without a full understanding of the internal mathematical
algorithms and correlations" — but that understanding is exactly what a
closed simulator withholds. When a number is wrong, you find out from the
well.

This kernel takes the opposite position. The equations are in the open,
each one named and attributed. The accuracy claims are measured against
data rather than asserted, and stated with their limits attached. Where a
method does not work, that is written down next to where it does.

## Reproduce a result before you trust anything here

The quickest way to judge a simulator is to make it reproduce something
you can check. Two commands, no account, no licence key:

```bash
pip install -r requirements.txt
python validation/fetch_3w.py       # Petrobras 3W 2.0.0, CC BY 4.0
python validation/hydrate_3w.py     # reproduces the result below
```

On 594 real instances from offshore production lines, the hydrate
detector flags **7 of 7** events where the instrumentation was in range,
against **1 false alarm in 537** normal-operation instances. That single
false alarm is a thermometer frozen at −13 °F; it is reported as a false
alarm rather than quietly filtered out.

And a limit, stated in the same breath: **it detects, it does not
predict.** The margin is flat in the hour before onset — identical at
sixty minutes and at one minute on one well. Hammerschmidt gives a
necessary condition, not a leading indicator, and any lead time quoted
from it would be the length of the window someone chose.

## What is validated, and what is not

Used in the ASME V&V 20 sense, because most of what this industry calls
validation is code-to-code comparison, which is verification:

* **Verification** — the code solves the equations it claims to. Checked
  against closed forms, another implementation, or a correlation's own
  published values. No measurement involved.
* **Validation** — the equations describe reality. Requires measuring the
  physical thing.

| Capability | Status | Evidence |
|---|---|---|
| Hydrate conditions | **validated** | 7/7 events, 1 false alarm in 537, on the public 3W benchmark — reproducible above |
| Tubing correlation (VLP) | **validated** | 2.9% median absolute bottomhole error on 150 measured gradient surveys, 25.1 psi rms along the string |
| Warm-started network solve | **validated** | Identical root, 44–50% fewer Newton iterations on hub topologies |
| Z-factor, PVT, numerics | verified | Table within 1.8×10⁻⁵ of Dranchuk–Abou-Kassem; ES-MDA lands on the closed-form Gaussian posterior |
| CO₂ corrosion | correlation only | de Waard & Milliams reproduces its own nomogram to 2%. Never checked against a caliper log or a coupon |
| Rod pump / Gibbs | verified only | No open labelled dynamometer set exists anywhere. Checked against the analytical formulation and API 11L |
| Nodal *intersection* | **not validated** | The traverse is validated — given a rate, the engine reproduces measured pressure at every depth. Predicting the **rate** needs a well where inflow and outflow were measured together |
| Gas network hydraulics | **not validated** | No public source pairs a topology with measured node pressures. GasLib carries nominations, not measurements |
| Gas lift optimisation | **not validated** | No open lift-test data |
| Pressure-transient analysis | not built | Needs pressure against time; see the issues list |

The gradient-survey validation ran against a private operator archive.
The results are published; the data cannot be. Everything marked
*reproducible* runs from a clean machine with no agreement from anyone.

**Every "not validated" row in that table is an open invitation.** If you
have data that would close one, that is the single most valuable
contribution this project can receive — more than code.

## Three machine-learning layers that were built, measured and declined

Listed because a vendor's deck shows what its ML added and never what it
did not.

| Layer | Measured against | Result |
|---|---|---|
| Hydrate predictor | Tuned physics, 594 real instances | Identical at its best operating point, worse elsewhere. The information is not in the channels |
| Grey-box residual | Two fitted constants, held-out wells | 94.1 psi against 87.9. Its apparent win on a weaker correlation was recovering that correlation's own error |
| Larger surrogate | A smaller one, held-out cases | Tail error 3.94% → 9.97% → 25.91% as capacity grew |

The one learned component that did ship is a warm start — an *initial
guess* for Newton, not a prediction. It cannot change the answer, only
how fast it is reached, and the solver verifies that it didn't.

## What the engine knows

| Area | Methods |
|---|---|
| Multiphase flow | Beggs–Brill, Hagedorn–Brown; flow-regime maps |
| Gas flow | Weymouth, Panhandle A/B, AGA; Dranchuk–Abou-Kassem Z with Sutton pseudo-criticals and Wichert–Aziz sour correction |
| PVT | Standing, Vazquez–Beggs, Lee–Gonzalez–Eakin viscosity, real-gas pseudopressure |
| Nodal analysis | IPR/VLP intersection with high-side bracketing; Vogel, Fetkovich, Rawlins & Schellhardt, Houpeurt |
| Networks | Newton–Raphson with analytic Jacobian, sparse or dense; warm start |
| Reservoir | Arps and Duong decline, p/z and Havlena–Odeh material balance, capacitance–resistance models |
| Artificial lift | Continuous and intermittent gas lift, Asheim F1/F2 stability, valve mechanics, Gibbs damped rod-string wave equation |
| Flow assurance | Hammerschmidt hydrates, Turner/Coleman critical velocity, erosional velocity (API RP 14E) |
| Integrity | de Waard & Milliams CO₂ corrosion with siderite film and oil-wetting flags |
| Uncertainty | Split conformal prediction, CQR, ES-MDA history matching |

Dependencies: `phasespace/sim` — the whole of the above except the
reservoir models — runs on **numpy and the standard library alone**, and
CI enforces that on every commit. It is a real constraint rather than a
preference: it is what lets the engine deploy into a serverless function
with a hard size limit that scipy does not fit inside. Where scipy is
present it is used for sparse linear algebra and bounded least squares,
and the engine falls back to dense numpy when it is not.

`phasespace/reservoir` (CRM and the synthetic generators) does require
scipy, and is not under that constraint.

## Use it from an agent

The engine ships as an [MCP](https://modelcontextprotocol.io) server, so a
language model can call the physics instead of imitating it:

```bash
python -m phasespace.mcp --list      # the catalogue
python -m phasespace.mcp             # serve on stdio
```

```json
{
  "mcpServers": {
    "phasespace": {
      "command": "python",
      "args": ["-m", "phasespace.mcp"],
      "cwd": "/path/to/phasespace-engine"
    }
  }
}
```

The rule an engineering agent has to obey is that it never computes physics
in its own head — every number it shows comes from a tool call. A prompt
cannot enforce that: a model told not to do arithmetic does arithmetic
anyway, and the failure is silent, because a fabricated number and a
computed one look identical in a transcript.

So it is enforced mechanically. Every answer carries a provenance block
with the engine fingerprint, and the bundle is stable sorted JSON, so the
same question twice is byte-identical and the digest means something. A
number without one did not come from the server.

Every answer also states its own limits in the same payload as the number,
rather than as a footnote that gets dropped on the way to a user. Ask for a
nodal operating point and it volunteers that the intersection is not
validated. Ask for a traverse on a dry gas well and it tells you the
default correlation is the wrong one, with both error figures.

And a refusal is a result rather than an exception, because an agent can
read a sentence and act on it but cannot reason about a traceback — what it
does with one is retry, or fill the gap itself. Ask about scale and it
names the six water-chemistry values it would need. Feed it sour gas and it
cites NACE MR0175 instead of running a CO₂ correlation. Hand it a rising
production history and it refuses rather than forcing an Arps fit.

Seven tools, implemented against the standard library with no SDK, so it
runs from a stock Python on an air-gapped box. See
[phasespace/mcp/README.md](phasespace/mcp/README.md).


## Provenance

Every answer can carry the SHA-256 fingerprint of the engine source that
produced it, so a figure quoted eighteen months ago can be re-derived
from the build that made it:

```python
from phasespace.sim.provenance import engine_fingerprint, verify_fingerprint
engine_fingerprint()     # '61c9045ea5ee'
verify_fingerprint()     # (True, stamped, computed)
```

Determinism is demonstrated rather than asserted: a cross-section of the
engine runs in separate processes under different hash seeds and every
float is compared bit for bit.

That fingerprint is also the check on everything claimed above. It is
computed from the source in *this* repository, and it is the same
fingerprint the commercial build reports — so the engine that produced
the validation figures in the table is the engine you just cloned, not a
reduced version of it. If the two ever diverge, the hash says so.

## Measuring what a recommendation delivered

An optimiser that says "+10 bbl/d" and is never checked is a guess with a
decimal point. `phasespace/sim/actions.py` carries a recommendation from
proposal through a named engineer's approval to a measurement of what it
actually produced.

The measurement is the hard part, because production moves for a dozen
reasons at once and the convenient ones are easy to claim:

| how the number goes wrong | what happens |
|---|---|
| unsubtracted decline | the baseline is carried forward along the well's own fitted curve |
| two changes on one well | **refused** — no split from one well's history is defensible |
| uptime moved | **refused** above 15% drift |
| window under 7 days | **refused** |

The decline correction does not bias the way people assume. On a well
declining 30%/yr, a month-on-month rise of 8.0 bbl/d is a gain of **10.4**
against where the well was actually heading — so failing to correct
*understates* the result, because the action is not credited for the fall
it prevented. On a rising well it overstates. Both directions are in the
docstring with the worked numbers.

And the ceiling on all of it: this measures association. The well did
better after the change, which is not the same as better *because* of it.
Only a control — a comparable well left alone — supports the stronger
word, and every measurement carries that sentence so it cannot be dropped
on the way to a slide.

The scorecard is deliberately unflattering. Losses count in the total
rather than being dropped as outliers, refusals stay visible so an
unmeasurable action cannot quietly flatter the ratio by selection, and
below five measured actions it reports **no field ratio at all** — a ratio
from two actions is a story about two wells, and it will be quoted as a
track record.


## Keeping the evidence

A provenance bundle that exists only in one reply is evidence nobody can
retrieve. Point the server at a file and every call is recorded, refusals
included:

```bash
PHASESPACE_LEDGER=evidence.jsonl python -m phasespace.mcp
PHASESPACE_LEDGER=evidence.jsonl python -m phasespace.mcp --verify
```

```
2 records, chain intact
head:   70f468b66b04477ac7f92fb788b9aa7daaa5352bf3bff62abddf165d39adc9b0
engines: 61c9045ea5ee
```

One JSON object per line, each carrying the digest of the record before it.
`verify` detects an edited value, a deleted record, a reordering, or a
removed beginning, and names the first record where the file stops being
self-consistent.

**It is tamper-evident, not tamper-proof**, and the distinction is in the
tests rather than the small print: anyone who can write the file can
rewrite it whole and recompute the chain, and the result verifies cleanly.
`test_a_whole_rewrite_verifies_cleanly_and_that_is_the_known_limit`
demonstrates it. What the chain buys is raising the cost of a quiet edit
from changing one number to rewriting every record after it, and making a
careless edit certain to be caught. Publishing the head digest somewhere
the writer cannot reach is what closes the rest.

It is not a signature either, and the timestamps are whatever the machine
believed.


## Running the tests

```bash
python -m pytest tests/ -q
```

The suite is slow on purpose in places — several tests sweep parameter
ranges rather than checking one point, because a correlation that is
right at one pressure and wrong at another passes a single-point test.

## Contributing

Start with [CONTRIBUTING.md](CONTRIBUTING.md). Short version: the issues
labelled `physics` are real engineering problems, not plumbing, and the
*not validated* rows above are worth more than any of them.

Contributions are accepted under the Apache 2.0 licence via the
[Developer Certificate of Origin](https://developercertificate.org/) —
sign off with `git commit -s`. There is no CLA.

## Licence and the commercial product

Apache 2.0. Use it commercially, fork it, embed it — the patent grant
comes with it.

This kernel is developed alongside a commercial product (PhaseSpace
Studio) that adds a workspace UI, trained surrogate models, a data
factory, multi-well audit trails and operator validation. That is a
separate closed codebase. The split is stated plainly so nobody
contributes under a misunderstanding: **the physics is open and stays
open.** Correlations from the literature were never going to be anyone's
moat, and a simulator nobody can check is the problem this project
exists to address.
