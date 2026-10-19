# Contributing

This project is for people who have run wells, read the papers, or both.
Domain knowledge is worth more here than Python fluency, and a well-argued
issue that says "Hagedorn-Brown will mislead you for condensate above
this GLR, here is the paper" is a real contribution even with no code
attached.

## The most valuable contribution is data

Look at the *not validated* rows in the [README](README.md) table. Each is
a capability this engine has and cannot prove. Closing one is worth more
than any feature:

* **Gas-network hydraulics.** Needs a topology paired with measured node
  pressures. GasLib carries nominations rather than measurements, so it
  cannot do this. An operator's SCADA extract could.
* **The nodal intersection.** Needs a well where inflow and outflow were
  measured together — simultaneous downhole pressure, wellhead pressure
  and rate. Equinor's Volve dataset has exactly this shape.
* **Rod pump diagnosis.** No open labelled dynamometer set exists
  anywhere in the world. A few hundred cards with an engineer's
  classification would be the first.
* **Gas lift.** No open lift-test data: injection rate against production
  rate on a real well.
* **Pressure transients.** Pressure against time at any usable sampling
  rate. Most archives reduce a gauge record to three numbers before
  anyone sees it, which is why this is not built yet.

If you can release such data, or point at a public source that was
missed, open an issue. Anonymised is fine; so is "I can share this under
these conditions".

## Good first issues, in physics

Marked `physics` in the tracker. A sample of the kind of thing that
belongs there:

* **NORSOK M-506 alongside de Waard & Milliams.** The two disagree, and
  reporting both with the disagreement visible is more honest than
  picking one. `phasespace/sim/integrity.py`
* **The Jones IPR**, which adds a turbulence term Vogel and Fetkovich
  lack. `phasespace/sim/wells.py` already has Vogel, Fetkovich,
  productivity-index, composite and gas back-pressure forms, so this
  slots in beside them.
* **The Gray correlation**, which is absent. It is the usual choice for
  gas condensate, where the two correlations here were developed on
  something else -- Hagedorn-Brown on vertical oil wells and Beggs-Brill
  on an inclined air-water loop.
* **Flow-regime transition criteria** other than Beggs–Brill's, for
  example Taitel–Dukler.
* **Temperature prediction.** The thermal model is given measured
  temperature rather than asked to predict it, which is a real gap:
  see `phasespace/sim/thermal.py`.

## House rules

These are unusual, and they are the point of the project.

**1. State what a method cannot do, next to what it can.** Every module
docstring here names the assumptions that break it. A contribution that
adds a correlation without naming its validity range will be sent back,
not because of style but because an unlabelled correlation is how people
get hurt.

**2. Measure, do not assert.** If you claim a method is more accurate or
faster, include the comparison that shows it. Against a real alternative,
not against nothing. The three declined ML layers in the README are there
because they were measured and lost; that is a normal outcome.

**3. Cite the source.** Author, year, and what the correlation was fitted
on. "Beggs & Brill 1973, fitted on a 1 in and 1.5 in inclined air-water
loop" tells a reader more than any amount of code commentary.

**4. Tests sweep, they do not sample.** A correlation that is right at one
pressure and wrong at another passes a single-point test. Several tests
here deliberately sweep a range for this reason.

**5. No operator data, ever.** Not in code, not in fixtures, not in test
names, not in comments. `tests/test_nothing_private_ships.py` enforces
this against the whole tracked tree and will fail the build. If you need a
realistic fixture, generate one — `phasespace/reservoir/synthetic.py` and
`phasespace/rodpump/synthetic.py` exist for this.

## Mechanics

```bash
pip install -r requirements.txt
python -m pytest tests/ -q
```

Contributions are accepted under Apache 2.0 via the
[Developer Certificate of Origin](https://developercertificate.org/).
Sign off your commits:

```bash
git commit -s -m "Add Taitel-Dukler transition criteria"
```

There is no CLA to sign. You keep your copyright; the DCO is you
confirming you have the right to submit the work.

## Scope

This repository is the physics kernel. The commercial product built on it
(a workspace UI, trained surrogates, a data factory, operator validation)
is a separate closed codebase, and the boundary is stated in the README so
nobody contributes under a misunderstanding. Pull requests that move
physics *into* the open kernel are welcome. The physics stays open.
