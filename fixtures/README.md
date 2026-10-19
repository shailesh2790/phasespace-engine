# Fixtures

Every fixture here is synthetic. That is a rule, not a description of the
current contents, and `tests/test_nothing_private_ships.py` enforces it.

## metric_gathering.csv

A small metric-unit gathering network: wells feeding a header feeding an
offtake, with pressures in kg/cm², rates in m³/day and bean sizes in
millimetres.

It exists because unit handling is where this kind of software quietly
goes wrong, and the errors are large rather than subtle:

| conversion | the cost of getting it wrong |
|---|---|
| ksc gauge → psia | a factor of about 14, plus the 14.7 offset |
| m³/day → scf/day | a factor of 35 |
| bean mm → 64ths | an 8 mm bean is 20/64 in, not 8/64 — entering the millimetre figure quarters the bean |

`tests/test_metric_units.py` checks each of these against hand
calculation. None of them needs field data, which is the point: a unit
bug is detectable without a single measurement, and it is the most
common serious defect in production-engineering software.

## What replaced a real network, and why

This fixture stands in for one built from a private operator study,
which held real well names and measured rates. That fixture does not
ship and never will. What it demonstrated — that the engine reads a
metric field's data and converts it correctly — is reproduced here with
generated values and generic names.

What it genuinely proved and this cannot: that the conversions are right
*for a real asset's data as recorded*, including the local conventions
for how a bean size and a gauge pressure get written down. Closing that
gap needs an open metric-unit field dataset, which as far as we know does
not exist. If you know of one, please open an issue — it is listed in
[CONTRIBUTING.md](../CONTRIBUTING.md) among the contributions worth more
than code.

## Adding a fixture

Generate it. `phasespace/reservoir/synthetic.py` and
`phasespace/rodpump/synthetic.py` exist for this and produce data with
realistic structure. If a generated fixture cannot express what you need
to test, say so in the issue rather than reaching for a real one.
