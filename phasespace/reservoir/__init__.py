"""Subsurface models.

The engine's existing half covers the wellbore and the surface: VLP, IPR,
nodal analysis, gas networks, flow assurance. That half answers "what will this
well do at this pressure". It has no answer at all to "and what happens to the
pressure over the next three years", because nothing in it models a reservoir.

This package is the other half. It starts with the capacitance-resistance
model, which is the most PhaseSpace-shaped thing in reservoir engineering:
settled physics, a few hundred lines of numpy, every coefficient physically
meaningful, and no grid, geology or seismic required -- only the production and
injection rates an operator already has.
"""
