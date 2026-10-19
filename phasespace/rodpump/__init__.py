"""Sucker-rod pump diagnostics: surface dynamometer card to downhole card.

The commercial problem this addresses: roughly a million rod-pumped wells
worldwide each carry a dynamometer, and reading those cards is how an
operator knows whether a pump is full, pounding fluid, gassy, leaking a
valve, or tagging bottom. The calculation that turns a *surface* card into a
*downhole* card has been public since Gibbs (1963), yet it is sold as a
per-well subscription.

The structure here is the same one that works elsewhere in this codebase:

  wave.py       the deterministic core -- a damped wave equation solved down
                the rod string. This is the audit anchor; everything above it
                is only as good as this is.
  rodstring.py  tapered string geometry and material properties
  cards.py      card representation and the measurements read off one
  synthetic.py  physics-generated cards with known faults, which is what
                makes a labelled training set possible without owning a
                fleet of wells
  diagnosis.py  what is wrong with the pump

Units are US oilfield throughout, because that is what a dynamometer reports
and what a rod-string tally is written in: position in inches, load in pounds
force, depth in feet, time in seconds.
"""
