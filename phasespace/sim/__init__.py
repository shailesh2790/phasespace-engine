"""PhaseSpace simulation engine.

The deterministic core: a component-based steady-state network solver that
is physics-agnostic. `phasespace.physics.network` hard-coded the Weymouth
equation into the Newton-Raphson assembly, which meant gas hydraulics was
the only thing it could ever solve. Here the solver knows nothing about
any particular flow correlation -- it only asks each branch for
(q, dq/dp_up, dq/dp_dn) -- so gas pipes, liquid pipes, compressors, pumps,
valves and regulators all assemble into the same Jacobian.
"""
