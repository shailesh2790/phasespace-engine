"""The only module in this repository that imports JutulDarcy.

It is a script, run as a subprocess, and it is deliberately not importable
into anything long-lived. The reason is measured rather than defensive: on
the first run here, SPE9 solved correctly and then the process died with

    ConcurrencyViolationError("Vector can not be resized concurrently")
    _pyjl_dealloc at PythonCall/src/JlWrap/C.jl:35

which is Julia's garbage collector finalising Python objects off-thread
during interpreter teardown. It took the run's output with it. A web
server or an agent runtime that imports `juliacall` inherits a GC that can
kill the interpreter at a moment of its own choosing, and no amount of
exception handling in Python catches `fatal: error thrown and no exception
handler available`.

So the contract here is: compute, write the answer to disk the moment it
exists, and let the process exit however it likes. A crash during teardown
then costs nothing, because the results are already durable and the
process was always going to die.

**Everything is written in SI, exactly as JutulDarcy returns it.** The
conversion to field units happens in the parent, in `jutul.py`, where it
is tested. Converting here would put arithmetic in the one file that
cannot be imported by the test suite without dragging in a 1.8 GB Julia
depot -- which is how a unit bug survives.

Run:

    python -m phasespace.reservoir._jutul_worker <deck.DATA> <out.npz> [meta.json]
"""
import json
import os
import sys
import time

# Single-threaded by default. Three environment settings were applied
# together when the teardown crash stopped reproducing, and which one
# mattered was never isolated, so all three are set rather than guessed at.
os.environ.setdefault("JULIA_NUM_THREADS", "1")
os.environ.setdefault("PYTHON_JULIACALL_THREADS", "1")
os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "no")

# Summary curves are per-timestep series; states are per-cell arrays. Both
# are kept, because a reservoir answer that cannot be mapped is half an
# answer, and the per-cell pressure is what a surrogate trains on.
STATE_KEYS = ("Pressure", "Saturations", "Rs")


def _meta(path, **fields):
    """Append facts to the metadata file as soon as they are known."""
    if not path:
        return
    existing = {}
    if os.path.exists(path):
        try:
            with open(path) as handle:
                existing = json.load(handle)
        except ValueError:
            existing = {}
    existing.update(fields)
    with open(path, "w") as handle:
        json.dump(existing, handle, indent=2, default=str)
        handle.flush()
        os.fsync(handle.fileno())


def run(deck: str, out: str, meta: str = "") -> int:
    started = time.time()
    _meta(meta, deck=deck, started=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                 time.gmtime()),
          units="SI, as returned by JutulDarcy -- the parent converts")

    if not os.path.isfile(deck):
        _meta(meta, error="no deck at %s" % deck)
        print("no deck at %s" % deck, file=sys.stderr)
        return 2

    import numpy as np

    t0 = time.time()
    import jutuldarcy as jd
    _meta(meta, import_seconds=round(time.time() - t0, 1))

    t0 = time.time()
    try:
        result = jd.simulate_data_file(deck, convert=True)
    except Exception as exc:                                # noqa: BLE001
        # A Julia error arrives as a juliacall.JuliaError whose message is
        # the whole Julia stacktrace. Recorded rather than re-raised, so
        # the parent gets a reason instead of an exit code.
        _meta(meta, error="%s: %s" % (type(exc).__name__, exc))
        print("simulation failed: %s" % exc, file=sys.stderr)
        return 1
    _meta(meta, solve_seconds=round(time.time() - t0, 1))

    if not isinstance(result, dict):
        _meta(meta, error="convert=True returned %s, not a dict"
                          % type(result).__name__)
        return 1

    arrays = {}
    days = np.asarray(result.get("DAYS", []), dtype=float)
    arrays["DAYS"] = days

    for name, curves in (("FIELD", result.get("FIELD") or {}),):
        for key, series in curves.items():
            values = np.asarray(series, dtype=float)
            if values.ndim == 1 and values.size == days.size:
                arrays["%s.%s" % (name, key)] = values

    wells = result.get("WELLS") or {}
    for well, curves in wells.items():
        for key, series in (curves or {}).items():
            values = np.asarray(series, dtype=float)
            if values.ndim == 1 and values.size == days.size:
                arrays["WELL.%s.%s" % (well, key)] = values

    states = result.get("STATES") or []
    for key in STATE_KEYS:
        stack = []
        for state in states:
            value = (state or {}).get(key)
            if value is None:
                stack = []
                break
            stack.append(np.asarray(value, dtype=float))
        if stack:
            try:
                arrays["STATE.%s" % key] = np.stack(stack)
            except ValueError:
                # Saturations can be per-phase and ragged across steps on
                # some models. Keeping the first step is better than
                # silently dropping the key, and the parent is told.
                arrays["STATE.%s" % key] = stack[0]
                _meta(meta, **{"state_%s_note" % key.lower():
                               "steps were not stackable; only the first "
                               "step was kept"})

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    np.savez_compressed(out, **arrays)

    _meta(meta,
          wells=sorted(wells),
          well_count=len(wells),
          field_keys=sorted(result.get("FIELD") or {}),
          well_keys=sorted((wells[sorted(wells)[0]] or {}))
                    if wells else [],
          state_keys=[k.split(".", 1)[1] for k in arrays
                      if k.startswith("STATE.")],
          steps=int(days.size),
          days_first=float(days[0]) if days.size else None,
          days_last=float(days[-1]) if days.size else None,
          cells=int(arrays["STATE.Pressure"].shape[-1])
                if "STATE.Pressure" in arrays else None,
          arrays=sorted(arrays),
          output=os.path.abspath(out),
          total_seconds=round(time.time() - started, 1),
          completed=True)
    print("wrote %s (%d arrays)" % (out, len(arrays)))
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    return run(argv[0], argv[1], argv[2] if len(argv) > 2 else "")


if __name__ == "__main__":
    raise SystemExit(main())
