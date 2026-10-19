"""Nothing operator-derived may exist in this repository.

This is the public repository, so the threat model is wider than a
deployment's. A deployed application leaks only what it serves; a
published repository leaks **every byte of every tracked file**, and a
filename alone can disclose a field name without a single line of data
behind it.

So this guard scans the whole tracked tree rather than one directory,
checks paths as well as contents, and refuses measurement-data formats
outright. The ancestor of this test scanned only the directory that got
deployed -- correct for that threat model, and the reason several
operator-derived files once sat tracked without ever tripping it.

**The checks here are deliberately generic.** A guard that listed real
operator names, field names and well identifiers would itself publish
them: anyone reading it would learn exactly whose data was withheld and
from where. That list belongs in a private repository, next to the data
it protects. What is enforceable in the open is structural -- no
measurement-data formats, nothing that labels itself confidential, no
credentials -- and those catch the realistic accident, which is a
spreadsheet committed by reflex rather than a deliberate leak.

Contributors who maintain a private fork with real data should add their
own identifier list there. `PRIVATE_MARKERS_FILE` below is read if
present and is itself untracked, so a local list can extend this guard
without ever being published.
"""
import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# An untracked, gitignored file of extra regexes, one per line, for anyone
# working against real data in a private fork. Absent in a clean checkout.
PRIVATE_MARKERS_FILE = ROOT / ".private-markers"

# Self-labelled confidential material. Anything carrying its own warning
# has already told us it does not belong in a public tree. Nothing here
# names a real operator, field or well -- see the module docstring.
MARKERS = [
    r"operator[- ]derived",
    r"not for publication",
    r"\bconfidential\b",
    r"\bproprietary data\b",
    r"\bdo not distribute\b",
    r"\bunder NDA\b",
    # Credentials, which are the other thing that escapes by reflex.
    r"AKIA[0-9A-Z]{16}",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
]


def _markers():
    extra = []
    try:
        extra = [line.strip() for line
                 in PRIVATE_MARKERS_FILE.read_text(encoding="utf-8").splitlines()
                 if line.strip() and not line.startswith("#")]
    except OSError:
        pass
    return MARKERS + extra


PATTERN = re.compile("|".join(_markers()), re.IGNORECASE)

# Formats that carry measurements. The engine runs on numpy and the
# standard library and needs none of them, so any arrival is a dataset
# nobody meant to commit. `.json` is omitted because configuration
# legitimately uses it; it is scanned as text instead.
DATA_SUFFIXES = {".xlsx", ".xls", ".xlsm", ".csv", ".npz", ".npy", ".h5",
                 ".hdf5", ".parquet", ".pkl", ".pickle", ".sqlite", ".db"}

# Fixtures are the one place a .csv is legitimate, and they are synthetic
# by rule -- see CONTRIBUTING.md.
DATA_ALLOWED = {"fixtures"}

TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".csv", ".yml", ".yaml",
                 ".toml", ".cfg", ".ini", ".ts", ".tsx", ".js", ".html",
                 ".css"}

# This file necessarily contains the patterns it searches for.
SELF = Path(__file__).name


def tracked_files():
    """Every file git would publish, as (repo-relative name, path)."""
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT,
                         capture_output=True, text=True, check=True)
    for name in out.stdout.split("\0"):
        if name:
            yield name, ROOT / name


def _git_available():
    try:
        subprocess.run(["git", "rev-parse", "--git-dir"], cwd=ROOT,
                       capture_output=True, check=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


needs_git = pytest.mark.skipif(not _git_available(),
                               reason="not a git checkout")


# ------------------------------------------------- the guard's own health --


def test_the_pattern_would_catch_a_real_leak():
    """A guard is only worth what it fires on, so it is tested against
    specimens rather than trusted by inspection."""
    assert PATTERN.search("# PRIVATE -- operator-derived. Not for publication.")
    assert PATTERN.search("Confidential feasibility study")
    assert PATTERN.search("shared under NDA, do not redistribute")
    assert PATTERN.search("AKIAIOSFODNN7EXAMPLE")
    # And not so loose that it fires on ordinary engineering prose.
    assert not PATTERN.search("the operator reported a rate of 1.5 MMscfd")
    assert not PATTERN.search("a proprietary simulator was used")


@needs_git
def test_the_scan_covers_something():
    """A scan that silently matches nothing passes for the wrong reason."""
    files = list(tracked_files())
    assert len(files) > 40, (
        "only %d tracked files found -- is this a git checkout?" % len(files))
    assert any(name.endswith(".py") for name, _ in files)


# ------------------------------------------------------------ the checks --


@needs_git
def test_nothing_confidential_in_any_tracked_path():
    """Filenames first: a name like `wellsdata<field><field>.pdf` discloses
    two field names before anyone opens it."""
    offenders = [name for name, _ in tracked_files()
                 if PATTERN.search(name)]
    assert not offenders, (
        "confidential markers in tracked filenames:\n  "
        + "\n  ".join(offenders))


@needs_git
def test_nothing_confidential_in_any_tracked_file():
    offenders = []
    for name, path in tracked_files():
        if Path(name).name == SELF:
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for match in PATTERN.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            offenders.append("%s:%d  %r" % (name, line, match.group(0)))

    assert not offenders, (
        "confidential markers in tracked files:\n  "
        + "\n  ".join(offenders[:20])
        + "\n\nThis repository is public. See CONTRIBUTING.md.")


@needs_git
def test_no_measurement_data_formats_are_tracked():
    """The realistic accident: a spreadsheet committed by reflex."""
    offenders = [
        name for name, path in tracked_files()
        if path.suffix.lower() in DATA_SUFFIXES
        and not set(Path(name).parts) & DATA_ALLOWED
    ]
    assert not offenders, (
        "measurement-data files are tracked:\n  "
        + "\n  ".join(offenders)
        + "\n\nGenerate fixtures instead -- see "
          "phasespace/reservoir/synthetic.py.")


@needs_git
def test_the_private_marker_file_is_never_tracked():
    """The local extension list must stay local. If it were committed, the
    identifiers it protects would be published by the very mechanism meant
    to keep them out."""
    tracked = {name for name, _ in tracked_files()}
    assert PRIVATE_MARKERS_FILE.name not in tracked, (
        "%s is tracked -- it holds the identifiers this guard exists to "
        "keep unpublished" % PRIVATE_MARKERS_FILE.name)
