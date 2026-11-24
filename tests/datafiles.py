"""Access to the test data of tests/data (see tests/data/README.md).

The logs are stored gzipped to keep the repository small; log() returns the path of an
uncompressed copy (same file name) in a temporary folder removed at the end of the run.
"""
import atexit
import gzip
import os
import shutil
import tempfile

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_tmp = None


def log(relpath):
    """Path of an uncompressed copy of tests/data/<relpath>.gz, e.g. log("gaussian/h2co/OPT.log")."""
    global _tmp
    if _tmp is None:
        _tmp = tempfile.mkdtemp(prefix="quchemreport_tests_")
        atexit.register(shutil.rmtree, _tmp, True)
    target = os.path.join(_tmp, relpath)
    if not os.path.isfile(target):
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with gzip.open(os.path.join(DATA, relpath + ".gz"), "rb") as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return target


def geometry(name):
    """(atomic numbers, coordinates in Angstrom) of tests/data/geometries/<name>.xyz."""
    from cclib.parser.utils import PeriodicTable
    table = PeriodicTable()
    with open(os.path.join(DATA, "geometries", name + ".xyz")) as fd:
        lines = fd.read().splitlines()
    atoms = [line.split() for line in lines[2:2 + int(lines[0])]]
    Z = np.array([table.number[a[0]] for a in atoms])
    return Z, np.array([[float(v) for v in a[1:4]] for a in atoms])
