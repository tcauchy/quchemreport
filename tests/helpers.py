"""Shared helpers of the tests: parsed models of a log, and full report runs."""
import contextlib
import glob
import io
import os
import subprocess
import sys
import tempfile

import yaml

from tests.datafiles import log

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_steps(relpath_or_path, solver="gaussian"):
    """Parsed models (LogData) of all the steps of a log of tests/data (or of any path)."""
    from quchemreport.parser_engine.scanlog import logfile_to_obj, split_logfile
    path = relpath_or_path if os.path.isabs(relpath_or_path) else log(relpath_or_path)
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        return [logfile_to_obj(step) for step in split_logfile(path, solver, tmp)]


def report_config(logfiles, solver="gaussian", **options):
    """example.yaml with the given log files and small, fast images.

    logfiles: list of dicts (path relative to tests/data or absolute, type, and optionally
    reference / excited_state_number). options: dotted keys of the YAML, e.g.
    {"output.include.wavefunction.MEP_maps": False}.
    """
    with open(os.path.join(ROOT, "example.yaml")) as fd:
        cfg = yaml.safe_load(fd)
    cfg["common_solver"] = solver
    cfg["logfiles"] = []
    for entry in logfiles:
        entry = dict(entry)
        if not os.path.isabs(entry["path"]):
            entry["path"] = log(entry["path"])
        cfg["logfiles"].append(entry)
    defaults = {
        "output.image_size": 300,
        "output.image_grid_reduction": 3,
        "output.image_grid_reduction_sensitive": 2,
        "output.include.wavefunction.MEP_maps": False,
        "output.include.excited_states.selection.mode": "all_states",
        "logging.level": "info",
    }
    for key, value in {**defaults, **options}.items():
        node = cfg
        *path, last = key.split(".")
        for part in path:
            node = node[part]
        node[last] = value
    return cfg


class Report:
    """Result of a full quchemreport run: exit code, console output, produced files, LaTeX."""

    def __init__(self, returncode, output, outdir):
        self.returncode, self.output = returncode, output
        self.files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(outdir, "**", "*"), recursive=True))
        tex = glob.glob(os.path.join(outdir, "**", "*.tex"), recursive=True)
        self.tex = open(tex[0]).read() if tex else ""

    def images(self, pattern):
        return [f for f in self.files if f.endswith(".png") and pattern in f and "_cam2" not in f]


def run_report(logfiles, solver="gaussian", **options):
    """Runs cli.py in a subprocess (sys.exit and Mayavi state stay isolated) in a temporary folder."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = report_config(logfiles, solver, **options)
        cfg["output"]["output_path"] = os.path.join(tmp, "output")
        with open(os.path.join(tmp, "report.yaml"), "w") as fd:
            yaml.safe_dump(cfg, fd, sort_keys=False)
        proc = subprocess.run([sys.executable, os.path.join(ROOT, "cli.py"), "-c", "report.yaml"], cwd=tmp,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              env={**os.environ, "PYTHONPATH": ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")})
        return Report(proc.returncode, proc.stdout, cfg["output"]["output_path"])


def without_lines(relpath, predicate, name):
    """Copy of a log of tests/data without the lines matching predicate (synthetic logs)."""
    target = os.path.join(os.path.dirname(log(relpath)), name)
    with open(log(relpath)) as src, open(target, "w") as dst:
        dst.writelines(line for line in src if not predicate(line))
    return target
