#!/bin/bash
# Runs the tests with the environment of the lab machine (see tests/README.md).
#   tests/run_tests.sh          fast tests only (parser, checks, computations: about a minute)
#   tests/run_tests.sh all      also the full report runs (marked slow)
#   tests/run_tests.sh all -k h2co -x     extra arguments are passed to pytest
cd "$(dirname "$0")/.." || exit 1
PY=${QUCHEMREPORT_PYTHON:-/applis/conda_envs/quchemreport/bin/python}
export PYTHONPATH=${CCLIB_PATH:-/applis/cclib/lib/python3.12/site-packages}${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
selection=(-m "not slow")
if [ "$1" = all ]; then selection=(); shift; fi
exec xvfb-run -a "$PY" -m pytest -p no:cacheprovider "${selection[@]}" "$@"
