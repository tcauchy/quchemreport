# Tests

```bash
tests/run_tests.sh          # fast tests (parser, quality checks, computations)
tests/run_tests.sh all      # + full report runs (marked slow, several minutes)
tests/run_tests.sh -k ecp   # any pytest argument
```

The script uses the conda environment and the cclib of the lab machine (`QUCHEMREPORT_PYTHON`,
`CCLIB_PATH` to change them), 4 OpenMP threads, and `xvfb-run` for the Mayavi images.

- Test logs: `tests/data` (gzipped, size budget 2 MB, see `tests/data/README.md`), read with
  `tests.datafiles.log("gaussian/h2co/OPT.log")`.
- A known defect gets its test first, marked `@pytest.mark.xfail(strict=True, reason=...)`; the
  fix removes the mark (strict: a test that starts passing without removing the mark fails).
- Full report runs are marked `@pytest.mark.slow`.
