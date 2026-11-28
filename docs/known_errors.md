# Known errors

Errors found in QuChemReport and not fixed yet. `[refactor]` changes (see `CONTRIBUTING.md`)
reproduce them exactly; each one is fixed by its own `[science]` change, after the maintainer's approval, and then
removed from this list (with the commit that fixed it).

Source: full review of 2026-11-28 (findings 1.x). File:line refer to commit 08f9220.

| # | Error | Where | Evidence | Reports affected |
|---|---|---|---|---|
| 1.1 | Total energy is the SCF part only for double hybrids, MP2, coupled cluster; A, I and the conceptual DFT indices follow | `scanlog.py:567-574`, used by `generate_report.py:1000`, `1562-1565`, `calc_orb.py:915`, `971` | ORCA B2PLYP `tests/data/orca6/functionals/SP_B2PLYP.out`: −76.22474 Eh instead of FINAL SINGLE POINT ENERGY −76.28924 Eh (64.5 mEh) | only these methods |
| 1.2 | Energy of an optimized excited state is the ground state at the excited geometry; for `freq_es`, E + ZPE mixes ground-state E and excited-state ZPE | `scanlog.py:567-574`, `generate_report.py:1562-1565` | acetone OPT_ES1: −192.89516 (SCF Done) instead of E(TD) −192.78044 | acetone OPT_ES1 / OPT_ET1 rows; label to decide |
| 1.3 | Transition table printed with no rows when the grids are missing (TD without amplitudes, no MO coefficients, ORCA full TD-DFT) | `generate_report.py:2090-2099`, `2159-2166` | H2CO report, TD without amplitudes | none of the reference reports |
| 1.4 | Crash of the conceptual DFT part if Hirshfeld charges exist on one side only | `calc_orb.py:921-931`, `977-987` | `np.subtract(array, [])` raises ValueError | none of the reference reports |
| 1.5 | Orientation of the single points used for the Fukui functions never checked: a rotated SP gives wrong f+, f−, dual descriptor silently | `compute.py:990-1019`, `calc_orb.py:789-794` | code reading | none of the reference reports |
| 1.8 | Condensed Fukui tables missing when `CDFT_indices: false` | `compute_flags.py:362`, `generate_report.py:1917-1919` | code reading | none of the reference reports |
| OB1 | A set of logs including an ion of charge ±3 or more stops with "too many different formulas": the charge signs are removed by counting the digits of the charge (written for `+2`), but the formula carries one sign per unit (`+++`) | `conformity.py:150-170` (`formulas_nocharges`) | charge 3: `C3H6O+++` cleaned to `C3H6O+`, not `C3H6O` (found 28/11/2026 while replacing Open Babel; signs checked for charges −2 to +3) | none of the reference reports; any set with a trication or trianion |
