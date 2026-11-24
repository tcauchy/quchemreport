# Test data

Small Gaussian and ORCA outputs used by the tests. The logs are gzipped (`*.log.gz`,
`*.out.gz`) to keep the repository small; `tests/datafiles.py` uncompresses them on demand:

```python
from tests.datafiles import log, geometry
path = log("gaussian/h2co/OPT.log")      # uncompressed copy in a temporary folder
Z, xyz = geometry("phenanthrene")         # geometries/phenanthrene.xyz
```

**Size budget: 2 MB for the whole folder** (1.6 MB in November 2026). Use the smallest
molecule and basis set that shows the behaviour to test, and avoid `pop=full` / MO printing on
large systems.

## Contents

| Folder | Program | What it covers |
|---|---|---|
| `gaussian/h2o` | Gaussian 09 | H2O examples of the project (OPT, FREQ, TD singlets/triplets, SP±, unrestricted, second basis set, no `pop=full`, ES optimisation, truncated TD) |
| `gaussian/h2co` | Gaussian 09 D.01 | formaldehyde, PBE0/6-31G(d): OPT, FREQ, TD, TD triplets, SP+ / SP−, S1 and T1 optimisations (emission) |
| `gaussian/hcn_ts` | Gaussian 09 D.01 | HCN ⇄ HNC transition state, OPT=TS + FREQ (one imaginary frequency) |
| `gaussian/ch3i` | Gaussian 09 D.01 | CH3I with the LANL2DZ ECP on iodine (46 core electrons) |
| `gaussian/ch2bri` | Gaussian 09 D.01 | CH2BrI with ECPs, with and without `IOp(3/33=1)` (two layouts of the pseudopotential table) |
| `gaussian/mixed` | Gaussian 09 D.01 | 6-31G(2df) with Cartesian d / spherical f (6D 7F, default) and the reverse (5D 10F) |
| `gaussian/small` | Gaussian 09 D.01 | He, Ne, H2 (s functions only, single-sign orbitals), planar NH3 (saddle point), ROHF O2 triplet |
| `gaussian/ch3i_mep` | Gaussian 09 D.01 | CH3I single points (nosymm) with LANL2DZ (ECP) and DGDZVP (all-electron iodine, 1s energy printed `**********`) |
| `gaussian/ch3x_mep` | Gaussian 09 D.01 | CH3Cl and CH3Br single points (nosymm), PBE0/6-31G(d), all-electron halogens (MEP core correction) |
| `orca5/h2o_td` | ORCA 5.0.4 | H2O TD-DFT (frozen: ORCA 5 is no longer installed) |
| `orca6/h2co` | ORCA 6.1 | formaldehyde, PBE0/def2-SVP: OPT+FREQ, TD, TD singlets+triplets, TD with SOC, SP+ / SP−, S1 and T1 optimisations |
| `orca6/nh3_plan` | ORCA 6.1 | planar NH3, OPT+FREQ (one imaginary frequency) |
| `orca6/maxiter` | ORCA 6.1 | H2O optimisation stopped at MaxIter (not converged, but "terminated normally") |
| `orca6/functionals` | ORCA 6.1 | H2O single points: PBE0, B3LYP-D3BJ, B3LYP-D4, ωB97X-D3, ωB97X-V, CAM-B3LYP, B2PLYP, TPSSh |
| `orca6/ch3i` | ORCA 6.1 | CH3I with the def2 ECP on iodine (28 core electrons) |
| `geometries` | — | coordinates only (phenanthrene), for tests that do not need a log |
| `mep_reference` | Gaussian 09 D.01 | exact MEP (`cubegen 8 potential=SCF`, fine grid) at 400 random points of the 0.002 a.u. iso-density surface, for `gaussian/h2co/OPT.log`, the two `gaussian/ch3i_mep` logs and the two `gaussian/ch3x_mep` logs (checkpoints of these jobs); `make_mep_reference.py JOB.fchk OUT.txt label` makes them (g09 environment needed) |

All ORCA inputs print the basis set and the MO coefficients (`Print[P_Basis] 2`, `Print[P_MOs] 1`),
which quchemreport needs for the grids (MO, MEP, EDD, NTO).

## Regenerating the logs

Inputs (`.inp`, ORCA `.xyz`) are stored next to the logs, with a `jobs.txt` giving the order of
the jobs (and, for Gaussian, the job whose checkpoint is reused, e.g. `FREQ OPT`):

```bash
tests/data/regenerate.sh gaussian/h2co orca6/functionals
```

The script runs the jobs in a temporary folder with Gaussian 09 (`g09root`, default `/applis`)
and ORCA (`ORCA_BIN`, default `/applis/orca6.1/bin/orca`) and writes back only the gzipped logs.
`gaussian/h2o` (from `examples/`) and `orca5/` have no `jobs.txt` and are not regenerated.
New logs differ from the stored ones only by dates, process numbers and timings.
