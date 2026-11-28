# QuChemReport

QuChemReport turns the output files of quantum chemistry calculations on one molecule into a
PDF report. From Gaussian or ORCA calculations (optimization, frequencies, single points of the
ions, excited states, optimized excited states), it checks that they describe the same molecule
at the same level of theory, then reports:

- computational details, energies and thermochemistry, IR vibrations;
- molecular orbitals, molecular electrostatic potential on the molecular surface with its
  minima and maxima, population analysis;
- conceptual DFT indices and Fukui functions;
- excited states: transitions, absorption, CD, emission and CPL spectra, electron density
  differences, charge-transfer indices, natural transition orbitals;
- with ORCA, spin-orbit couplings between singlets and triplets, Jablonski-Perrin diagram and
  phosphorescence (radiative rates and lifetime of the T1 sublevels).

## Quick start (MOLTECH-Anjou cluster)

```bash
conda activate /applis/conda_envs/quchemreport
cp /applis/quchemreport/example.yaml report.yaml     # then edit the log files and options
xvfb-run -a python /applis/quchemreport/cli.py -c report.yaml
```

The report is written in `<output_path>/<molecule_designated_name>/`.

## Documentation

See [docs/](docs/README.md):

- [Preparing the calculations](docs/calculations.md) (Gaussian and ORCA keywords, example
  inputs in `examples/acetone`);
- [Running QuChemReport](docs/running.md);
- [The YAML configuration file](docs/configuration.md);
- [The report](docs/report.md);
- [Troubleshooting](docs/troubleshooting.md);
- [Architecture](docs/architecture.md), for developers. Tests: `tests/README.md`.

## Requirements

Python 3.12 with NumPy, SciPy, pydantic, PyYAML, cclib, RDKit, QMCkl, Mayavi/VTK, matplotlib,
pylatex, and a LaTeX distribution (`pdflatex`). orbkit is only
needed to compare with the former grid engine.

cclib: use its development version (GitHub master); the release 1.8.1 cannot read ORCA 6 outputs.
Some cclib changes made for QuChemReport are not merged yet. Without them the reports are still
made (tested with cclib master on the whole test suite), with these losses:

| cclib change | Without it |
|---|---|
| core electrons with ECPs (PR #1972) | no MEP map for molecules with effective core potentials (skipped with a message) |
| dispersion model (PR #1973) | the dispersion is not compared: B3LYP and B3LYP-D3(BJ) logs are not told apart |
| Gaussian de-excitation amplitudes Y | transition densities from the X amplitudes only (marked approximate) |
| ORCA transition dipoles | ORCA: transition dipoles computed from the amplitudes |
| ORCA velocity and magnetic transition dipoles | ORCA: no magnetic dipole arrow and no R(velocity) |
| ORCA negative RPA weights | ORCA `tda false` with `tprint` below the default: cclib stops |

The installation of the laboratory (`/applis/cclib`) has all of them.

## History

The project was started in 2010 by Thomas CAUCHY, assistant professor in the MOLTECH-Anjou
laboratory of the University of Angers (France). At first a computational chemistry program
maintained by Thomas CAUCHY and Yohann MORILLE, the project boomed with the involvement of
Benoit DA MOTA, computer scientist and assistant professor in the LERIA laboratory of the
University of Angers.

Repository: https://github.com/tcauchy/QuChemReport
