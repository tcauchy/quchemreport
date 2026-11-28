# QuChemReport architecture

This page is for the people who maintain QuChemReport. It describes how a report is built,
the data that flow between the stages, and the conventions of the code. The user
documentation is in the other pages of this folder.

The C4 diagrams (context, containers, components as they are and as they should become) are in
[architecture/c4](architecture/c4/README.md); the rules for changing the code are in
[CONTRIBUTING.md](../CONTRIBUTING.md).

## Overview

A report is built from a YAML configuration file (see `example.yaml`) listing log files of
quantum chemistry calculations (Gaussian or ORCA) on one molecule, with their types (`opt`,
`freq`, `td`, `sp`...). The command line

    python cli.py -c report.yaml

runs six stages, called in order by `execution_engine/engine.run`:

```
YAML file
  │  config/            Config (pydantic): validated configuration
  ▼
log files
  │  parser_engine/     one LogData per calculation step (cclib + Open Babel)
  ▼
List[LogData]
  │  quality_check_engine/   conformity tests, choice of the reference log
  ▼
ValidatedData
  │  data_enrichment/   grids (MO, MEP, Fukui, EDD, NTO), spectra, indices
  ▼
ComputedData
  │  visualization_engine/   PNG pictures (Mayavi) and plots (matplotlib)
  ▼
pictures + ComputedData
  │  report_generator/  LaTeX document (pylatex) compiled with pdflatex
  ▼
<output_path>/<molecule_designated_name>/<name>_<verbosity>_report.txt.pdf
```

Which computations are run depends on the outputs requested in the YAML file:
`data_enrichment/compute_flags.resolve_compute_flags` turns the `output.include` options into a
`ComputeFlags` object, read by the computation and visualization stages.

## Stages

### 1. Configuration (`config/`)

- `YAML_loader.load_yaml_config` reads the YAML file and builds a `config.Config` object.
- `config.py` defines the pydantic models of the configuration (`Config`, `LogFile`, `Output`,
  `OutputInclude`...). Validation happens when the object is created: unknown solvers, missing
  log files or inconsistent options stop the program with a message.

### 2. Reading the log files (`parser_engine/scanlog.py`)

- `split_logfile` splits a log file into its calculation steps (Gaussian `Link1` jobs, ORCA
  compound jobs) and writes each step to a temporary folder. A log file without a normal
  termination gives no step and stops the program with an explicit message.
- `logfile_to_obj` reads one step with cclib, builds the molecule with Open Babel (formula,
  connectivity, InChI, SMILES) and fills a `LogData` model, section by section
  (`molecule_section`, `parameters_section`, `results_section`, `metadata_section`).
- `process_logfile_list` does this for every log file of the YAML file and gives the type
  written in the YAML file to every step of the file.

`utility_services/log_data.py` defines `LogData`, a pydantic model with four sections:
`metadata`, `molecule`, `comp_details` (program, method, basis set...) and `results`
(geometry, wavefunction, frequencies, excited states).

The spin-orbit coupling results of ORCA (`dosoc true`) are read by
`parser_engine/orca_soc.py` into `results.excited_states.soc`: cclib reads only the spin-free
states of these calculations, on purpose (cclib issues #1172 and #1419); this reader should
move to cclib once it has attributes for them. `data_enrichment/compute.py`
(`compute_phosphorescence`) derives the phosphorescence of an optimized triplet from them.

QuChemReport relies on cclib for the parsing. Two cclib changes made for QuChemReport are
installed in `/applis/cclib` until they are released: core electrons with effective core
potentials (cclib PR #1972) and the dispersion model (cclib PR #1973).

### 3. Conformity tests (`quality_check_engine/conformity.py`)

`conformity.tests` checks that the log files describe the same molecule at the same level of
theory, then chooses the reference log:

- `extract_data_models` gathers the values to compare (formulas, functionals with their
  dispersion correction, basis sets, charges, nuclear repulsion energies...);
- `run_selected_tests` runs the tests enabled in `quality_control.checks` (same formula, same
  theory, same conformer from the nuclear repulsion energy, consistent charges and
  multiplicities, presence of a ground state optimization). A failed test stops the program;
- `select_reference_log` takes the log file marked `reference: true` in the YAML file (exactly
  one is required) and `reference_model_index` its first step with MO coefficients;
- `select_reference_data` chooses the model used for the grid computations: the reference
  itself if it has the basis set and the MO coefficients, otherwise another log of the same
  charge, multiplicity, geometry (orientation included) and solvent that has them;
  `test_basis_and_mo_coeff` checks that this basis set can be put on a grid.

The result is a `ValidatedData` object (`utility_services/validated_data.py`).

### 4. Computations (`data_enrichment/`)

`compute.compute_data` fills a `ComputedData` object (`utility_services/computed_data.py`):

- simple data copied from each log (`extract_simple_data`);
- MO labels and energies, MO pictures (`compute_MO_diagrams`);
- molecular electrostatic potential and its extrema on the molecular surface
  (`compute_MEP_maps`, `mep_surface.py`);
- population analysis, conceptual DFT indices and Fukui functions (`compute_reactivity`);
- excited states: selection of the states, absorption and CD spectra, transitions, electron
  density differences, natural transition orbitals (`compute_excited_states`, `nto.py`);
- optimized excited states: emission and CPL spectra, transitions, density differences
  (`compute_optimized_excited_states`).

The grids are computed by `calc_orb.py`. Orbitals and densities are evaluated with QMCkl
(`qmckl_backend.py`); the former engine, orbkit, can still be selected with the environment
variable `QUCHEMREPORT_ORBITAL_BACKEND=orbkit` for comparisons. The MEP is the nuclear
potential minus the Coulomb potential of the electron density, computed by FFT convolution on
the grid; the cores of heavy atoms, badly sampled by the grid, are integrated apart
(`calc_orb.Potential`). On the molecular surface, its root mean square difference with
Gaussian's cubegen is below 0.002 a.u. (`tests/data/mep_reference`,
`test_mep_against_gaussian`).

### 5. Pictures (`visualization_engine/`)

`generate_visualization.generate_visualization` writes the pictures in the output folder:

- `visu_mayavi.py`: 3D pictures (topology, MO, MEP, EDD, Fukui, NTO), rendered off-screen with
  Mayavi/VTK;
- `camera.py`: points of view chosen from the geometry, independent of the orientation of the
  program; a second picture (`_cam2`) only for the molecules that need one;
- `visu_plots.py`: spectra (IR, UV-visible, CD, emission) with matplotlib;
- `visu_txt.py`: text files of the spectra.

### 6. Report (`report_generator/generate_report.py`)

`generate_report` builds a LaTeX document with pylatex, section by section (`fill_*_section`
functions, which call `render_*` functions for each table or figure), and compiles it with
`pdflatex` (`render_report_document`). The `output.verbosity` option gives three kinds of
report: `full` (all tables and pictures), `si` (supporting information layout, fewer
pictures) and `text` (no grid pictures).

## Data models

| Model | Module | Content | Built by |
|---|---|---|---|
| `Config` | `config/config.py` | YAML configuration | `load_yaml_config` |
| `LogData` | `utility_services/log_data.py` | one calculation step | `scanlog.logfile_to_obj` |
| `ValidatedData` | `utility_services/validated_data.py` | conformity results, reference log | `conformity.tests` |
| `ComputedData` | `utility_services/computed_data.py` | everything shown in the report | `compute.compute_data` |

`ComputedData` is a tree of dataclasses. Most fields are lists with one value per `LogData`
(`init_computed_data(n)` creates them with `None`); the fields computed for the reference log
only (MEP, Fukui functions...) are single values.

## Conventions

- **Units.** Geometries of `LogData` are in Angstrom (as given by cclib). Grids, MEP and
  densities are in bohr and atomic units. `utility_services/units.a0_to_A` is the Bohr radius in
  Angstrom: coordinates in Angstrom are divided by it to get bohr (it was named `A_to_a0`
  until November 2026).
- **Log indexes.** The models are numbered in the order of the YAML file, a multi-step log file
  giving several models. `metadata.config_idx` is the index of the YAML entry of a model;
  `ComputedData.metadata.ref_log_file_idx` the index of the reference model.
- **Picture names.** Pictures are named `img-log<k>-<content>.png`, where `k` is the index of
  the model plus one (`img-log1-...` for the first model of the YAML file); the second view of a
  3D picture is `..._cam2.png`. Pictures of the reference only, such as the Fukui functions,
  have no number (`img-fukui-SP_plus.png`). Text files of the spectra are `log<k>-...txt`.
- **Documentation.** Each non-obvious choice is explained where it is made, with the
  measurements that decided it and the alternatives that were rejected (for instance
  `visu_mayavi._save_views_through`, `camera.py`, `mep_surface.py`). Docstrings use a sentence
  saying what the function does, then NumPy-style `Parameters` and `Returns` sections for the
  public functions.

## Tests

The tests are in `tests/` (see `tests/README.md`): `tests/run_tests.sh` runs the fast tests,
`tests/run_tests.sh all` also the full report runs. They use small Gaussian 09 and ORCA 6.1
outputs stored in `tests/data` (see `tests/data/README.md`).
