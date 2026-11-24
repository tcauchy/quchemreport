# Running QuChemReport

## On the MOLTECH-Anjou cluster

QuChemReport is installed in `/applis/quchemreport`, with its Python environment in
`/applis/conda_envs/quchemreport`. The shell profile of the cluster defines the alias
`quchemreport` (`python /applis/quchemreport/cli.py`), which needs that environment:

```bash
conda activate /applis/conda_envs/quchemreport
quchemreport -c report.yaml
```

The 3D pictures are rendered off-screen. Without a graphical display (ssh without X, batch
jobs), run it through a virtual display:

```bash
xvfb-run -a python /applis/quchemreport/cli.py -c report.yaml
```

The grid computations (orbitals, MEP, densities) are the long part: a few minutes for a small
molecule, much more for large ones with the MEP and the Fukui functions. For those, use a
batch job and set `resources.nproc` to the number of cores you reserve:

```bash
#!/bin/bash
#SBATCH -J report
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
source /etc/profile.d/chemistry.sh
xvfb-run -a /applis/conda_envs/quchemreport/bin/python /applis/quchemreport/cli.py -c report.yaml
```

## Output

The report and its pictures are written in `<output_path>/<molecule_designated_name>/`:

- `<name>_<verbosity>_report.txt.pdf`: the report (and its `.tex` source);
- `img-log<k>-*.png`: the pictures, `k` being the number of the calculation in the YAML file
  order (1 for the first); `_cam2` pictures are second points of view;
- `log<k>-UV-*.txt`: the spectra as text (wavenumber, wavelength, epsilon).

The terminal shows the progress: reading of the log files, conformity tests, computations,
pictures, report. A test that fails stops the program with an explanation (see
[Troubleshooting](troubleshooting.md)).

## Installing elsewhere

QuChemReport needs Python 3.12 with NumPy, SciPy, pydantic, PyYAML, cclib (with the two
changes not yet released: core electrons of effective core potentials, PR #1972, and dispersion
model, PR #1973), Open Babel with its Python bindings, QMCkl (Python bindings), Mayavi with
VTK, matplotlib, pylatex, and a LaTeX distribution with `pdflatex`. orbkit is only needed for
comparisons with the former grid engine.
