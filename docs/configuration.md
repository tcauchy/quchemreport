# The YAML configuration file

A report is described by a YAML file: the log files to read, the checks to run and the parts
of the report. `example.yaml` at the root of the repository is a complete, commented example;
copy it and edit the values. This page lists every option.

General rules:

- indent with spaces, not tabs; write `true` / `false` without quotes;
- options marked *required* must be present, even when their part of the report is disabled;
- a wrong value stops QuChemReport before any log file is read, with a message naming the
  option (for instance `('output', 'verbosity'): Output verbosity must be one of ...`);
- an output whose data are missing in the log files is simply left out of the report.

## `common_solver` (required)

Program of all the log files: `gaussian` or `orca` (GAMESS and NWChem are accepted but not
tested). All the log files must come from this program.

```yaml
common_solver: gaussian
```

## `logfiles` (required)

List of the log files, one entry per file:

| Key | Required | Meaning |
|---|---|---|
| `path` | yes | Path of the log file (`.log` for Gaussian, `.out` for ORCA). |
| `type` | yes | Type of calculation, or a list of types for a file with several steps (e.g. `[opt, freq]`). |
| `reference` | exactly one entry | `true` for the reference calculation (see below). |
| `excited_state_number` | for `opt_es`, `freq_es` | Number of the optimized excited state (1 for S1 or T1); forbidden for the other types. |

Types:

| Type | Calculation |
|---|---|
| `opt` | ground state optimization |
| `freq` | frequencies (ground state) |
| `sp` | single point, e.g. the cation and anion at the optimized geometry (Fukui functions) |
| `td` | vertical excited states (TD-DFT) |
| `opt_es`, `freq_es` | optimization / frequencies of an excited state (emission) |
| `nmr` | NMR (not shown in the report yet) |

The **reference** is the ground state optimization of the molecule: the molecule section, the
orbitals, the MEP and the Fukui functions are drawn from it. It must contain the basis set and
the MO coefficients (see [Preparing the calculations](calculations.md)); if it does not, a log
at the same geometry that has them is used for the pictures.

```yaml
logfiles:
  - path: "calc/OPT.log"
    type: opt
    reference: true
  - path: "calc/FREQ.log"
    type: freq
  - path: "calc/TD.log"
    type: td
  - path: "calc/OPT_ES1.log"
    type: opt_es
    excited_state_number: 1
```

A log file with several jobs (Gaussian `--Link1--`, ORCA `$new_job`) is split into steps; all its
steps get the type of the entry.

## `quality_control` (required)

Checks that the log files describe one molecule at one level of theory. A failed check stops
the program with the values that differ.

| Key | Values | Meaning |
|---|---|---|
| `method_consistency` | `strict`, `lax` | Consistency level (required). |
| `checks.formula` | `true`/`false` | Same molecule (formula without charge). |
| `checks.theory` | `true`/`false` | Same functional, with its dispersion correction, and same basis set. |
| `checks.nuclear_repulsion` | `true`/`false` | Same geometry for the ground state calculations (optimized excited states are left out). |
| `checks.charge` | `true`/`false` | Same charge, single points apart. |
| `checks.multiplicity` | `true`/`false` | Same multiplicity, single points apart. |
| `checks.ground_state_optimization` | `true`/`false` | At least one `opt` or `freq` calculation. |

All checks default to `true`.

## `output` (required)

| Key | Default | Meaning |
|---|---|---|
| `output_path` | required | Folder of the reports; the report is written in `<output_path>/<molecule_designated_name>/`. |
| `molecule_designated_name` | required | Name of the molecule (title of the report, name of the folder and of the file). |
| `format` | required | `pdf` (`html` and `markdown` are accepted but give a PDF for now). |
| `verbosity` | required | `full` (all tables and pictures), `si` (supporting information: plain layout, fewer pictures) or `text` (no orbital, MEP or density pictures). |
| `image_size` | 1200 | Side of the square 3D pictures, in pixels (200 to 8000). 1200 px is 360 dpi in an 8.5 cm column; 2000 px is 600 dpi. |
| `image_grid_reduction` | 2 | The orbital, MEP and NTO pictures are drawn on every k-th point of the computation grid (0.075 bohr): 1 = full grid, 2 = 0.15 bohr (2.5 to 3 times faster to draw). The numerical results always use the full grid. |
| `image_grid_reduction_sensitive` | 1 | Same, for the density differences (EDD, Fukui functions), which are more sensitive. |
| `spectrum_axis` | `wavenumber` | Main axis of the UV-visible, CD, emission, CPL and phosphorescence spectra: `wavenumber` (cm⁻¹, linear in energy) or `wavelength` (nm, as most experimental spectra). The other unit is on the top axis. |
| `spectrum_range` | automatic | Range of the spectra on the main axis, in its unit (cm⁻¹ for `wavenumber`, nm for `wavelength`), for example `[200, 600]`. Automatic: every computed band, with three band widths (FWHM, 3000 cm⁻¹) on each side. |
| `camera_views` | `auto` | Views of the 3D pictures: `auto` (a second picture, at 60-90 degrees, only when one view is not enough, see [The report](report.md)), `1` (never a second picture) or `2` (always, for example to show a stereocentre from two sides; linear molecules keep one view, their side views being all the same). |
| `colors` | see below | Colours of the iso-surfaces of the 3D pictures, below. |
| `renderer` | `vtk` | Renderer of the 3D pictures: `vtk`, or `povray` for ray-traced pictures (anti-aliased edges, soft shadows, ambient occlusion), about 4 s more per 1200 px picture with 16 threads. The scene and camera are those of VTK, so the framing and the labels are the same. Without POV-Ray, a warning is printed and VTK renders the pictures. |
| `povray_executable` | none | Path of the POV-Ray executable, when it is neither in the `PATH` nor in the conda environment of quchemreport. |
| `specular` | 0.5 | Specular reflections (white highlights, glossy look) of the atoms, bonds and iso-surfaces, 0 to 1; 0 is diffuse light only, as before November 2026. The MEP surface keeps diffuse light, so that no highlight hides its colour scale. |
| `include` | required | Parts of the report, below. |

### `output.colors`

Colours of the positive and negative parts of the iso-surfaces. Every key is optional; an
omitted colour keeps its default:

| Key | Positive (default) | Negative (default) |
|---|---|---|
| `MO` | `"#66003C"` purple (Tyrian purple): one orbital phase | `"#FFEBDC"` cream: the other phase |
| `EDD` | `"#008080"` blue: density gained (excited electron); also the charge-transfer dipole arrow | `"#FFEBDC"` cream: density lost (hole) |
| `fukui` | `"#008080"` blue: F+ (electrophilic), F- (nucleophilic), dual descriptor (electrophilic) | `"#FFEBDC"` cream: dual descriptor (nucleophilic) |
| `NTO` | `"#5B6A8F"` blue-grey: one phase of the hole and electron orbitals | `"#FFEBDC"` cream: the other phase |
| `transition_density` | `"#F28000"` orange: positive charge transition density | `"#FFEBDC"` cream: negative charge transition density (the transition dipole goes from cream to orange) |

```yaml
output:
  colors:
    NTO: {positive: "#2F4B7C", negative: lightpink}
    EDD: {positive: [0.0, 0.4, 0.7], positive_name: "dark blue"}
```

A colour is `"#RRGGBB"` (quoted: `#` starts a YAML comment), a colour name known to matplotlib
(`teal`, `lightpink`, `tab:blue`...) or `[r, g, b]` with values in 0-1. The captions of the
report name the colours: `positive_name` / `negative_name` when given, otherwise the colour word,
otherwise a plain name made from the value ("greyish blue", "light pink"). Unknown keys stop the
program (a misspelt `postive` would otherwise leave the default silently).

The NTO colours were green and gold until November 2026; they were replaced because they merge
for red-green colour-blind readers (CIEDE2000 difference 18 for protanopia, 45 in normal vision),
while the blue-grey (from the oleron map of F. Crameri) and the cream differ mainly in lightness
and stay at 44-47 for protanopia, deuteranopia and tritanopia. Choose pairs of a dark and a
light colour for the same reason; they also survive grey-scale printing. The light surfaces are
cream rather than white, so that a small lobe is not taken for a hydrogen (white spheres); a
paler cream such as `#FFF3F0` already looks white once shaded. The MEP colour scale (turbo) is
not set here.

### `output.include`

Every key below is required (write `false` to disable a part).

**`molecule`**

| Key | Meaning |
|---|---|
| `molecule_representation` | Picture of the molecule with the atom numbers (and of the other topologies, if any). |
| `molecule_identification` | List of tags among `formula`, `charge`, `spin multiplicity`, `monoisotopic mass`, `inchi`, `directory name`. The list must not be empty; for now all the available values are shown. |
| `atomic_coordinates` | Tables of the atomic coordinates (at the end of the report for large molecules). |

**`computational_details`**: table of the program, method, functional (with its dispersion
correction), basis set, solvent and convergence criteria of each log file.

**`wavefunction`**

| Key | Meaning |
|---|---|
| `MO_analysis.MO_list` | Orbitals to analyse: `homo`, `lumo`, with an offset (`homo-1`, `lumo+2`). Required when `energies` or `MO_diagrams` is true. |
| `MO_analysis.energies` | Energies of these orbitals. |
| `MO_analysis.MO_diagrams` | Pictures of these orbitals. |
| `MEP_maps` | Molecular electrostatic potential on the molecular surface, with its minima and maxima (see [The report](report.md)). Memory-hungry for large molecules: if it fails, the report goes on without it. |
| `population_analysis` | Atomic charges that deviate from the mean (Mulliken, Hirshfeld, CM5 when available). |

**`frequencies`**: `thermo_data` (thermochemistry) and `IR_spectrum` (intense IR vibrations).

**`reactivity`**: `CDFT_indices` (conceptual DFT global indices) and `fukui_functions`
(condensed values and 3D pictures). Both need the single points of the cation and of the
anion at the reference geometry (type `sp`).

**`excited_states`** (logs of type `td`)

| Key | Meaning |
|---|---|
| `selection.mode` | States shown: `dominant_only` (the first five, then the intense ones), `first_n` or `all_states`. |
| `selection.first_n` | Number of states for `first_n`. |
| `transitions` | Table of the states (wavelength, strengths, main excitations, charge-transfer indices). |
| `absorption_spectrum`, `CD_spectrum` | Spectra (Gaussian broadening). |
| `electron_density_difference` | 3D pictures of the density difference of each selected state. |
| `natural_transition_orbitals.enabled` | Natural transition orbitals of the selected states (default `false`). |
| `natural_transition_orbitals.max_pairs` | Pairs shown per state (default 2). |
| `natural_transition_orbitals.min_weight` | Minimum relative weight of a pair (default 0.01). |

**`optimized_excited_states`** (logs of type `opt_es`): `transitions`, `emission_spectrum`,
`CPL_spectrum`, `electron_density_difference` of the emitting state.

**`NMR`**: accepted but not shown in the report yet.

## `logging` (required)

| Key | Values | Meaning |
|---|---|---|
| `console` | `true` | Messages in the terminal (must be true for now). |
| `level` | `debug`, `info`, `warning`, `error` | `debug` also prints the details of the tests, the tracebacks and the pdflatex output, and keeps the `.aux` and `.log` files of LaTeX (the `.tex` file is always kept). |

## `resources` (optional)

| Key | Default | Meaning |
|---|---|---|
| `nproc` | 20 | Threads for the grid computations (orbitals, MEP, densities). Use the number of cores you reserved. |
| `memory`, `mayavi_headless` | | Not used yet. |

## Ignored sections

`camera_view` (still present in older YAML files) is ignored: the points of view of the
pictures are chosen automatically from the geometry (see [The report](report.md)); only their number
can be set, with `output.camera_views`.
