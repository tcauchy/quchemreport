# The report

The report is a PDF file. Its sections follow the order below; a section without data, or
disabled in `output.include`, is left out. With `verbosity: si` the layout is plain and some
pictures are left out; with `verbosity: text` there are no orbital, MEP or density pictures.

## Sections

1. **Log files**: the calculations, their types, the reference, and **warnings** on the
   structures (optimization not converged, imaginary frequencies: not a minimum).
2. **Molecule**: picture with the atom numbers, identification (formula, charge,
   multiplicity, mass, InChI, SMILES), coordinates of the reference (for molecules of fewer
   than 15 atoms; the others are at the end), with the calculations done at the same geometry.
3. **Computational details**: program and versions, method, functional with its dispersion
   correction (e.g. `B3LYP + D3(BJ)`), basis set, solvent, convergence criteria. A value that
   differs between calculations is given with the log files concerned. A second table gives
   what is specific to each log file: convergence of the optimization (with the thresholds
   once when they are the same), nuclear repulsion energy, temperature, excited states read.
4. **Wavefunction**: one table of the total energies and of the energies of the requested
   orbitals (one row per group of identical calculations, alpha and beta rows for unrestricted
   ones); orbital pictures;
   molecular electrostatic potential (MEP) with its extrema; atoms whose charge deviates by
   more than one standard deviation from the mean.
5. **Frequencies**: thermochemistry (zero-point, thermal, enthalpy, free energy, entropy) and
   IR vibrations more intense than 50 km/mol (otherwise a sentence: no IR-active vibration, as
   for O2, or the most intense one).
6. **Reactivity**: conceptual DFT indices (electron affinity A = E(N) - E(N+1), ionization
   potential I = E(N-1) - E(N), electronegativity (I + A)/2, hardness I - A,
   electrophilicity, DeltaN, in eV), condensed Fukui functions (Mulliken and Hirshfeld
   charges; every atom up to 20 atoms, otherwise the atoms whose dual descriptor departs from
   the mean by more than one standard deviation) and 3D Fukui functions and dual descriptor.
7. **Excited states**: table of the selected states (S<n>/T<n>, with the rank of the state in
   the log when singlets and triplets are mixed; spatial symmetry; wavelength, wavenumber,
   oscillator and rotatory strengths; Lambda, d_CT in pm, q_CT in e; main excitations in %
   named from the HOMO and LUMO, e.g. H-1->L), absorption and CD spectra (Gaussian bands,
   3000 cm-1 wide; main axis in wavenumber or wavelength, `output.spectrum_axis`; range: every
   computed band with three widths on each side, or `output.spectrum_range`), density
   differences and natural transition orbitals shown as grids of the selected states.
   Open-shell reference (e.g. triplet O2): TD-DFT only gives the states with the same spin
   projection, named after the multiplicity of the reference (T1, T2... for a triplet; D1... for
   a doublet), with their <S^2> in an extra column, in bold when it is more than 0.1 away from
   the pure value (spin contamination). Singlet O2 (a1Delta_g) is not among them: it needs a
   spin-flip TD-DFT or a separate singlet calculation.
   With ORCA and `dosoc true`: spin-orbit coupling matrix elements and Jablonski diagram
   (see below).
8. **Optimized excited states**: emitting state, emission and CPL spectra, density difference.
   For a triplet optimized with `dosoc true` (ORCA): phosphorescence (see below).
9. **Other topologies** and **atomic coordinates** of the geometries that differ from the
   reference (optimized ions or excited states); calculations at the same geometry share a table.

## Reading the pictures

**Points of view.** The view is chosen from the geometry of the molecule, so it does not depend
on the orientation written by the program: the atoms are spread as much as possible, with few
atoms hiding each other; heteroatoms are put in front; flat molecules are seen face-on, slightly
tilted; the long axis is horizontal. All the pictures use the same view, so that the atom numbers
of the molecule picture can be followed on the orbitals, the MEP or the density differences. A
second picture, at about 90 degrees, is added only when one view is not enough: rings facing
each other (pi-stacking, helicenes, spiro compounds, folded molecules), rings seen edge-on in
the first view, or a globular molecule. `output.camera_views: 1` or `2` in the YAML file removes
or forces the second picture (see [Configuration](configuration.md)).

The colours named below are the defaults; `output.colors` changes them, and the captions follow
(see [Configuration](configuration.md)).

The 3D pictures are rendered by VTK (Mayavi), with white highlights on the atoms and surfaces
(`output.specular`), or, with `output.renderer: povray`, ray-traced by POV-Ray from the same
scene and camera: smooth edges, soft shadows and ambient occlusion, same framing and labels.

**Orbitals.** Positive lobes purple, negative lobes cream; the surfaces enclose 20 % of
each sign. Unrestricted calculations: alpha (left) and beta (right) pictures of each orbital.
Degenerate orbitals (energies within 0.01 eV, e.g. the pi pairs of O2) are named in the caption;
any combination of them is as valid, so a degenerate pair is drawn as the two combinations seen
alike from the point of view (for a pi pair seen from the side, both at 45 degrees), instead of
one orbital in the picture plane and the other pointing at the camera.

**Density differences** (excited state minus ground state): blue where the excited electron
goes, cream for the hole; each surface encloses 30 % of the density gained or lost. Two states
with the same single excitation (e.g. S1 and T1, both HOMO->LUMO) therefore give the same picture.

**Natural transition orbitals** (option `natural_transition_orbitals.enabled`): for each
selected state, the hole and electron orbitals of its main pairs, blue-grey and cream for
the two phases (absolute signs are arbitrary), with their weights. They are computed from the printed
excitation amplitudes X (the de-excitation amplitudes Y of TD-DFT are left out); the squared
singular values are normalized over all the printed amplitudes of the state, alpha and beta
blocks of open-shell states being decomposed separately with a common normalization. Missing
or truncated amplitudes change both the orbitals and the weights (with Gaussian, `IOp(9/40=4)`
prints more of them; a warning says when the printed amplitudes are incomplete). The amplitudes
refer to the orbitals of the TD calculation: QuChemReport uses the orbitals of the TD log when
it has them, otherwise those of the reference, after checking that geometry, spin, basis set,
method and orbital energies match. These checks cannot prove that orbital phases or rotations
are the same: prefer a TD log with its own orbitals, or a TD started from the checkpoint of the
reference.

**Density differences** (excited states): blue where the electron density increases in the
excited state, cream where it decreases; the surfaces enclose 30 % of each sign. The table of
the states gives the charge-transfer indices: Tozer's Lambda (overlap of the orbitals involved,
low for a charge transfer), the distance d_CT between the barycentres of the density increase
and decrease (pm) and the transferred charge q_CT (e).

**Fukui functions**: blue regions are the most electrophilic (F+) or nucleophilic (F-) sites;
dual descriptor: blue electrophilic, cream nucleophilic.

**Molecular electrostatic potential.** The MEP is mapped on the 0.001 e-/bohr3 electron density
surface (the van der Waals surface of Bader, as in the work of Murray and Politzer). Red is
negative (lone pairs, pi systems: electron-rich), blue positive (acidic hydrogens, sigma-holes
of halogens: electron-poor), green near zero. The colour scale is fixed and symmetric, the same
for all compounds of the same charge, so that pictures of different compounds can be compared:
-0.05 to +0.05 a.u. (-31 to +31 kcal/mol) for neutral molecules, 0.20 a.u. per unit of charge
for ions (-0.20 to +0.20 a.u. for a cation or an anion). These ranges were calibrated on 141
calculations of the laboratory: for a neutral molecule, only the strongest extrema saturate
(about 2 % of the surface), and their values are in the table. The MEP of an ion has the sign of
its charge everywhere: a cation is all blue, an anion all red, the colours showing the more and
less positive (or negative) regions. The range is given by the colour bar and the caption. The surface is opaque, with the molecule
drawn over it in transparency.

The **local minima and maxima** of the MEP on the surface are marked by numbered spheres (white:
minima, black: maxima) and listed in the table below the picture, with their value in kcal/mol
and atomic units and the nearest atom. The most negative and most positive values (V_S,min and
V_S,max) are in bold. Extrema that stand out by less than 0.5 kcal/mol from their surroundings
(prominence: difference between the value at the extremum and the median of the MEP 1.5 to 2.5
bohr around it) are left out: a study of 4661 extrema of 146 calculations of the laboratory showed
that these are artefacts of the surface mesh (0.6 % of the extrema), whereas extrema with a small
value can be real sites (sigma-holes near 0 kcal/mol); when none is left, the caption says so. Extrema on the hidden side of the molecule
are in the table but their numbers are not written in the picture (with two points of view, each
picture numbers the extrema it shows). These values agree with Gaussian (cubegen) within 0.75 kcal/mol.

## Spin-orbit coupling and phosphorescence (ORCA)

These results need ORCA TD-DFT calculations with `dosoc true` ([Preparing the
calculations](calculations.md)).

**Spin-orbit couplings.** The table gives |<Tn|H_SO|Sm>| in cm-1 between the first triplets
(rows, up to T6) and the ground state and first singlets (columns, S0 to S3), each being the
norm over the three M_S components of the triplet. Small values for states of the same
character and large ones between states of different orbital characters (El-Sayed rules):
for acetone, <T1|H_SO|S1> is close to 0 (both n-pi*), <T2|H_SO|S1> about 40 cm-1.

**Jablonski-Perrin diagram.** Vertical energies (eV) of the spin-free singlets (blue) and
triplets (red) at the geometry of the calculation. The dashed arrows are the intersystem
crossings from S1 to the triplets below S1 + 0.3 eV, and from T1 to S0, labelled with their
coupling in cm-1. Energies are vertical (no relaxation of the excited states): the diagram
shows which channels are allowed by the couplings and the energies, not their rates.

**Phosphorescence.** For a triplet optimized with `dosoc true`, SOC splits T1 into three
sublevels, which borrow intensity from the singlets. The table gives, for each sublevel, its
energy, wavelength, oscillator strength f to S0 and radiative rate (Strickler-Berg,
k_r = f nu^2 / 1.499 s-1 with nu in cm-1). The three sublevels being in thermal equilibrium at room
temperature (their splitting, 0.3 cm-1 for acetone and rarely more than a few tens of cm-1, is
much smaller than kT = 207 cm-1 at 298 K), the radiative rate of the triplet is their mean and
the radiative lifetime its inverse. This is a radiative lifetime: the measured lifetime is
shorter, because of the non-radiative decay (not computed). The spectrum has one Gaussian
band (FWHM 3000 cm-1) per sublevel, weighted by its oscillator strength. The emission is
vertical (energy at the T1 geometry, no vibronic structure).
