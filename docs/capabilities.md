# Capabilities and limits

What QuChemReport does today (version of 27/11/2026), and what it does not do or does only
approximately. For how to prepare the calculations, see [calculations.md](calculations.md); for
the content of each section of the report, [report.md](report.md); for what is planned,
[roadmap.md](roadmap.md).

## What it reads

- **Programs**: Gaussian (09 and 16) and ORCA (5 and 6), through cclib. Tested on real
  calculations with Gaussian 09 D.01 and ORCA 6.1 (ORCA 5: one frozen test output; Gaussian 16:
  only through cclib's own tests).
- **Calculations on one molecule (one conformer)**: ground-state optimization and frequencies,
  single points of the cation and anion, vertical excited states (TD-DFT, TDA, singlets and
  triplets), optimized excited states (S1, T1), several steps in one file (`--Link1--`,
  compound ORCA jobs).
- **Wavefunctions**: restricted and unrestricted (open-shell references such as triplet O2),
  spherical and Cartesian basis functions up to g, mixed 6D/7F basis sets, effective core
  potentials, implicit solvent (PCM, SMD, CPCM).

## What it checks

The log files must describe the same molecule at the same level of theory: same atoms and
topology, same conformer (nuclear repulsion energy), consistent charges and multiplicities, same
functional with the same dispersion correction, same basis set and solvent. The report stops with
a message when they do not. It warns about optimizations that did not converge, imaginary
frequencies, spin contamination, incomplete excitation amplitudes, and excited-state densities
that are only approximate.

## What it reports

- **Molecule and calculations**: picture with atom numbers, formula, mass, InChI, SMILES; table
  of the log files; computational details (method, functional and dispersion, basis set,
  solvent, convergence).
- **Wavefunction**: energies, frontier and requested orbitals, molecular electrostatic potential
  on the van der Waals surface with its minima and maxima, Mulliken, Hirshfeld and CM5 charges.
- **Frequencies**: thermochemistry and intense IR vibrations.
- **Reactivity**: conceptual DFT indices, condensed and 3D Fukui functions, dual descriptor.
- **Excited states**: table of the selected states (energies, oscillator and rotatory strengths in
  the length and velocity gauges, charge-transfer indices Lambda, d_CT, q_CT, main excitations);
  absorption and CD spectra (CD only for a chiral molecule); exact electron density differences
  (attachment - detachment); natural transition orbitals; transition densities with the electric
  and magnetic transition dipoles drawn to scale, and the dissymmetry factor g.
- **Optimized excited states**: emission and CPL spectra, density difference of the emission.
- **Spin-orbit coupling (ORCA `dosoc`)**: coupling matrix elements, Jablonski-Perrin diagram,
  phosphorescence (radiative rates and lifetime of the T1 sublevels).
- **Coordinates** of every distinct geometry, in pm.

## Pictures and output

- 3D pictures rendered off screen with VTK (Mayavi), or with POV-Ray (soft shadows, ambient
  occlusion); one or two views chosen from the shape of the molecule, the same for all the
  geometries of a molecule; a second view when a dipole arrow points at the viewer.
- Spectra with the main axis in cm-1, eV or nm, and another unit (or none) on the top axis.
- Report in PDF (LaTeX), in three levels of detail (full, SI, text); the pictures are kept as PNG
  files for talks and papers.
- A folder that already holds a report is not overwritten unless asked (`output.if_exists`).

## Limits

### Inputs

- Only Gaussian and ORCA. Other programs read by cclib are not supported.
- One molecule, one conformer per report: no Boltzmann average of conformers, no comparison of
  several molecules.
- The pictures are rebuilt from the basis set and the molecular orbitals: they must be printed
  (Gaussian `pop=full gfprint`; ORCA `Print[ P_Basis ] 2` and `Print[ P_MOs ] 1`).
- cclib: the release 1.8.1 cannot read ORCA 6 outputs; use its development version. Several cclib
  changes made for QuChemReport are not merged yet (effective core potentials, dispersion model,
  de-excitation amplitudes, ORCA transition and magnetic dipoles); without them some outputs are
  missing or approximate, see the table in the [README](../README.md#requirements). The
  installation of the laboratory has all of them.

### Excited states

- The programs print by default only the main excitation amplitudes (|X| > 0.1, no de-excitation
  amplitudes Y): density differences, NTO and transition densities are then approximate (the
  report says so). Gaussian `IOp(9/40=4)` or ORCA `tprint 1e-6` print all of them.
- ORCA without TDA (`tda false`) prints only the weights of the excitations: no transition density
  for these states. With TDA (ORCA's default), the rotatory strengths of the length and velocity
  gauges can differ strongly, even in sign (marked "sign uncertain").
- The magnetic transition dipole and R(length) depend on the origin of the program;
  R(velocity) does not.
- Spin-orbit couplings and phosphorescence: ORCA only (Gaussian does not compute them).
- Open-shell references: TD-DFT gives only the states with the spin projection of the reference
  (no spin-flip; singlet O2 is not among the states of triplet O2).
- Emission: one optimized excited state per optimization; no vibronic structure, no Kasha
  analysis between several optimized states.
- Choice of the states drawn: the first five states plus the bright ones (f > 0.1 or R > 10), with
  singlets and triplets counted together; to be replaced by a selection of the relevant states
  ([roadmap](roadmap.md)).

### Spectra

- Gaussian bands of fixed width (FWHM 3000 cm-1, not configurable), no vibronic structure, no
  solvent broadening beyond the computed energies.
- The highest computed states are only a part of the spectrum: the region above the last computed
  state is unknown.

### Electrostatic potential and reactivity

- MEP: memory grows quickly with the size of the molecule (the grid and a Fourier transform);
  run large molecules on a node with enough memory, or disable `MEP_maps`. Effective core
  potentials need the cclib change for the core electrons.
- Fukui functions need the single points of the cation and anion with their orbitals, at the
  geometry of the reference.

### Practical

- Rendering is done in software (Xvfb, Mesa): the 3D pictures take most of the run time (full
  report of acetone with its excited states: about 3 minutes with VTK, 11 with POV-Ray; much
  more for large molecules).
- The report is in English only.
