# The report

The report is a PDF file. Its sections follow the order below; a section without data, or
disabled in `output.include`, is left out. With `verbosity: si` the layout is plain and some
pictures are left out; with `verbosity: text` there are no orbital, MEP or density pictures.

## Sections

1. **Molecule**: picture with the atom numbers, identification (formula, mass, InChI,
   SMILES).
2. **Log files**: one table of the calculations grouped by kind, in the order of a study
   (ground-state optimization, frequencies, vertical excited states, single points,
   excited-state optimizations), with the charge and multiplicity of each log, its geometry
   (optimized, or "of OPT.log" for a calculation at the geometry of an optimization) and the
   states (number of singlets/triplets of a TD, optimized state S1/T1); the reference is
   marked. Then **warnings** on the structures (optimization not converged, imaginary
   frequencies: not a minimum).
3. **Computational details**: program and versions, method, functional with its dispersion
   correction (e.g. `B3LYP + D3(BJ)`), basis set, implicit solvent (model, solvent and
   dielectric constant, e.g. `water (SMD-IEFPCM, dielectric constant 78.36)`, or "none (gas
   phase)"), convergence criteria (restricted or unrestricted: see the energy table of section
   4). A value that
   differs between calculations is given with the log files concerned. A second table gives
   what is specific to each log file: convergence of the optimization (with the thresholds
   once when they are the same), nuclear repulsion energy, temperature, excited states read.
4. **Wavefunction**: one table of the total energies and of the energies of the requested
   orbitals (one row per group of identical calculations, alpha and beta rows for unrestricted
   ones); orbital pictures;
   molecular electrostatic potential (MEP) with its extrema; atoms whose charge deviates by
   more than one standard deviation from the mean, for each charge model in the log: Mulliken,
   Hirshfeld and CM5 (Gaussian `pop=hirshfeld` prints both; ORCA `! Hirshfeld` gives
   Hirshfeld, ORCA has no CM5).
5. **Frequencies**: thermochemistry (zero-point, thermal, enthalpy, free energy, entropy) and
   IR vibrations more intense than 50 km/mol (otherwise a sentence: no IR-active vibration, as
   for O2, or the most intense one).
6. **Reactivity**: conceptual DFT indices (electron affinity A = E(N) - E(N+1), ionization
   potential I = E(N-1) - E(N), electronegativity (I + A)/2, hardness I - A,
   electrophilicity, DeltaN, in eV), condensed Fukui functions (Mulliken and Hirshfeld
   charges; every atom up to 20 atoms, otherwise the atoms whose dual descriptor departs from
   the mean by more than one standard deviation) and 3D Fukui functions and dual descriptor,
   side by side in one figure.
7. **Excited states**: table of the selected states (S<n>/T<n>, with the rank of the state in
   the log when singlets and triplets are mixed; spatial symmetry; wavelength, wavenumber,
   oscillator and rotatory strengths, R in the length and velocity gauges when the log gives the
   velocity and magnetic transition dipoles (Gaussian); Lambda, d_CT in pm, q_CT in e; main excitations in %
   named from the HOMO and LUMO, e.g. H-1->L), absorption and CD spectra (CD only for a chiral molecule; Gaussian bands,
   3000 cm-1 wide by default, `output.spectrum_fwhm`; axes in cm-1, eV or nm, `output.spectrum_axis` and
   `spectrum_secondary_axis`; range: from the lowest computed state, dark states as open circles, to
   the last visible band, with a 10 % margin, or `output.spectrum_range`), density
   differences and natural transition orbitals shown as grids of the selected states.
   **Selected states** (`selection.mode: relevant`, the default): the three lowest states of
   each multiplicity (S1-S3, T1-T3), even dark; for each band of the absorption spectrum, and of
   the positive and negative CD bands, at least 20 % of the largest band of the same spectrum
   (`band_share`), the states that make half of it at its maximum; with spin-orbit couplings
   (ORCA), T1 and the triplets within 0.3 eV of S1 coupled to it; the states of
   `selection.states`; at most 15 (`max_selected_states`). A band near the last computed state
   is left out (the spectrum is cut there). The note under the table gives the reason of each
   state. Calibrated on [6]helicene (40 singlets, 20 triplets), a push-pull dye, acetone and
   CHFClBr: 7 to 11 states.
   Open-shell reference (e.g. triplet O2): TD-DFT only gives the states with the same spin
   projection, named after the multiplicity of the reference (T1, T2... for a triplet; D1... for
   a doublet), with their <S^2> in an extra column, in bold when it is more than 0.1 away from
   the pure value (spin contamination). Singlet O2 (a1Delta_g) is not among them: it needs a
   spin-flip TD-DFT or a separate singlet calculation.
   With ORCA and `dosoc true`: spin-orbit coupling matrix elements and Jablonski diagram
   (see below).
8. **Optimized excited states**: emitting state, emission and CPL spectra, density difference
   drawn for the emission (see below).
   For a triplet optimized with `dosoc true` (ORCA): phosphorescence (see below).
9. **Other topologies** and **atomic coordinates** in pm: the reference first, then the
   geometries that differ from it (optimized ions or excited states); calculations at the same
   geometry share a table. All lengths of the report are in pm (coordinates, MEP extremum to
   nearest atom, d_CT).

In the tables, the first column is aligned as before and the other columns are centred; the
two-column lists (molecule identification, computational details) stay left-aligned.

## Reading the pictures

**Points of view.** The view is chosen from the geometry of the molecule, so it does not depend
on the orientation written by the program: the atoms are spread as much as possible, with few
atoms hiding each other; heteroatoms are put in front; flat molecules are seen face-on, slightly
tilted; the long axis is horizontal. All the pictures use the same view, so that the atom numbers
of the molecule picture can be followed on the orbitals, the MEP or the density differences. The
other geometries of the molecule (optimized excited states, ions) are first superimposed on the
reference, so that they are seen from the same side even when the program wrote them in another
orientation. In the report the white border of the pictures is cut (the same for all the
pictures of a grid); the picture files themselves are kept whole. A
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
They are the unrelaxed density differences of the amplitudes printed in the log, X and Y:
attachment minus detachment densities, which keep the cross terms between configurations sharing
an orbital (until November 2026 the diagonal sum C^2 (phi_a^2 - phi_i^2) left them out, up to
50 % of the map off for the mixed states of HCFClBr). When the printed amplitudes of a state are
incomplete (sum X^2 below 0.99), the report says that its density difference, d_CT and q_CT are
approximate: up to about 20 % of the map and 35 pm of d_CT on acetone and HCFClBr; print all
the amplitudes (`IOp(9/40=4)`) for exact ones.
The density differences of the optimized excited states are drawn for the emission (ground
state minus excited state, at the excited-state geometry): blue where the electron goes back to
when the molecule returns to the ground state, cream where it comes from, so the colours are
the reverse of an absorption picture of the same transition.

**Transition densities** (with the density differences): for each selected singlet state
(each state of an unrestricted calculation), two pictures side by side: the charge transition
density, orange where it is positive and cream where it is negative (30 % of each sign), and
the transition dipole as a blue arrow through its centre, from the cream to the orange regions
(1 e.bohr = 4 bohr), or a blue sphere for a forbidden transition. The arrows have a constant
thickness and a length proportional to the dipole, on a thinner skeleton, so that a short arrow or
one lying along a bond stays visible; there is no minimum length, but the values are written in
the upper left corner of the picture (|mu| in e.a0, |m| in Bohr magnetons), with ⊙ / ⊗ for an
arrow pointing towards / away from the viewer (within 30 degrees of the line of sight). In that
case the state is shown from a second point of view too, turned by the smallest angle (45 to 90
degrees) that puts every arrow at least 45 degrees from the line of sight. For a chiral molecule
(non-zero rotatory strengths) the magnetic transition dipole is drawn too, in magenta, oriented
so that the rotatory strength is positive when the two arrows make an acute angle (as the E-M
angle of Gaussian). One Bohr magneton is drawn as 4 bohr, like 1 e.bohr: since 1 Bohr magneton
= alpha/2 e.bohr in common units, the magnetic arrow is magnified 2/alpha = 274 times, and the
picture is quantitative: g = 4R/D = (4/274) x (projection of the magenta arrow on the blue one) /
(length of the blue one). The label of each chiral state gives f and g (from R(length)); the
rotatory strengths are in the table of the excited states (2 decimals, both gauges). m and
R(length) depend on the origin of the program, R(velocity) does not; when the two gauges give
opposite signs the label says "sign uncertain" after g. R(velocity)
is computed from the velocity and magnetic transition dipoles printed by the program (cclib reads
only R(length)), within 0.04 of the printed value for Gaussian and 0.01 for ORCA; for ORCA, cclib
needs the branch orca-magnetic-dipoles (velocity and magnetic dipoles; otherwise no magenta arrow
and no R(velocity)). With the Tamm-Dancoff approximation (TDA, the default of ORCA) the two gauges
can disagree strongly (HCFClBr: 3 states of 10 with opposite signs, S1 0.05 against 9.32), which the
"sign uncertain" mark shows; full TD-DFT (`tda false`) agrees much better, but ORCA then prints
only the weights of the excitations, hence no transition density pictures. The
transition dipole is the first moment of this density, and its square gives the oscillator
strength, f = 2/3 Delta E |mu|^2: a bright state has large regions of opposite signs on either
side of the molecule; a forbidden one has no arrow, the dipoles of its regions cancelling (e.g.
the n->pi* state of a ketone). The density is sum (X + Y) phi_i phi_a over the excitation (X)
and de-excitation (Y) amplitudes of the log; the arrow is the transition dipole printed in the
log (Gaussian, ORCA), and the density is given the sign that matches it (the sign of a
transition density is arbitrary). By default the programs print only the amplitudes X above
0.1 and no Y: the shape is then approximate, and the report lists the states whose density
gives a dipole more than 35 % away from that of the log. With Gaussian, `IOp(9/40=4)` in the TD
route prints all the amplitudes, X and Y, and the densities are exact (dipoles of the log found
within 2 %); with ORCA (TDA by default: no Y), `tprint 1e-6` in `%tddft`. The same picture is
drawn for the emitting state of an optimized excited state. The restricted triplets have no
charge transition density (spin-forbidden). Lambda uses the pairs of the main amplitudes
(|X| above 0.1 after cclib's normalization) with k = X + Y, normalized as in Peach et al.

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

**Fukui functions**: F+, F- and the dual descriptor side by side (one row each with two
points of view). Blue regions are the most electrophilic (F+) or nucleophilic (F-) sites;
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
V_S,max) are in bold. **Equivalent extrema** (near atoms that are equivalent in the molecular
graph, with values within 0.3 kcal/mol: the two lone pairs of a carbonyl, the H atoms of a methyl
group) share one number in the picture and one line in the table, marked ×n. An extremum is
reported when it is separated from any stronger, non-equivalent extremum by a saddle at least
0.5 kcal/mol deep (topological prominence, computed on the surface mesh). This leaves out the
"shoulders" on the flank of a stronger site and the small creases between the H atoms of a
methyl group, but keeps weak isolated sites (sigma-holes near 0 kcal/mol). On 146 calculations of
the laboratory (4661 extrema), this reports 16 % fewer extrema than the former criterion
(difference with the median of the MEP 1.5 to 2.5 bohr around), with no real site lost in the
cases examined; when none is left, the caption says so. Extrema on the hidden side of the molecule
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
