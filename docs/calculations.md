# Preparing the calculations

QuChemReport reads the output of Gaussian (09 and 16) and ORCA (5 and 6). It can only show
what the programs print, and the pictures (orbitals, MEP, Fukui functions, density
differences, NTO) are rebuilt from the basis set and the molecular orbital coefficients: these
must be printed in the output. This page says what to ask for.

Complete example inputs for acetone, with every type of calculation, are in
`examples/acetone/gaussian` and `examples/acetone/orca` (`jobs.txt` gives the order and the
checkpoint each job starts from).

## A typical series

| Calculation | YAML type | Needed for |
|---|---|---|
| Ground state optimization | `opt` (reference) | everything; orbitals and MEP pictures |
| Frequencies | `freq` (or `[opt, freq]` in one file) | thermochemistry, IR, check of the minimum |
| Cation and anion at the optimized geometry | `sp` | conceptual DFT indices, Fukui functions |
| Vertical excited states, singlets and triplets | `td` | absorption and CD spectra, transitions, density differences, NTO |
| Optimized excited states (S1, T1) | `opt_es` + `excited_state_number` | emission and CPL spectra, density difference of the emitting state |

All the calculations must use the same functional (with the same dispersion correction) and
basis set: QuChemReport checks it and stops otherwise.

## Gaussian

Add to the route of every calculation whose orbitals are used (the optimization, the single
points, the TD and excited state optimizations):

    pop=full gfprint

`gfprint` prints the basis set and `pop=full` all the MO coefficients. Without them, the
numerical results are still reported but the corresponding pictures are skipped, with a
warning.

Start the other calculations from the checkpoint of the optimization, so that they use the
same geometry **in the same orientation** (the grids of one calculation are drawn with the atoms
of the reference):

    %Chk=TD                                  (copy of OPT.chk)
    #p PBE1PBE/6-31G(d) pop=full gfprint td=(nstates=10) geom=check guess=read

- Single points of the ions: `geom=check`, with charge and multiplicity `1 2` / `-1 2` for a
  neutral closed-shell molecule.
- Triplets: `td=(triplets,nstates=10)`.
- Excited state optimization: `td=(nstates=10,root=1) opt geom=check guess=read`, starting
  from the TD checkpoint; give `excited_state_number: 1` in the YAML file.
- All the excitation and de-excitation amplitudes (exact transition densities, better NTO):
  add `IOp(9/40=4)` to the TD route. The log is larger (about 400 amplitudes per state for a
  five-atom molecule in 6-31G(d)); the report time hardly changes.
- Empirical dispersion (`EmpiricalDispersion=GD3BJ`...) must be the same in all the
  calculations.

Effective core potentials (LANL2DZ, def2 basis sets on heavy atoms, `GenECP`) are supported.

## ORCA

Print the basis set and the orbitals in every calculation:

    %output
     Print[ P_Basis ] 2
     Print[ P_MOs ] 1
    end

Read the optimized geometry from the `.xyz` file written by the optimization, so that all the
calculations share it:

    ! PBE0 def2-SVP
    %tddft
     nroots 10
     printlevel 3
    end
    * xyzfile 0 1 OPT_FREQ.xyz

- Optimization and frequencies can be one job: `! PBE0 def2-SVP Opt Freq` (YAML type
  `[opt, freq]`).
- All the excitation amplitudes (exact transition densities with the default TDA, better
  NTO): `tprint 1e-6` in `%tddft`.
- Triplets: `triplets true` in `%tddft`; excited state optimization: `iroot 1` (and
  `irootmult triplet` for T1) with `Opt`.
- Spin-orbit coupling (SOC): `dosoc true` in `%tddft` (it implies the triplets). In a TD
  calculation, the report gives the SOC matrix elements between the singlets and the
  triplets and a Jablonski diagram. In the T1 optimization (`iroot 1`, `irootmult triplet`,
  `dosoc true`, YAML type `opt_es`), it also gives the phosphorescence: the three spin-orbit
  sublevels of T1, their radiative rates, the radiative lifetime and the spectrum. The
  geometry is optimized without SOC (perturbative SOC, computed again at each step; the
  last one, at the final geometry, is used). Gaussian does not compute these couplings.
- Parallel runs (`%pal nprocs 4 end`): ORCA calls `mpirun`, which must be the OpenMPI of the
  system. If a conda environment of yours provides MPICH, ORCA stops in `MPI_Init`; add
  `export PATH=/usr/bin:$PATH` to your job script before calling ORCA.

## What is not needed

- The reference does not need to be a separate file: the first step of an `[opt, freq]` file
  with the MO coefficients is used.
- Frequencies, single points and excited states do not need `pop=full gfprint` for their
  numerical results (energies, spectra, conceptual DFT indices). The Fukui pictures need the
  orbitals of the single points. For a TD calculation without orbitals, the orbitals of the
  reference are used for its density differences and NTO: fine with Gaussian and
  `guess=read`, approximate with ORCA, which redoes the SCF (print the orbitals there). An
  excited state optimization without orbitals gives an approximate emission density
  difference, with a warning.
