# Development plan

What is planned, roughly in order, with its state. Decisions are taken by Thomas Cauchy; the
reasons of the choices already made are written in the code where they apply. What QuChemReport
does today is in [capabilities.md](capabilities.md).

Last update: 27/11/2026.

## In progress

1. **Selection of the relevant excited states** (decided; calibrated on 27/11/2026: 7 to 11 states for
   acetone, CHFClBr, a dye and [6]helicene, the share of the bands hardly matters; 10 % and a cap
   of 15 proposed). Replace the current
   rule (first five states, plus f > 0.1 or R > 10) by the union of:
   - the lowest states of each multiplicity (S1-S3 and T1-T3), always interesting;
   - the states that make the bands of the broadened spectra: for each maximum of the absorption
     curve, and separately of the positive and negative CD bands, above a share of the largest
     one, the states that give most of the intensity at the maximum; bands cut by the last
     computed state are left out;
   - with spin-orbit couplings, the triplets near S1 that couple to it, and T1.

   A cap on the number of states, a reason shown for each state, and a list of states given by
   the user. The same selection for the density differences, NTO, transition densities and
   dipoles; the emission keeps its optimized state. The share of the bands and the cap are being
   calibrated on [6]helicene (40 singlets, 20 triplets), a push-pull dye, acetone and CHFClBr
   (10 states was found too small when absorption, CD, singlets and triplets add up).

## cclib (external)

2. **Pull requests**, to follow up:
   - #1972 core electrons with effective core potentials, and #1973 dispersion model: open,
     waiting for the maintainers;
   - XYZ writer of Gaussian optimizations (branch `xyz-writer-comments`, pushed to the fork):
     to open; QuChemReport no longer needs it (RDKit takes the coordinates directly since
     28/11/2026), but other users of cclib do;
   - ORCA transition dipoles, Gaussian de-excitation amplitudes, ORCA negative RPA weights, ORCA
     velocity and magnetic transition dipoles: prepared, to open after the answers on #1972 and
     #1973 (the last two add the same new test file and will need a rebase).
3. **ORCA functional**: let cclib expose the definition of the functional printed by ORCA
   (exchange, correlation, share of exact exchange), so that QuChemReport no longer reads it
   from the log itself.
4. **Spin-orbit couplings**: propose a format for them to cclib (issue #1419), and move the ORCA
   reader of QuChemReport there.

## Report

5. **Summary at the top of the report** (key numbers of the molecule and its excited states).
6. **Equivalent MEP extrema grouped** (by symmetry), with a topological prominence instead of
   the fixed shell threshold: decided by Thomas Cauchy on 27/11/2026 and done in the test copy
   (mep_surface: group_prominences, symmetry_classes, EQUIVALENT_TOL 0.3 kcal/mol; one line per
   group in the table). Study: validation/calculs_20261127/mep_topology (146 calculations, 4661
   extrema: 16 % fewer extrema reported, 3120 groups).
7. **Jablonski diagram**: relaxed states (optimized S1 and T1) and radiative arrows.
8. **Height of the picture grids** adapted to the length of their caption (a grid is now cut
   after 190 mm of pictures even when the page could hold it with its caption).

## Calculations

9. **Condensed Fukui functions and density differences per atomic basin** (Becke, Hirshfeld,
   Bader), with the grids and gradients of QMCkl. Studied on 27/11/2026 (prototype on the grid of
   QuChemReport: Voronoi, Becke, Hirshfeld with computed free atoms, Bader by steepest ascent):
   the Hirshfeld values match those printed by Gaussian and ORCA within 0.01; condensed Fukui
   functions do not depend on the grid step (0.25 to 0.10 bohr) and take about a second;
   condensed density differences give the charge moved per atom and fragment. Absolute charges
   would need the core treatment of the MEP. Decision pending.
10. **MEP on the molecular surface only**: studied on 24/11/2026, no gain worth it today;
    to reconsider together with the atomic grids of point 9 (it would save the memory of the
    Fourier transform for large molecules).
11. **TREXIO export** of the wavefunction (optional).

## Tests on real calculations

12. Cases still to cover: Gaussian triplet emission, ORCA emission end to end, range-separated
    and double-hybrid functionals with ORCA, Gaussian 16 outputs.

## Done recently

- 27/11/2026: transition densities and dipoles (electric and magnetic, to scale), rotatory
  strengths in both gauges and dissymmetry factor; exact density differences; arrows of constant
  thickness and second view along the line of sight; spectrum axes in cm-1, eV or nm with dark
  states shown; no CD for achiral molecules; no silent overwrite of a previous report; works with
  the cclib development version (crash of its XYZ writer avoided).
- 26/11/2026: report layout, open-shell references, configurable colours, POV-Ray rendering.
- 24/11/2026: MEP extrema, automatic camera views, spin-orbit coupling and phosphorescence (ORCA).
