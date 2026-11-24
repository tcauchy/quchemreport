# QuChemReport documentation

QuChemReport turns the output files of quantum chemistry calculations on one molecule
(Gaussian, ORCA) into a PDF report: computational details, orbitals, electrostatic potential,
reactivity, excited states and their spectra, with consistency checks between the
calculations.

For users:

1. [Preparing the calculations](calculations.md): what to ask Gaussian and ORCA to print, with
   example inputs.
2. [Running QuChemReport](running.md): command line, batch jobs, output files.
3. [The YAML configuration file](configuration.md): every option.
4. [The report](report.md): content of the sections and how to read the pictures.
5. [Troubleshooting](troubleshooting.md): messages and what to do.

For developers:

- [Architecture](architecture.md): stages, data models, conventions, tests.
- The code itself: every module and function has a docstring (NumPy style), and the reasons of
  the non-obvious choices are written where they are made.
