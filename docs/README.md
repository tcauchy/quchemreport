# QuChemReport documentation

QuChemReport turns the output files of quantum chemistry calculations on one molecule
(Gaussian, ORCA) into a PDF report: computational details, orbitals, electrostatic potential,
reactivity, excited states and their spectra, with consistency checks between the
calculations.

For users:

1. [Capabilities and limits](capabilities.md): what QuChemReport does, and what it does not do.
2. [Preparing the calculations](calculations.md): what to ask Gaussian and ORCA to print, with
   example inputs.
3. [Running QuChemReport](running.md): command line, batch jobs, output files.
4. [The YAML configuration file](configuration.md): every option.
5. [The report](report.md): content of the sections and how to read the pictures.
6. [Troubleshooting](troubleshooting.md): messages and what to do.

For developers:

- [Development plan](roadmap.md): what is planned, with its state.
- [Architecture](architecture.md): stages, data models, conventions, tests.
- The code itself: every module and function has a docstring (NumPy style), and the reasons of
  the non-obvious choices are written where they are made.
