# Troubleshooting

QuChemReport stops with an `ERROR` when the set of log files cannot give a consistent report,
and goes on with a `Warning` when only one output cannot be produced. The messages below are
those printed in the terminal.

## The program stops

**`Validation Error | ('output', ...): ...`** A value of the YAML file is wrong: the message gives
the option and the allowed values ([The YAML configuration file](configuration.md)).

**`Log file does not exist: ...`** Wrong `path` in `logfiles` (paths are relative to the folder
where you run the command).

**`No completed gaussian calculation found in '...': the termination marker 'Normal termination'
is missing`** (ORCA: `Timings for individual modules:`). The calculation did not finish: look at
the end of the log. An interrupted last step of a multi-step log only gives the warning
`the last step of ... has no ... (interrupted job?); it is ignored`.

**`Reading of '...' (step k/n) failed in cclib or Open Babel`** The log could not be read. Check
that it is a complete output of the program declared in `common_solver`.

**`Too many different formulas`** The log files are not all of the same molecule.

**`... with the same level of theory`** The functionals or the basis sets differ. The functional
includes the dispersion correction: a B3LYP and a B3LYP-D3(BJ) calculation are not mixed. The
detected values are printed above the message.

**`... unique conformer without counting the optimized excited states`** The ground state
calculations are not at the same geometry. Start the frequencies, single points and TD from
the optimized geometry (Gaussian `geom=check`, ORCA `* xyzfile`).

**`... same charge without counting the single points`**, **`... same electronic state`**
Calculations other than the `sp` ones have different charges or multiplicities. Declare the
cation and anion as `type: sp`.

**`all the log files are single points (type sp)`** The report needs the ground state
optimization as reference.

**`one OPT or FREQ of a Ground State should be provided`** Add the optimization or the
frequency calculation.

**`No reference log file found`** / **`Multiple reference log files found`** Mark exactly one
entry with `reference: true`.

**`Probably a LateX error.`** The PDF was not produced. Run again with `logging.level: debug`
to see the output of pdflatex.

## An output is missing

**`no log file with MO coefficients and basis set matches the reference ... no MO, MEP, Fukui or
EDD pictures`** The reference, and every log at its geometry, lack the orbitals: add
`pop=full gfprint` (Gaussian) or `Print[ P_Basis ] 2` and `Print[ P_MOs ] 1` (ORCA)
([Preparing the calculations](calculations.md)). The logs must have the same geometry in the same
orientation.

**`MO data of ... could not be read`** The orbitals of this log are incomplete; its pictures are
skipped.

**`MEP map skipped: ...`** The reason follows: no orbitals, or an electron count inconsistent
with the effective core potentials. **`This might be caused by the MEP Map computation exceeding
available memory`**: the MEP of large molecules needs a lot of memory; run the report on a
node with more memory, or disable `MEP_maps`.

**`Fukui functions skipped`** The single points are missing, or have no orbitals, or have an
electron count problem (given in the message).

**`no excitation amplitudes in the log for the state(s) ...`** The program did not print the
excitations of these states: no density difference nor charge-transfer indices for them.

**`NTO skipped for ...: TD and reference orbitals/geometries are incompatible`** The TD
calculation does not use the orbitals of the reference (other geometry, orientation,
functional, basis set...). Run the TD from the checkpoint of the reference (Gaussian
`geom=check guess=read`) or print the orbitals in the TD calculation.

**`printed X amplitudes are incomplete`** The NTO weights and shapes are approximate: with
Gaussian, add `IOp(9/40=4)` to the TD route.

**`excited_state_number ... is not among its states`** The `excited_state_number` of an `opt_es`
entry does not match the states of the log.

**`some labels are invalid for ...`** An orbital of `MO_list` does not exist (e.g. `homo+1`,
`lumo-1`, or beyond the printed orbitals); the others are kept.

## ORCA stops in `MPI_Init`

Not a QuChemReport problem, but frequent: an ORCA job with `%pal` that stops at once with
`MPI_Init ... NULL communicator` and `[mpiexec@...] HYDT_...` lines uses the `mpirun` of an
MPICH installed in a conda environment. Add `export PATH=/usr/bin:$PATH` to the job script
before calling ORCA.
