# Contributing to QuChemReport

QuChemReport was designed and first written by chemists, then extended by computer scientists.
Both read its code. This page gives the rules for every change, whether it is written by a
person or by an AI agent (agents also read `CLAUDE.md`, which points here).

The rules have one purpose: **improve the code without ever introducing a scientific error.**
A report that looks better but gives a wrong energy is worse than the old one.

Status: draft of 2026-11-28, awaiting the maintainer's approval. Items marked *(to be
written)* or *(to be set)* are decided but not done yet.

---

## 1. Never change a result by accident

### 1.1 Four kinds of change

Before writing a change, decide which kind it is. The kind is the first word of the commit
message, in brackets: `[refactor] Split generate_report.py into sections`.

| Kind | What changes | Example | What you must show |
|---|---|---|---|
| `[refactor]` | Nothing in the results: the code is reorganized, renamed, commented, documented | Split a 3000-line file into modules | The reference reports come out identical (section 1.3) |
| `[presentation]` | How the results are shown, not their values | Put S1 before T1 in a table; change a caption | The list of expected differences, written before the change, then checked |
| `[science]` | A value: a number, a spectrum, a grid, what a picture shows | Use the TD-DFT energy, not the SCF one, for an optimized excited state | Proof that the old value is wrong (section 1.2), and the maintainer's approval |
| `[calibration]` | A threshold or a default chosen from data | The colour scale of the MEP, ±0.05 a.u. | The study that chose it (section 4), and the maintainer's approval |

- **One kind per commit.** A `[refactor]` that "also fixes" a value is two commits: first the
  refactor, then the fix.
- **The results to reproduce are those of the last version the maintainer approved**, not those
  of the original code. Fixes approved since then are part of them.
- **A `[refactor]` reproduces the known errors too, exactly.** Even an obvious error is not
  fixed in passing. Known errors are listed in `docs/known_errors.md`, each with the
  `[science]` change that will fix it, so that none of them stays forever.
- **A convention is not an error.** The sign of a density difference (excited − ground or the
  reverse), the units shown, which state is called S1: changing one of these is a
  `[presentation]` or `[science]` change, and always the maintainer's decision.

### 1.2 Proving that a value is wrong (`[science]`)

The new value must be compared with a source that does not depend on the code being changed.
Use the best source available, in this order:

1. an analytical result, or a published value or equation (with its reference, section 3.4);
2. another program computing the same quantity (for example Gaussian `cubegen` for the MEP);
3. the value the quantum chemistry program prints in the log itself (for example `SCF Done`).

The output of the new code never proves itself right.

Before the change is made, also write down:

- how large the error is, in numbers (for example 64.5 mEh, that is 40 kcal/mol);
- which reports change, and which do not;
- a test on a real log stored in `tests/data`, which fails before the change and passes after,
  with the expected value taken from the independent source.

When two sources disagree (the log against the literature, cclib against the log...), change
nothing and ask the maintainer. Libraries can be wrong too: orbkit's g functions, cclib's
Gaussian de-excitation amplitudes before November 2026.

### 1.3 Checking that results are identical (`[refactor]`)

- **Numbers.** The report rounds its numbers, so it can hide a change in the 5th significant
  digit. The values passed between the stages (`LogData`, `ComputedData`) are saved to a file
  and compared with a stated relative tolerance *(tool and tolerance to be set at step 0 of the
  review plan)*. The numbers written in the report source must be strictly identical.
- **Pictures.** Pictures rendered by VTK must be identical to the pixel. Pictures rendered by
  POV-Ray may differ slightly, because its soft light is random: the tolerance is the one
  **measured** by running the same code twice (number of different pixels, largest
  difference), written next to the reference pictures. "A small difference" is not accepted.
- **Spectra written as text files:** identical.
- **Same environment, same settings.** Results depend on the libraries (section 5) and on the
  number of threads (sums done in parallel change the last digits). The references record
  both; a comparison counts only with the same environment, `resources.nproc` and thread
  variables.
- **Cases covered.** The reference reports (azulene, acetone with Gaussian and ORCA, H2CO) do
  not go through every part of the code: open shells, ECPs, ORCA spin-orbit coupling, ions
  and Fukui functions, the `si` and `text` verbosities... Each change lists the parts of the
  code it touches and shows that a reference or a test goes through them; otherwise a case is
  added **first**, with the old code. The table of cases × features is kept in
  `tests/README.md`.
- Code that computes grids also reports the run time and memory of the azulene report: a
  change that keeps the results but doubles the time must say so.

### 1.4 What may change during the refactoring

Decision of 2026-11-28: during the current refactoring, **nothing is frozen**. Anything may
change when it is a clear improvement; there are few users and they will adapt.

- **Python functions inside QuChemReport** are not used by anyone else's scripts: they are
  renamed or moved freely, without keeping the old name alongside.
- **What users see** may change too: the command line (`cli.py` options), the YAML keys and
  their meaning, the content of the report, the names of the files in the report folder
  (pictures and spectra are reused in talks and papers). But each such change is a change of
  its own (`[presentation]`, `[science]` or `[calibration]`), approved by the maintainer and
  explained in the user documentation (`docs/configuration.md`, `docs/report.md`). A renamed
  YAML key gives a clear error message, never a silent default.
- Once the refactoring is done, the maintainer decides what becomes stable, and this section
  is rewritten as a promise to the users (it will define the version numbers).

---

## 2. Keep each kind of code in its place

The structure of the program is drawn in `docs/architecture/c4/` (PlantUML sources, SVG
pictures next to them) and explained in `docs/architecture.md`.

- **Order of the stages.** `utility_services` (the data passed between stages, units,
  geometry, names) uses nothing else from QuChemReport. Each stage uses only
  `utility_services` and the stages before it: `config` → `parser_engine` →
  `quality_check_engine` → `data_enrichment` → `visualization_engine` →
  `report_generator`. `execution_engine` runs them in this order.
- **One place for each kind of code.**
  - Reading a log: `parser_engine`, through cclib. If cclib cannot read something, fix cclib
    and propose the fix to its developers; reading the log directly is a temporary workaround,
    marked as such with the cclib pull request it waits for.
  - Every number shown in the report is computed in `data_enrichment`. The pictures draw and
    the report formats; neither computes.
  - Physical constants and unit conversions: `utility_services/units.py` only.
- **Data goes from one stage to the next as objects** (`LogData`, `ValidatedData`,
  `ComputedData` in `utility_services`), never through global variables.
- A change that moves the code away from the target drawing (`3_components_target.puml`) is
  discussed first. A change of the structure updates the drawings in the same commit.

---

## 3. Write for two readers

The code is read by chemists who are not programmers and by programmers who are not chemists.

### 3.1 Docstrings

A first sentence saying what the function does, then NumPy-style `Parameters`, `Returns`,
`Raises` (and `References`, section 3.4) for every function used outside its module. Give the
**units** of every input and output and the **conventions**: spin, sign, indices counted from
0 or from 1, which geometry.

### 3.2 Comments: why, not what

- Do not repeat the code in words (`# Skip if no data` above `if data is None: continue`).
- Explain, where the choice is made:
  - for the programmer, the chemistry (why the density of the cation is subtracted);
  - for the chemist, the programming (why a cache, why a copy of the array);
  - what was decided, the key measurement and its date, the options rejected, and when the
    choice should be looked at again.
- Keep such a comment to 3-6 lines; the full story (tables, figures, scripts) goes in a
  decision record, `docs/decisions/<yyyy-mm-dd>-<topic>.md`, linked from the comment.
- **Comments get old.** Every change rereads the comments, docstrings and documentation around
  the code it touches, and corrects those that became false.

### 3.3 Language

Everything in the repository is in English: code, comments, docstrings, documentation, commit
messages. French comments of the original authors are translated; the original is kept once,
below the translation (`# (original: ...)`), so that no nuance is lost.

### 3.4 Scientific references

- A method taken from the literature cites it in the docstring, in a `References` section:
  authors, journal, year, volume, first page, **DOI**, and the equation used. If the code
  differs from the paper, say how.
- The full list is in `docs/references.md`.
- Each reference says how far it was checked:
  - `DOI checked`: the DOI leads to this paper;
  - `equation checked against <source>`: the code was compared with the paper;
  - `to be confirmed`: not checked yet (for example, the paper was not accessible).
- **Never invent a reference.** An agent that cannot check one writes `to be confirmed` and
  asks; the maintainer can provide the paper.

---

## 4. A threshold keeps the study that chose it

A threshold or default chosen from data (MEP colour scale, minimum prominence of the MEP
extrema, minimum share of a band when selecting the excited states, camera criteria,
transparency...) is a `[calibration]` change.

- The study is kept in `studies/<yyyy-mm-dd>-<topic>/`: the scripts, the list of the logs used
  (paths and SHA-256 checksums, since the logs themselves are often private or too large), the
  summarized results, and a `README.md` saying how to rerun it and what was decided.
- The code gives the decision, the key number and a link to the study.
- Changing the value needs the original logs or new ones, rerun through the same study.

---

## 5. Environment

Results depend on the libraries as much as on the code: cclib (a development version with
fixes not yet released), QMCkl, VTK/Mayavi, POV-Ray, Open Babel, LaTeX, NumPy and its linear
algebra library.

- Reference reports record the library versions (and cclib's commit), `resources.nproc` and
  the thread variables (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`).
- After a change of the environment, the references are made again and compared with the
  previous ones, before any change of the code is checked against them.
- The installation guide (`docs/running.md`, "Installing elsewhere") must let someone install
  QuChemReport on another machine and get the reference results within the tolerances of
  section 1.3 *(to be written: a file listing the exact environment, and a script that checks
  it)*.

---

## 6. Code style

The code is brought to standard Python practice **little by little**: only in code changed
for another reason, or in a `[refactor]` commit made for that purpose. Never in the middle of
a `[science]` change.

- PEP 8 (the Python style guide), checked by the `ruff` tool with its settings stored in the
  repository.
- Type annotations on every new or changed function signature.
- NumPy-style docstrings (section 3.1).
- No new global variable that the code modifies; settings are passed as arguments.
- Errors: raise an exception, caught by `cli.py`; no `sys.exit` outside `cli.py`.
- Names in English, `snake_case`; no function of the program named `test_*` (that name is for
  tests).
- Comments written by the original authors are kept (translated if needed), even when they
  look obvious.

---

## 7. How a change is made

- **One point at a time.** Each point is presented before anything is changed: what was found,
  the evidence, the exact change, how it will be checked, what could change in the reports,
  how to undo it. Work starts only after the maintainer's explicit approval. Two points are
  never done in a row without approval.
- **Changes small enough to be read.** Moving code and changing code are never in the same
  commit: a move is checked mechanically (`git diff --color-moved`, and the reference reports
  must be identical), and the change that follows stays small enough to be read line by line.
- **Checks for every commit:** fast tests, slow tests (`tests/run_tests.sh all`), the
  reference reports compared as in section 1.3.
- **Tests.** Expected values come from an independent source (section 1.2). A saved output
  used as a reference (report source, picture checksums, saved values) is never updated in the
  same commit as the code it checks, unless the difference is shown and approved.
- **Shared installation.** Nothing is installed in the shared installation or environment,
  and nothing is pushed, without a backup and the maintainer's approval.
- **In doubt about the chemistry, ask; do not "fix".** Code that looks odd may follow a
  convention or a calibrated choice.
- Commit messages: in English, starting with the kind of change, the reason in the body, and
  the attribution lines if an AI agent wrote the change.

---

## Pages referred to

- `CLAUDE.md`: short instructions for AI agents.
- `docs/architecture.md`, `docs/architecture/c4/`: structure of the program and its drawings.
- `docs/decisions/`: decision records.
- `docs/known_errors.md`: known errors waiting for their `[science]` fix.
- `docs/references.md`: bibliography *(to be written)*.
- `studies/`: calibration studies *(to be gathered from the validation folders)*.
- `tests/README.md`: tests and the table of cases × features.
