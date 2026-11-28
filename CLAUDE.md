# Instructions for AI agents working on QuChemReport

Read `CONTRIBUTING.md` before writing any code: it is the editorial rule of this project and
it applies to you in full. The essentials:

1. **Never introduce a scientific error.** Decide the kind of every change before writing it:
   `[refactor]` (results unchanged), `[presentation]` (how results are shown), `[science]` (a
   value changes), `[calibration]` (a threshold or default chosen from data). One kind per
   commit. A `[refactor]` reproduces even the known errors (`docs/known_errors.md`).
2. **A `[science]` change needs proof that the old value is wrong**: an independent source
   (literature or analytical result, another program, the log itself), the size of the error
   in numbers, a failing-then-passing test on a real log. Output of your own new code is not evidence.
3. **Do not invent.** No reference, DOI, value or measurement you have not checked. Write
   `to be confirmed` and ask.
4. **In doubt about the chemistry or a convention, ask the maintainer; do not "fix".**
5. **One point at a time**, presented in detail (finding, evidence, change, checks, risk,
   rollback), with the maintainer's explicit approval before starting it.
6. **Respect the layers** of `docs/architecture/c4/` (`3_components_target.puml`): every number
   is computed in `data_enrichment`; logs are read through cclib (fix cclib upstream rather
   than reading the log directly); constants live in `utility_services/units.py`.
7. **Explain why** in comments, for a chemist and for a programmer; key numbers and dates in
   the code, the full study in `docs/decisions/` or `studies/`. Everything in English.
8. **Check** with the fast and slow tests and the reference reports, in the same environment
   and thread settings as the references. Do not regenerate a snapshot in the commit that
   changes the code it checks.
9. **Never** install in a shared installation, modify a shared environment, push, or open a
   pull request without a backup and the maintainer's approval.
