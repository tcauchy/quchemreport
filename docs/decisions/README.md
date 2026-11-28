# Decision records

One file per decision that is not obvious from the code: `<yyyy-mm-dd>-<topic>.md`. The code
keeps the decision, the key number and the date in a short comment, with a link to the file.

A record is never rewritten. When a decision changes, a new record says which one it
replaces, and the old one gets a line `Superseded by <file>` at the top.

## Template

```markdown
# <Title>

- Date: yyyy-mm-dd
- Status: accepted | superseded by <file>
- Kind: presentation | science | calibration (see CONTRIBUTING.md)
- Code: <module.function>, <constant>
- Study: studies/<folder> (if a calibration)

## Problem
What was observed, on which molecules or logs.

## Options
Each option considered, with its measurements.

## Decision
What was chosen, by whom, and the key numbers.

## When to reconsider
What would make this choice wrong (new program, new kind of molecule...).

## References
Literature, with DOI and verification status.
```

## Records

*(none yet: the decisions already taken are described in the code comments; moving their long
descriptions here is a step of the review plan)*
