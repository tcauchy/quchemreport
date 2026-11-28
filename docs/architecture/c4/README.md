# C4 diagrams of QuChemReport

The architecture drawn with the [C4 model](https://c4model.com) (context, containers,
components), in PlantUML text files with the C4-PlantUML library, so that they can be edited
and compared with `git diff`. The rendered SVG files are versioned next to the sources.

| File | Shows |
|---|---|
| `common.puml` | Every element's name and description, defined once and used by all diagrams |
| `1_system_context.puml` | The user, QuChemReport and the external software |
| `2_containers.puml` | The report application (one process) and the files it reads and writes |
| `3_components_current.puml` | The stages as the code is today, with the dependencies against the layer order |
| `3_components_target.puml` | The stages as the refactoring aims to make them |
| `4_dynamic_pipeline.puml` | The order of the stages for one report and where they stop |

The refactoring plan is the path from `3_components_current` to `3_components_target`; when a
step is done, the current diagram is updated in the same commit.

These diagrams replace the PDF diagrams of the first version (system context, container, and
two component diagrams), whose sources were lost. Main differences: orbkit replaced by QMCkl,
POV-Ray, Open Babel, LaTeX and the Gaussian/ORCA programs added, the two former "containers"
shown as components of one process, the data passed between stages on the arrows, no logging
container and no docx export (neither exists).

Conventions (2026-11-28):
- The external software is grouped by theme (reading the logs, identity of the molecule,
  wavefunction on grids, pictures, report document), one arrow per use.
- orbkit is no longer drawn: QMCkl replaces it (faster, reliable, maintained); the remaining
  orbkit code is removed by a separate change.
- Open Babel is drawn as "to be replaced": RDKit takes over its work (bonds, InChI, SMILES,
  mass, equivalent atoms). Remove it from the diagrams in the commit that removes it from the
  code.
- Files are drawn as plain boxes, not as the C4 "database" cylinder: QuChemReport has no
  database.

## Rendering

Needs Java and the PlantUML jar (1.2024.8 was used; it includes the C4 library, and its
`smetana` layout engine replaces Graphviz):

    java -Djava.awt.headless=true -jar plantuml-1.2024.8.jar -tsvg docs/architecture/c4/[1-4]_*.puml

The jar is available from https://github.com/plantuml/plantuml/releases.
