"""Models of the YAML configuration file (pydantic).

One class per section of the YAML file (see example.yaml for a commented example): Config is the
whole file, LogFile an entry of ``logfiles``, Output the ``output`` section and OutputInclude its
``include`` subsection, which selects the parts of the report. The validators check the values
when the Config object is built (YAML_loader.load_yaml_config): a wrong value stops the program
before any log file is read, with a message naming the option.
"""

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator, validator, root_validator
from typing import List, Optional, Tuple, Union, Dict
from pathlib import Path
import warnings
import os

SUPPORTED_SOLVERS = {
    "adf", "dalton", "firefly", "gamess", "gamess-uk", "gaussian",
    "jaguar", "molcas", "molpro", "mopac", "nbo", "nwchem",
    "orca", "psi4", "q-chem", "turbomole"
}

ALLOWED_CALCULATION_TYPES = {"sp", "opt", "opt_es", "freq", "freq_es", "td", "nmr"}

class LogFile(BaseModel):
    """One entry of ``logfiles``: a log file, the types of its calculations and its role.

    ``type`` is a type or a list of types among ALLOWED_CALCULATION_TYPES (a log file with
    several steps, e.g. an optimization followed by frequencies, has a list); ``reference``
    marks the log used as reference (exactly one); ``excited_state_number`` is required for
    optimized excited states (types ending in ``_es``) and forbidden otherwise.
    """
    path: Path
    type: List[str] # Single string accepted but converted into a list
    reference: Optional[bool] = False
    excited_state_number: Optional[int] = None

    @field_validator("path")
    def check_file_exists(cls, v: Path) -> Path:
        """Refuse a log file that does not exist."""
        if not v.is_file():
            raise FileNotFoundError(f"Log file does not exist: {v}")
        return v

    @field_validator("type", mode="before")
    def validate_type_values(cls, v) -> List[str]:
        """Accept a type or a list of types, all in ALLOWED_CALCULATION_TYPES."""
        if isinstance(v, str):
            v = [v]
        elif not isinstance(v, list):
            raise TypeError(f"type must be a str or list of str, got {type(v)}")
        
        if not v:
            raise ValueError("type must not be empty")

        invalid_types = [t for t in v if t not in ALLOWED_CALCULATION_TYPES]
        if invalid_types:
            raise ValueError(f"type contains invalid values: {invalid_types}. Allowed: {ALLOWED_CALCULATION_TYPES}")
        return v

    @model_validator(mode="after")
    def check_excited_state_number(self) -> "LogFile":
        """Require a positive excited_state_number for the _es types only."""
        types = [self.type] if isinstance(self.type, str) else self.type or []
        has_es_type = any("_es" in t for t in types)

        if has_es_type:
            if self.excited_state_number is None:
                raise ValueError("excited_state_number must be set when a type contains '_es'")
            if self.excited_state_number <= 0:
                raise ValueError("excited_state_number must be > 0")
        else:
            if self.excited_state_number is not None:
                raise ValueError("excited_state_number should only be set if type contains '_es'")
        return self

class QualityChecks(BaseModel):
    """``quality_control.checks``: conformity tests to run (quality_check_engine.conformity)."""
    formula: bool = True
    theory: bool = True
    nuclear_repulsion: bool = True
    charge: bool = True
    multiplicity: bool = True
    ground_state_optimization: bool = True

class QualityControl(BaseModel):
    """``quality_control`` section: consistency level (strict or lax) and tests to run."""
    method_consistency: str
    checks: QualityChecks

    @field_validator("method_consistency")
    def method_consistency_must_be_valid(cls, v: str) -> str:
        """Accept strict or lax."""
        allowed = {"strict", "lax"}
        if v not in allowed:
            raise ValueError(f"method_consistency must be 'strict' or 'lax'. Not '{v}' .")
        return v


class MoleculeInclude(BaseModel):
    """``output.include.molecule``: molecule picture, identification tags, atomic coordinates."""
    molecule_representation: bool
    molecule_identification: List[str]
    atomic_coordinates: bool

    @field_validator("molecule_identification", mode="before")
    def check_identification_values(cls, v):
        """Accept only the known identification tags."""
        allowed = {"directory name", "formula", "charge", "spin multiplicity", "monoisotopic mass", "inchi"}
        if not isinstance(v, list):
            raise TypeError("molecule_identification must be a list")
        for item in v:
            item_lower = item.lower()
            if item_lower not in allowed:
                raise ValueError(f"Invalid molecule_identification value: {item_lower}. Must be one of {allowed}")
        return v


class WavefunctionMOAnalysis(BaseModel):
    """``output.include.wavefunction.MO_analysis``: orbitals to show (``homo-1``, ``lumo+2``...),
    their energies and their pictures.
    """
    MO_list: List[str]
    energies: bool
    MO_diagrams: bool
    
    @model_validator(mode="after")
    def check_mo_list_if_needed(self) -> "WavefunctionMOAnalysis":
        """Require MO_list when energies or pictures are asked."""
        if (self.energies or self.MO_diagrams) and (not self.MO_list or len(self.MO_list) == 0):
            raise ValueError("MO_list must not be empty when energies or MO_diagrams is True")
        return self


class WavefunctionInclude(BaseModel):
    """``output.include.wavefunction``: orbitals, MEP map and population analysis."""
    MO_analysis: WavefunctionMOAnalysis
    MEP_maps: bool
    population_analysis: bool


class FrequenciesInclude(BaseModel):
    """``output.include.frequencies``: thermochemistry and IR spectrum."""
    thermo_data: bool
    IR_spectrum: bool


class ReactivityInclude(BaseModel):
    """``output.include.reactivity``: conceptual DFT indices and Fukui functions (they need the
    single points of the cation and anion at the reference geometry).
    """
    CDFT_indices: bool
    fukui_functions: bool


class ExcitedStatesSelection(BaseModel):
    """``output.include.excited_states.selection``: excited states shown in the tables and
    pictures: ``dominant_only``, the first ``first_n`` states, or ``all_states``.
    """
    mode: str
    first_n: Optional[int] = None

    @field_validator("mode")
    def mode_must_be_valid(cls, v: str) -> str:
        """Accept dominant_only, first_n or all_states."""
        allowed = {"dominant_only", "first_n", "all_states"}
        if v not in allowed:
            raise ValueError(f"mode must be one of {allowed}. Not '{v}' .")
        return v

    @model_validator(mode="after")
    def check_first_n_if_needed(self) -> "ExcitedStatesSelection":
        """Require a positive first_n in mode first_n."""
        if self.mode == "first_n":
            if self.first_n is None:
                raise ValueError("first_n must be set when mode is 'first_n'")
            if self.first_n <= 0:
                raise ValueError("first_n must be a positive integer")
        return self


class NTOInclude(BaseModel):
    """``output.include.excited_states.natural_transition_orbitals``: NTO pairs of the selected
    states (data_enrichment.nto). ``grid_step`` and ``nproc`` are kept for old YAML files
    but ignored: the NTO pictures use the common grid and resources.nproc.
    """
    enabled: bool = False
    max_pairs: int = Field(default=2, ge=1, le=20)
    min_weight: float = Field(default=0.01, ge=0, le=1)
    # Deprecated, ignored: the NTO images use the common grid (with
    # output.image_grid_reduction) and resources.nproc.
    grid_step: Optional[float] = Field(default=None, gt=0)
    nproc: Optional[int] = Field(default=None, ge=1)


class ExcitedStatesInclude(BaseModel):
    """``output.include.excited_states``: transitions, absorption and CD spectra, electron
    density differences and NTO of the vertical excited states (type td).
    """
    selection: ExcitedStatesSelection
    transitions: bool
    absorption_spectrum: bool
    CD_spectrum: bool
    electron_density_difference: bool
    natural_transition_orbitals: NTOInclude = Field(default_factory=NTOInclude)
    
    @model_validator(mode="after")
    def selection_required_if_transitions_or_edd(cls, values):
        """Require a selection for transitions or EDD."""
        selection = values.selection
        if (values.transitions or values.electron_density_difference) and not selection:
            raise ValueError(
                "A selection mode must be provided when transitions or "
                "electron_density_difference is enabled."
            )
        return values


class OptimizedExcitedStatesInclude(BaseModel):
    """``output.include.optimized_excited_states``: emission and CPL spectra, transitions and
    density differences of the optimized excited states (types opt_es, freq_es).
    """
    transitions: bool
    emission_spectrum: bool
    CPL_spectrum: bool
    electron_density_difference: bool


class NMRShifts(BaseModel):
    """NMR chemical shifts options (the NMR section of the report is not implemented)."""
    enabled: bool
    elements: List[str]


class NMRCouplings(BaseModel):
    """NMR coupling options (the NMR section of the report is not implemented)."""
    enabled: bool
    elements: List[str]


class NMRInclude(BaseModel):
    """``output.include.NMR`` (not read yet: the NMR section is not implemented)."""
    NMR_shifts: NMRShifts
    NMR_couplings: NMRCouplings

class OutputInclude(BaseModel):
    """``output.include``: the parts of the report. Each option also decides which computations
    are run (data_enrichment.compute_flags).
    """
    molecule: MoleculeInclude
    computational_details: bool
    wavefunction: WavefunctionInclude
    frequencies: FrequenciesInclude
    reactivity: ReactivityInclude
    excited_states: ExcitedStatesInclude
    optimized_excited_states: OptimizedExcitedStatesInclude
    # TODO : TO IMPLEMENT : NMR output is yet to be implemented
    # NMR: NMRInclude

class SurfaceColors(BaseModel):
    """Colours of the positive and negative parts of one kind of iso-surface (``output.colors.MO``,
    ``EDD``, ``fukui``, ``NTO`` or ``transition_density``), each "#RRGGBB", a colour name ("teal", "lightpink": the CSS
    names known to matplotlib) or [r, g, b] in 0-1; omitted: the default
    (utility_services.parameters.DEFAULT_SURFACE_COLORS). The captions name the colours:
    <sign>_name if given, else the word of the colour, else a name made from its value
    (parameters.color_name). Unknown keys are refused, so that a misspelt "postive" does not
    leave the default colour silently.
    """
    model_config = ConfigDict(extra="forbid")
    positive: Optional[Tuple[float, float, float]] = None
    negative: Optional[Tuple[float, float, float]] = None
    positive_name: Optional[str] = None
    negative_name: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def name_from_word(cls, values):
        """A colour given by a word names itself in the captions ("xkcd:" or "tab:" removed)."""
        if isinstance(values, dict):
            values = dict(values)
            for sign in ("positive", "negative"):
                v = values.get(sign)
                if isinstance(v, str) and not v.startswith("#") and not values.get(sign + "_name"):
                    values[sign + "_name"] = v.split(":")[-1]
        return values

    @field_validator("positive", "negative", mode="before")
    @classmethod
    def parse_color(cls, v):
        """(r, g, b) in 0-1 of "#RRGGBB", a colour name or [r, g, b]."""
        if v is None:
            return v
        from matplotlib.colors import to_rgb
        if isinstance(v, (list, tuple)) and (len(v) != 3 or not all(
                isinstance(x, (int, float)) and not isinstance(x, bool) and 0 <= x <= 1 for x in v)):
            raise ValueError(f"A colour list must be [r, g, b] with values in 0-1. Not {v} .")
        try:
            return tuple(float(x) for x in to_rgb(v))
        except (ValueError, TypeError):
            raise ValueError(f"Unknown colour '{v}': give \"#RRGGBB\", a colour name (e.g. teal) "
                             "or [r, g, b] in 0-1.")


class Colors(BaseModel):
    """``output.colors`` section: colours of the iso-surfaces of the 3D pictures (see
    SurfaceColors and docs/configuration.md)."""
    model_config = ConfigDict(extra="forbid")
    MO: SurfaceColors = Field(default_factory=SurfaceColors)
    EDD: SurfaceColors = Field(default_factory=SurfaceColors)
    fukui: SurfaceColors = Field(default_factory=SurfaceColors)
    NTO: SurfaceColors = Field(default_factory=SurfaceColors)
    transition_density: SurfaceColors = Field(default_factory=SurfaceColors)

    def as_dict(self):
        """kind -> sign -> ((r, g, b) or None, name or None), for parameters.set_surface_colors."""
        return {kind: {sign: (getattr(c, sign), getattr(c, sign + "_name"))
                       for sign in ("positive", "negative")}
                for kind, c in ((k, getattr(self, k)) for k in ("MO", "EDD", "fukui", "NTO", "transition_density"))}


class Output(BaseModel):
    """``output`` section: output folder and molecule name, format and verbosity of the report
    (full, si or text), size of the 3D pictures and grid reduction used to draw them, spectra,
    camera views and colours of the iso-surfaces.
    """
    output_path: str
    # folder chosen at launch (report_folder); not read from the YAML file
    _report_folder: Optional[str] = PrivateAttr(default=None)
    # Folder <output_path>/<molecule_designated_name> already holding a report: ask (in a
    # terminal; without one, stop), new (<name>_2...), overwrite (the files of quchemreport are
    # deleted first) or stop. See execution_engine/output_folder.py.
    if_exists: str = "ask"
    molecule_designated_name: str
    format: str # html | markdown | pdf # TODO : Output formats are subject to changes
    verbosity: str # full | si | text
    include: OutputInclude
    # Side of the (square) 3D images in pixels: 1200 px = 360 dpi in an 8.5 cm column,
    # 2000 px = 600 dpi. Larger images cost more rendering time.
    image_size: int = Field(default=1200, ge=200, le=8000)
    # The 3D images are drawn on every k-th point of the computation grid (k = 2:
    # 0.075 -> 0.15 bohr). Numerical results (MEP, Fukui, CT indices) keep the full grid.
    # Smooth surfaces: MO, MEP, NTO.
    image_grid_reduction: int = Field(default=2, ge=1, le=10)
    # Sensitive surfaces (density differences): EDD, Oif, Fukui, Fdual.
    image_grid_reduction_sensitive: int = Field(default=1, ge=1, le=10)
    # Main axis of the UV-visible, CD, emission and phosphorescence spectra: wavenumber (or
    # cm-1), energy (or eV), both linear in energy, or wavelength (or nm, as most experimental
    # spectra).
    spectrum_axis: str = "wavenumber"
    # Unit of the top axis of the spectra: auto (nm, or cm-1 when the main axis is in nm), a
    # unit as for spectrum_axis (different from it), or none.
    spectrum_secondary_axis: Optional[str] = "auto"
    # Range of the spectra on the main axis, in its unit (cm-1, eV or nm), e.g. [200, 600];
    # omitted: from the lowest computed state to the last visible band, with a margin.
    spectrum_range: Optional[List[float]] = None

    @field_validator("spectrum_range")
    def validate_spectrum_range(cls, v):
        """Two different positive numbers (any order)."""
        if v is None:
            return v
        if len(v) != 2 or min(v) <= 0 or v[0] == v[1]:
            raise ValueError(f"Output spectrum_range must be two different positive numbers. Not {v} .")
        return sorted(v)

    # Views of the 3D pictures: auto (a second view only when one is not enough, see
    # visualization_engine.camera), 1 (never a second view) or 2 (always, except for linear
    # molecules).
    camera_views: Union[int, str] = "auto"
    # Colours of the iso-surfaces (MO, EDD, Fukui, NTO), see Colors; omitted: the defaults
    colors: Colors = Field(default_factory=Colors)
    # Intensity of the specular reflections (white highlights) of the atoms, bonds and
    # iso-surfaces, 0-1; 0 = diffuse light only (the pictures before November 2026). Not on
    # the MEP surface. See visu_mayavi.SPECULAR.
    specular: float = Field(default=0.5, ge=0.0, le=1.0)
    # Renderer of the 3D pictures: vtk (Mayavi, default) or povray (ray-traced: anti-aliased
    # edges, soft shadows, ambient occlusion; about 4 s more per picture with 16 threads), see
    # visualization_engine/povray.py. povray_executable: path of POV-Ray when it is not in the
    # PATH nor in the conda environment. Without POV-Ray the pictures are rendered by VTK.
    renderer: str = "vtk"
    povray_executable: Optional[str] = None

    @field_validator("renderer")
    def validate_renderer(cls, v: str) -> str:
        """Accept vtk or povray."""
        v_lower = v.lower()
        if v_lower not in {"vtk", "povray"}:
            raise ValueError(f"Output renderer must be vtk or povray. Not '{v}' .")
        return v_lower

    @field_validator("camera_views", mode="before")
    def validate_camera_views(cls, v):
        """Accept auto, 1 or 2."""
        value = v.lower() if isinstance(v, str) else v
        if value in ("1", "2"):
            value = int(value)
        if isinstance(value, bool) or value not in ("auto", 1, 2):
            raise ValueError(f"Output camera_views must be auto, 1 or 2. Not '{v}' .")
        return value

    @field_validator("if_exists")
    def validate_if_exists(cls, v: str) -> str:
        """ask, new, overwrite or stop."""
        value = str(v).strip().lower()
        if value not in ("ask", "new", "overwrite", "stop"):
            raise ValueError(f"Output if_exists must be ask, new, overwrite or stop. Not '{v}' .")
        return value

    def report_folder(self) -> str:
        """Folder of the report, with the final separator: the one chosen at launch
        (execution_engine.output_folder, set_report_folder), else <output_path>/<name>."""
        folder = self._report_folder or os.path.join(self.output_path, self.molecule_designated_name)
        return folder if folder.endswith(os.sep) else folder + os.sep

    def set_report_folder(self, folder: str) -> None:
        """Folder chosen at launch (execution_engine.output_folder.resolve_output_folder)."""
        self._report_folder = folder

    @field_validator("spectrum_axis")
    def validate_spectrum_axis(cls, v: str) -> str:
        """wavenumber (cm-1), energy (eV) or wavelength (nm), returned as the unit name."""
        from quchemreport.visualization_engine.visu_plots import unit_name
        try:
            return unit_name(v)
        except ValueError:
            raise ValueError(f"Output spectrum_axis must be wavenumber (cm-1), energy (eV) or wavelength (nm). Not '{v}' .")

    @field_validator("spectrum_secondary_axis")
    def validate_spectrum_secondary_axis(cls, v) -> str:
        """auto, none, or a unit as for spectrum_axis (YAML null is none)."""
        if v is None or str(v).strip().lower() in ("none", "auto"):
            return "none" if v is None else str(v).strip().lower()
        from quchemreport.visualization_engine.visu_plots import unit_name
        try:
            return unit_name(v)
        except ValueError:
            raise ValueError(f"Output spectrum_secondary_axis must be auto, none, wavenumber (cm-1), energy (eV) "
                             f"or wavelength (nm). Not '{v}' .")

    @model_validator(mode="after")
    def check_spectrum_axes(self):
        """The top axis of the spectra must differ from the main one."""
        if self.spectrum_secondary_axis == self.spectrum_axis:
            raise ValueError(f"Output spectrum_secondary_axis must differ from spectrum_axis ({self.spectrum_axis}); "
                             "use none for no top axis.")
        return self

    @field_validator("format")
    def validate_format(cls, v: str) -> str:
        """Accept html, markdown or pdf (only pdf is produced for now)."""
        allowed = {"html", "markdown", "pdf"}
        v_lower = v.lower()
        if v_lower not in allowed:
            raise ValueError(f"Output format must be one of {allowed}. Not '{v_lower}' .")
        return v_lower

    @field_validator("verbosity")
    def validate_verbosity(cls, v: str) -> str:
        """Accept full, si or text."""
        allowed = {"full", "si", "text"}
        v_lower = v.lower()
        if v_lower not in allowed:
            raise ValueError(f"Output verbosity must be one of {allowed}. Not '{v_lower}' .")
        return v_lower


class LoggingSettings(BaseModel):
    """``logging`` section: console output and level (debug prints details and tracebacks)."""
    console: bool
    level: str

    @field_validator("level")
    def level_must_be_valid(cls, v: str) -> str:
        """Accept debug, info, warning or error."""
        allowed = {"debug", "info", "warning", "error"}
        if v not in allowed:
            raise ValueError(f"Logging level must be one of: {allowed}. Not '{v}' .")
        return v

class Resources(BaseModel):
    # Processes (orbkit) or threads (qmckl) for the grid computations.
    # None keeps the former default of 20.
    """``resources`` section: number of threads for the grid computations (``nproc``, 20 when
    omitted). ``memory`` and ``mayavi_headless`` are not used yet.
    """
    nproc: Optional[int] = Field(default=None, ge=1)
    memory: Optional[float] = None          # not used yet
    mayavi_headless: bool = True            # not used yet (offscreen rendering is always on)


class Config(BaseModel):
    """The whole YAML configuration: program of the log files (``common_solver``), log files,
    quality control, output, logging and resources.
    """
    common_solver: str
    logfiles: List[LogFile]
    quality_control: QualityControl
    output: Output
    logging: LoggingSettings
    resources: Resources = Field(default_factory=Resources)

    @field_validator("common_solver", mode="before")
    def normalize_and_validate_solver(cls, v: str) -> str:
        """Accept a program of SUPPORTED_SOLVERS (lower case)."""
        if not v:
            raise ValueError("common_solver must be specified")
        v_lower = v.lower()
        if v_lower not in SUPPORTED_SOLVERS:
            raise ValueError(f"common_solver '{v}' is not supported. Supported solvers: {SUPPORTED_SOLVERS}")
        return v_lower

    @field_validator("logfiles")
    def logfiles_must_not_be_empty(cls, v: List[LogFile]) -> List[LogFile]:
        """Require at least one log file."""
        if not v:
            raise ValueError("At least one logfile must be specified")
        return v
