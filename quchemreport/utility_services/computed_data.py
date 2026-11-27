"""ComputedData: everything the pictures and the report need (dataclasses).

data_enrichment.compute.compute_data fills one ComputedData for the report. It mirrors the
LogData sections (``molecule``, ``comp_details``, ``results``) with, for most fields, a list of
one value per parsed model, in the order of the models; ``metadata`` gives the log files and the
reference; ``derived`` holds what is computed for the report (grids, spectra, selections,
indices), most of it for the reference only.

init_computed_data(n) builds an empty ComputedData for n models: list fields become
``[None] * n`` and the other fields None.
"""

from dataclasses import dataclass, fields, is_dataclass
from typing import List, Optional, Any, Tuple, Union, get_origin, get_args
import inspect

@dataclass
class Metadata:
    """Log files of the models, index of the reference model, output folder, and index of the
    YAML entry of each model (several models per multi-step log file).
    """
    log_files: List[str]
    ref_log_file_idx: int
    output_dir: str
    # YAML logfiles index of each model (several models per log file for multi-step logs)
    config_indexes: List[Optional[int]]


@dataclass
class MoleculeConnectivity:
    """Bonds of each model (atom index pairs, both directions)."""
    atom_pairs: List[Optional[List[Tuple[int, int]]]]
    # bond_orders: List[Optional[List[int]]]


@dataclass
class Molecule:
    """Charges, multiplicities and atoms of each model; identity of the molecule."""
    charges: List[Optional[int]]
    multiplicity: List[Optional[int]]
    atoms_Z: List[Optional[List[int]]]
    monoisotopic_mass: Optional[float]
    inchi: Optional[str]
    smi: Optional[str]
    connectivity: MoleculeConnectivity
    formula: Optional[str]


@dataclass
class GeneralCompDetails:
    """Computational details of each model (see log_data.CompDetailsGeneral)."""
    job_type: List[Optional[List[str]]]
    package: Optional[str]
    package_version: List[Optional[str]]
    last_theory: List[Optional[str]]
    functional: List[Optional[str]]
    dispersion: List[Optional[str]]
    basis_set_name: List[Optional[str]]
    basis_set_size: List[Optional[str]]
    is_closed_shell: List[Optional[bool]]
    integration_grid: List[Optional[str]]
    solvent: List[Optional[str]]
    scf_targets: List[Optional[List[List[float]]]]


@dataclass
class GeometryCompDetails:
    """Optimization targets of each model."""
    geometric_targets: List[Optional[List[float]]]


@dataclass
class FrequencyCompDetails:
    """Frequency options of each model."""
    temperature: List[Optional[float]]
    anharmonicity: List[Optional[List[List[float]]]]


@dataclass
class ExcitedStatesCompDetails:
    """Number of excited states of each model."""
    nb_et_states: List[Optional[int]]


@dataclass
class ComputationDetails:
    """``comp_details``: general, geometry, frequency and excited state details."""
    general: GeneralCompDetails
    geometry: GeometryCompDetails
    freq: FrequencyCompDetails
    excited_states: ExcitedStatesCompDetails


@dataclass
class GeometryResults:
    """Coordinates (Angstrom), nuclear repulsion energies and optimization data of each model."""
    elements_3D_coords: List[Optional[List[float]]]
    geometric_values: List[Optional[List[List[float]]]]
    nuclear_repulsion_energy_from_xyz: List[Optional[float]]
    OPT_DONE: List[Optional[bool]]


@dataclass
class FrequencyResults:
    """Frequencies, intensities, symmetries and thermochemistry of each model."""
    vibrational_int: List[Optional[List[float]]]
    vibrational_freq: List[Optional[List[float]]]
    vibrational_sym: List[Optional[List[str]]]
    zero_point_energy: List[Optional[float]]
    electronic_thermal_energy: List[Optional[float]]
    enthalpy: List[Optional[float]]
    free_energy: List[Optional[float]]
    entropy: List[Optional[float]]


@dataclass
class ExcitedStatesResults:
    """Excited state energies, strengths, amplitudes and dipoles of each model."""
    et_sym: List[Optional[List[str]]]
    # <S^2> of the states of an open-shell reference (None for closed shells)
    et_s2: List[Optional[List[float]]]
    et_energies: List[Optional[List[float]]]
    et_oscs: List[Optional[List[float]]]
    et_rot: List[Optional[List[Any]]]
    et_transitions: List[Optional[List[Any]]]
    # de-excitation amplitudes Y (occupied, virtual, Y) when printed (scanlog.split_deexcitations)
    et_deexcitations: List[Optional[List[Any]]]
    et_eldips: List[Optional[List[List[float]]]]
    et_veldips: List[Optional[List[List[float]]]]
    et_magdips: List[Optional[List[List[float]]]]
    # Spin-orbit coupling results (ORCA, parser_engine.orca_soc), per model
    soc: List[Optional[Any]]


@dataclass
class WavefunctionResults:
    """Total energies, HOMO indexes and MO energies of each model; atomic charges and
    conceptual DFT indices (global indices and condensed Fukui functions) of the reference.
    """
    total_molecular_energy: List[Optional[float]]
    homo_indexes: List[Optional[List[int]]]
    MO_energies: List[Optional[List[List[float]]]]
    Mulliken_partial_charges: Optional[List[float]]
    Hirshfeld_partial_charges: Optional[List[float]]
    CM5_partial_charges: Optional[List[float]]
    moments: List[Optional[List[List[float]]]]
    A: Optional[float]
    I: Optional[float]
    Khi: Optional[float]
    Eta: Optional[float]
    Omega: Optional[float]
    DeltaN: Optional[float]
    fplus_lambda_hirshfeld: Optional[List[float]]
    fminus_lambda_hirshfeld: Optional[List[float]]
    fdual_lambda_hirshfeld: Optional[List[float]]
    fplus_lambda_mulliken: Optional[List[float]]
    fminus_lambda_mulliken: Optional[List[float]]
    fdual_lambda_mulliken: Optional[List[float]]


@dataclass
class Results:
    """``results``: geometry, frequency, excited state and wavefunction results."""
    geometry: GeometryResults
    freq: FrequencyResults
    excited_states: ExcitedStatesResults
    wavefunction: WavefunctionResults


@dataclass
class MoleculeDerivedData:
    """Group of each model by molecular topology (pictures of the different topologies)."""
    topology_groups: List[Optional[int]]


@dataclass
class MODiagramsComputedData:
    """MO grids of one spin (alpha, beta or restricted): labels, voxels and grid axes (bohr)."""
    spin: str
    labels: List[str]
    discretized_MO_voxels: Any
    grid_X: Any
    grid_Y: Any
    grid_Z: Any


@dataclass
class MODiagramsDerivedData:
    """MO grids of the reference: alpha and beta (unrestricted) or restricted."""
    alpha: Optional[MODiagramsComputedData]
    beta: Optional[MODiagramsComputedData]
    restricted: Optional[MODiagramsComputedData]


@dataclass
class MOAnalysisDerivedData:
    """Labels (homo, lumo+1...) and indexes of the requested MO of each model, and the MO grids."""
    MO_labels: List[Optional[List[List[str]]]]
    MO_indexes: List[Optional[List[List[int]]]]
    MO_diagrams: MODiagramsDerivedData


@dataclass
class MEPMapsDerivedData:
    """Molecular electrostatic potential of the reference: density and MEP on the grid (bohr,
    atomic units), extrema of the MEP on the density isosurface (mep_surface), isovalue of
    that surface, and half-width of the colour scale of the picture (from the charge).
    """
    rho_voxels: Optional[Any]
    MEP_voxels: Optional[Any]
    grid_X: Optional[Any]
    grid_Y: Optional[Any]
    grid_Z: Optional[Any]
    # Local extrema of the MEP on the density isosurface (data_enrichment.mep_surface)
    surface_isovalue: Optional[float]
    surface_extrema: Optional[Any]
    # Half-width of the fixed colour scale of the MEP picture (hartree/e), from the charge
    color_range: Optional[float]


@dataclass
class PopAnalysisDerivedData:
    """Mean and standard deviation of each charge model, and atoms whose charge deviates."""
    Mulliken_mean: Optional[float]
    Mulliken_std: Optional[float]
    Hirshfeld_mean: Optional[float]
    Hirshfeld_std: Optional[float]
    CM5_partial_mean: Optional[float]
    CM5_partial_std: Optional[float]
    indices: Optional[List[int]]


@dataclass
class WavefunctionDerivedData:
    """Derived wavefunction data: MO analysis, MEP maps, population analysis."""
    MO_analysis: MOAnalysisDerivedData
    MEP_maps: MEPMapsDerivedData
    pop_analysis: PopAnalysisDerivedData


@dataclass
class FukuiData:
    """Grid of one Fukui function (f+, f- or dual descriptor) and its axes."""
    rho_voxels: Any
    grid_X: Any
    grid_Y: Any
    grid_Z: Any


@dataclass
class FukuiFunctionsDerivedData:
    """Fukui functions of the reference (from the cation and anion single points)."""
    SP_plus: Optional[FukuiData]
    SP_minus: Optional[FukuiData]
    dual: Optional[FukuiData]


@dataclass
class ReactivityDerivedData:
    """Derived reactivity data (Fukui functions)."""
    fukui_functions: FukuiFunctionsDerivedData


@dataclass
class ExcitedStatesDerivedData:
    """Derived data of the vertical excited states, per model: selected states, charge-transfer
    indices (Tozer lambda, d_ct, q_ct), density difference grids, transition dipoles,
    absorption and CD spectra, NTO pairs.
    """
    nto_pairs: List[Optional[Any]]
    nto_reference: Optional[Any]
    Tozer_lambda: List[Optional[List[float]]]
    d_ct: List[Optional[List[float]]]
    q_ct: List[Optional[List[float]]]
    TD_output: List[Optional[Any]]
    grid_X: List[Optional[Any]]
    grid_Y: List[Optional[Any]]
    grid_Z: List[Optional[Any]]
    all_data_dip: List[Optional[List[Any]]]
    # transition density and dipole of each selected state (compute.transition_dipole)
    transition_dipoles: List[Optional[List[Any]]]
    abso_spectrum: List[Optional[Any]]
    CD_spectrum: List[Optional[Any]]
    xvalues: List[Optional[Any]]


@dataclass
class OptimizedExcitedStatesDerivedData:
    """Derived data of the optimized excited states, per model: charge-transfer indices, density
    difference grids, emission and CPL spectra, and the model whose orbitals gave the
    density differences (``orbitals_log_idx``).
    """
    Tozer_lambda: List[Optional[List[float]]]
    d_ct: List[Optional[List[float]]]
    q_ct: List[Optional[List[float]]]
    TD_output: List[Optional[Any]]
    grid_X: List[Optional[Any]]
    grid_Y: List[Optional[Any]]
    grid_Z: List[Optional[Any]]
    all_data_dip: List[Optional[List[Any]]]
    # transition density and dipole of the emitting state (compute.transition_dipole)
    transition_dipoles: List[Optional[List[Any]]]
    emi_spectrum: List[Optional[Any]]
    CPL_spectrum: List[Optional[Any]]
    xvalues: List[Optional[Any]]
    # index of the log whose orbitals and geometry gave the emission EDD (the opt_es log
    # itself when it has MO coefficients, else the ground-state reference)
    orbitals_log_idx: List[Optional[int]]
    # Phosphorescence of an optimized triplet with SOC (compute.compute_phosphorescence):
    # sublevels (energy cm-1, fosc, k_r s-1), mean k_r, radiative lifetime, spectrum
    phosphorescence: List[Optional[Any]]

@dataclass
class DerivedData:
    """Everything computed for the report from the parsed data."""
    molecule: MoleculeDerivedData
    wavefunction: WavefunctionDerivedData
    reactivity: ReactivityDerivedData
    excited_states: ExcitedStatesDerivedData
    optimized_excited_states: OptimizedExcitedStatesDerivedData


@dataclass
class ComputedData:
    """Data of the report: metadata, molecule, details, results of each model, derived data."""
    metadata: Metadata
    molecule: Molecule
    comp_details: ComputationDetails
    results: Results
    derived: DerivedData


def initialize_field(field_type, n):
    """Empty value of a field: ``[None] * n`` for a list field, None otherwise.

    Parameters
    ----------
    field_type : type
        Annotation of the dataclass field.
    n : int
        Number of models.
    """
    origin = get_origin(field_type)
    args = get_args(field_type)

    # Handle list of values for each logfile
    # List[T]
    if origin is list and args:
        elem_type = args[0]
        return [None for _ in range(n)]

    # Handle unique values
    # Optional[T] (Union[T, None])
    if origin is Union and type(None) in args:
        return None

    # Default case : return None
    return None


def init_dataclass(cls, n):
    """Build an empty dataclass for n models, recursively (nested dataclasses are built too).

    ``Metadata.log_files`` starts as ``["N/A"] * n`` and ``Metadata.ref_log_file_idx`` as 0.

    Parameters
    ----------
    cls : dataclass type
    n : int

    Raises
    ------
    ValueError
        cls is not a dataclass.
    """
    if not is_dataclass(cls):
        raise ValueError(f"{cls} is not a dataclass")

    init_kwargs = {}
    for field in fields(cls):
        field_type = field.type

        # Special Case : Init metadata.log_files to "N/A"
        if cls.__name__ == "Metadata" and field.name == "log_files":
            init_kwargs[field.name] = ["N/A"] * n
            continue
        
        # Special Case : Init Metadata.ref_log_file_idx to 0
        if cls.__name__ == "Metadata" and field.name == "ref_log_file_idx":
            init_kwargs[field.name] = 0
            continue

        if inspect.isclass(field_type) and is_dataclass(field_type):
            init_kwargs[field.name] = init_dataclass(field_type, n)
        else:
            init_kwargs[field.name] = initialize_field(field_type, n)

    return cls(**init_kwargs)


def init_computed_data(n):
    """Empty ComputedData for n models.

    Parameters
    ----------
    n : int
        Number of parsed models (at least 1).

    Returns
    -------
    ComputedData
    """
    if (n < 1):
        raise ValueError(f"The value n (number of log files) to initialize ComputedData must be at least 1 and not {n}.")
    
    return init_dataclass(ComputedData, n)


def config_logfile(config, computed_data, log_idx):
    """YAML entry (Config.logfiles item) of a model.

    Models and YAML entries differ when a log file has several steps.

    Parameters
    ----------
    config : Config
    computed_data : ComputedData
    log_idx : int
        Index of the model.

    Returns
    -------
    LogFile
    """
    idx = computed_data.metadata.config_indexes[log_idx]
    return config.logfiles[log_idx if idx is None else idx]
