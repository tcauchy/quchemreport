"""LogData: the data read from one calculation step (pydantic model).

scanlog.logfile_to_obj fills one LogData per step of a log file, from cclib and Open Babel.
The model has four sections: ``metadata`` (file, parser), ``molecule`` (composition, charge,
connectivity, starting geometry), ``comp_details`` (program, method, basis set, options) and
``results`` (wavefunction, geometry, frequencies, excited states). Units are those of cclib:
Angstrom for the coordinates, hartree for the energies (eV values of cclib are converted),
cm-1 for the frequencies. Every field defaults to None: a value missing in a log stays None.
Assignments are validated (``validate_assignment``), so a value of the wrong type is refused.
"""

from typing import Any, Dict, List, Tuple, Union

from pydantic import BaseModel
import numpy as np

DISPERSION_LABELS = {"d2": "D2", "d3zero": "D3(0)", "d3bj": "D3(BJ)", "d4": "D4", "vv10": "VV10"}


def functional_label(functional, dispersion=None):
    """Functional with its dispersion correction, e.g. ``"B3LYP + D3(BJ)"``.

    This is the level of theory compared between log files (conformity.test_theory) and
    shown in the report. Built-in corrections are shown too (``"wB97XD + D2"``), since cclib
    reads the model from what the program prints.

    Parameters
    ----------
    functional : str
    dispersion : str, optional
        cclib dispersion model (d2, d3zero, d3bj, d4, vv10), see DISPERSION_LABELS.

    Returns
    -------
    str
    """
    if dispersion is None:
        return functional
    return f"{functional} + {DISPERSION_LABELS.get(dispersion, dispersion)}"


class CustomModel(BaseModel):
    """Base of the models: assignments are validated against the field types."""
    model_config = {
        "validate_assignment": True,
    }

class Metadata(CustomModel):
    """Origin of the model: parser version, step file name, whether it has MO coefficients
    (``discretizable``), and index of its entry in the YAML file (``config_idx``).
    """
    parser_version: str = None
    log_file: str = None
    discretizable: bool = None
    archivable: bool = None
    archivable_for_new_entry: bool = None
    # Index of the YAML logfiles entry this model comes from (a log file may contain several steps)
    config_idx: int = None


class MoleculeConnectivity(CustomModel):
    """Bonds (Open Babel): atom index pairs, both directions, and bond orders."""
    atom_pairs: List[Tuple[int, int]] = None
    bond_orders: List[int] = None


class Molecule(CustomModel):
    """Composition and identity of the molecule: formula, atoms, charge, multiplicity, InChI,
    SMILES, bonds and the starting geometry and energy of the calculation.
    """
    chirality: bool = None
    atoms_valence: List[int] = None
    inchi: str = None
    smi: str = None
    can: str = None
    monoisotopic_mass: float = None
    connectivity: MoleculeConnectivity = MoleculeConnectivity()
    formula: str = None
    nb_atoms: int = None
    nb_heavy_atoms: int = None
    charge: int = None
    multiplicity: int = None
    atoms_Z: List[int] = None
    # atoms_masses: List[float] = None # TODO: TO CHECK (Value unused / Maybe later)
    # nuclear_spins: List[float] = None # TODO: TO CHECK (Value unused / Maybe later)
    # atoms_Zeff: List[float] = None # TODO: TO CHECK (Value unused / Maybe later)
    # nuclear_QMom: List[float] = None # TODO: TO CHECK (Value unused / Maybe later)
    # nuclear_gfactors: List[float] = None # TODO: TO CHECK (Value unused / Maybe later)
    starting_geometry: List[List[float]] = None
    starting_energy: float = None
    starting_nuclear_repulsion: float = None

class CompDetailsGeneral(CustomModel):
    """Computational details: program and version, methods, functional and dispersion model,
    basis set (name, pickled cclib gbasis and its MD5 hash, size, AO names), closed or open
    shell, solvent, SCF targets, core electrons per atom (ECP), and the job types given in
    the YAML file.
    """
    package: str = None
    package_version: str = None
    all_unique_theory: List[str] = None
    last_theory: str = None
    list_theory: List[str] = None
    functional: str = None
    # Dispersion model from cclib (metadata["dispersion"]): d2, d3zero, d3bj, d4 or vv10
    dispersion: str = None
    basis_set_name: str = None
    basis_set: str = None
    basis_set_md5: str = None
    basis_set_size: int = None
    ao_names: List[str] = None
    is_closed_shell: bool = None
    integration_grid: str = None
    solvent: str = None
    solvent_reaction_field: str = None
    scf_targets: List[List[float]] = None
    core_electrons_per_atoms: List[int] = None
    job_type: List[str] = None

class CompDetailsGeometry(CustomModel):
    """Convergence targets of the geometry optimization."""
    geometric_targets: List[float] = None


class CompDetailsFreq(CustomModel):
    """Temperature and anharmonicity options of the frequency calculation."""
    temperature: Union[float, List[float]] = None # TODO : TO CHECK (DATA NOT FOUND IN SCANLOG & NOT SURE FOR TYPE)
    anharmonicity: List[List[float]] = None # TODO : TO CHECK (NOT SURE FOR TYPE)


class CompDetailsExcitedStates(CustomModel):
    """Number of computed excited states."""
    nb_et_states: int = None
    # TDA: str = None # TODO: Maybe Later
    # et_sym_constraints: str = None # TODO: Maybe Later


class CompDetails(CustomModel):
    """``comp_details`` section: general, geometry, frequency and excited state details."""
    general: CompDetailsGeneral = CompDetailsGeneral()
    geometry: CompDetailsGeometry = CompDetailsGeometry()
    freq: CompDetailsFreq = CompDetailsFreq()
    excited_states: CompDetailsExcitedStates = CompDetailsExcitedStates()


class ResultsWaveFunction(CustomModel):
    """Wavefunction results: HOMO indexes, MO energies, symmetries and coefficients (kept up to
    a cutoff, see scanlog.wavefunction_results_subsection), total energy, atomic charges,
    multipole moments and SCF values. The conceptual DFT fields (A, I, Khi, Eta, Omega,
    DeltaN, condensed Fukui functions) are not filled by the parser.
    """
    homo_indexes: List[int] = None
    MO_energies: List[List[float]] = None
    MO_sym: List[List[str]] = None
    MO_number: int = None
    MO_number_kept: int = None
    MO_coefs: List[List[List[float]]] = None
    total_molecular_energy: float = None
    Mulliken_partial_charges: List[float] = None
    virial_ratio: float = None
    moments: List[List[float]] = None # List[List[float]]
    SCF_values: List[float] = None
    Hirshfeld_partial_charges: List[float] = None # TODO : TO CHECK (Not sure for the type) + COMMENTED IN SCANLOG BUT USED IN CALC_ORB
    A: float = None # TODO : TO CHECK (Not sure for the type)
    I: float = None # TODO : TO CHECK (Not sure for the type)
    Khi: float = None # TODO : TO CHECK (Not sure for the type)
    Eta: float = None # TODO : TO CHECK (Not sure for the type)
    Omega: float = None # TODO : TO CHECK (Not sure for the type)
    DeltaN: float = None # TODO : TO CHECK (Not sure for the type)
    fplus_lambda_mulliken: List[Any] = None # TODO : TO CHECK (Not sure for the type)
    fminus_lambda_mulliken: List[Any] = None # TODO : TO CHECK (Not sure for the type)
    fdual_lambda_mulliken: List[Any] = None # TODO : TO CHECK (Not sure for the type)
    fplus_lambda_hirshfeld: List[Any] = None # TODO : TO CHECK (Not sure for the type)
    fminus_lambda_hirshfeld: List[Any] = None # TODO : TO CHECK (Not sure for the type)
    fdual_lambda_hirshfeld: List[Any] = None # TODO : TO CHECK (Not sure for the type)
    CM5_partial_charges: List[float] = None  # TODO : TO CHECK (Not found in scanlong & Not sure for the type)

class ResultsGeometry(CustomModel):
    """Geometry results: coordinates (Angstrom, flat list), nuclear repulsion energy,
    optimization status and convergence values.
    """
    nuclear_repulsion_energy_from_xyz: float = None
    OPT_DONE: bool = None
    elements_3D_coords: List[float] = None
    geometric_values: List[List[float]] = None


class ResultsFreq(CustomModel):
    """Frequency results: thermochemistry (hartree), frequencies (cm-1), IR and Raman
    intensities, symmetries, displacements, polarizabilities, anharmonicity.
    """
    entropy: float = None
    enthalpy: float = None # TODO : TO CHECK (Not sure for the type)
    free_energy: float = None # TODO : TO CHECK (Not sure for the type)
    zero_point_energy: float = None # TODO : TO CHECK (Not sure for the type)
    electronic_thermal_energy: float = None # TODO : TO CHECK (Not sure for the type)
    vibrational_freq: List[float] = None # TODO : TO CHECK (Not sure for the type)
    polarizabilities: List[List[float]] = None # TODO : TO CHECK (Not sure for the type)
    vibrational_int: List[float] = None # TODO : TO CHECK (Not sure for the type)
    vibrational_sym: List[str] = None # TODO : TO CHECK (Not sure for the type)
    vibration_disp: List[List[List[float]]] = None # TODO : TO CHECK (Not sure for the type)
    vibrational_anharms: List[List[float]] = None # TODO : TO CHECK (Not sure for the type)
    vibrational_raman: List[float] = None # TODO : TO CHECK (Not sure for the type)


class ResultsExcitedStates(CustomModel):
    """Excited state results: energies, oscillator and rotatory strengths, symmetries,
    excitation amplitudes and transition dipoles. The charge-transfer fields (Tozer lambda,
    d_ct, q_ct...) are not filled by the parser.
    """
    et_energies: Union[List[float], Dict[str, Any]] = None # TODO : TO CHECK (Not sure for the type)
    et_oscs: List[float] = None # TODO : TO CHECK (Not sure for the type)
    et_sym : List[str] = None # TODO : TO CHECK (Not sure for the type)
    et_transitions: List[List[Tuple[Tuple[int, int], Tuple[int, int], float]]] = None # TODO : TO CHECK (Not sure for the type)
    et_rot: List[float] = None # TODO : TO CHECK (Not sure for the type)
    et_eldips: List[List[float]] = None # TODO : TO CHECK (Not sure for the type)
    et_veldips: List[List[float]] = None # TODO : TO CHECK (Not sure for the type)
    et_magdips: List[List[float]] = None # TODO : TO CHECK (Not sure for the type)
    Tozer_lambda: List[Any] = None # TODO : TO CHECK (Not sure for the type)
    d_ct: List[Any] = None # TODO : TO CHECK (Not sure for the type)
    q_ct: List[Any] = None # TODO : TO CHECK (Not sure for the type)
    mu_ct: List[Any] = None # TODO : TO CHECK (Not sure for the type)
    e_barycenter: List[Any] = None # TODO : TO CHECK (Not sure for the type)
    hole_barycenter: List[Any] = None # TODO : TO CHECK (Not sure for the type)
    # Spin-orbit coupling results of ORCA (parser_engine.orca_soc.read_orca_soc), None otherwise
    soc: Any = None


class Results(CustomModel):
    """``results`` section: wavefunction, geometry, frequency and excited state results."""
    wavefunction: ResultsWaveFunction = ResultsWaveFunction()
    geometry: ResultsGeometry = ResultsGeometry()
    freq: ResultsFreq = ResultsFreq()
    excited_states: ResultsExcitedStates = ResultsExcitedStates()


class LogData(CustomModel):
    """Everything read from one calculation step: metadata, molecule, computational details
    and results (see the module documentation).
    """
    metadata: Metadata = Metadata()
    molecule: Molecule = Molecule()
    comp_details: CompDetails = CompDetails()
    results: Results = Results()