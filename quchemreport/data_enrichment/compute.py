"""Computations of the report: everything that is not read directly in the log files.

compute_data fills a ComputedData object from the parsed models and the conformity results,
running only the computations needed by the outputs asked in the YAML file (ComputeFlags):
labels and grids of the requested molecular orbitals, molecular electrostatic potential and
its extrema on the molecular surface, population analysis, conceptual DFT indices and Fukui
functions, spectra, transitions, electron density differences and natural transition orbitals
of the excited states and of the optimized excited states.

Grids are computed on the model chosen by the conformity tests (``data_for_discretization``):
the reference, or a log at the same geometry that has the basis set and the MO coefficients.
When no log has them, the grid outputs are skipped with a warning. Grids use a step of
obk_step (0.075 bohr) and nproc threads (resources.nproc, 20 when omitted).

Naming of the single points used for the Fukui functions: "SP plus" is the system with one
more electron (N+1, the anion of a neutral molecule), used for f+; "SP minus" the system with
one electron less (N-1, the cation), used for f-.
"""

from collections import defaultdict
import io
import json
import os
import re
import sys
import threading
from typing import Any, Dict, List, Optional, Union

import numpy as np
import multiprocessing as mp
import psutil

from quchemreport.config.config import Config
from quchemreport.data_enrichment import TD2UVvis, calc_orb, mep_surface
from quchemreport.data_enrichment.nto import build_nto_pairs, X_NORM_COMPLETE_THRESHOLD
from quchemreport.data_enrichment.compute_flags import ComputeFlags, resolve_compute_flags
from quchemreport.utility_services.computed_data import ComputedData, FukuiData, MODiagramsComputedData, init_computed_data
from quchemreport.utility_services.log_data import LogData
from quchemreport.utility_services.validated_data import ValidatedData
from quchemreport.utility_services.parameters import FWHM, obk_step
from quchemreport.utility_services.units import a0_to_A
from quchemreport.utility_services.naming import spin_letter
from quchemreport.utility_services.naming import spin_letter, state_number

# Number of processes/threads for the grid computations; replaced by
# config.resources.nproc (when given) at the start of compute_data.
nproc = 20



def normalize_et_energies_format(et_energies: Optional[Union[List[float], Dict[str, Any]]]) -> Optional[List[float]]:
    # Cases where no et_energies are available
    """Excited state energies as a list (cm-1), whatever the format given by cclib.

    cclib gives a list for Gaussian and, for some ORCA versions, a dict of spectra; the
    energies of the absorption spectrum through the electric transition dipoles are taken.

    Returns
    -------
    list of float or None
    """
    if et_energies is None:
        return None

    # Case where et_energies is already a list
    if isinstance(et_energies, list):
        return et_energies

    # Case where et_energies is a dictionary corresponding to the ORCA format
    if isinstance(et_energies, dict):
        key = 'ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS'
        if (
            key in et_energies
            and isinstance(et_energies[key], list)
            and et_energies[key]
            and isinstance(et_energies[key][0], list)
        ):
            return et_energies[key][0]

    # Default Case : Return None
    return None


def format_MO_label(acronym, offset):
    """Label of an orbital from its acronym and offset: (homo, 0) -> homo, (lumo, 2) -> lumo+2."""
    if offset == 0:
        return acronym
    return f"{acronym}{f'+{offset}' if offset > 0 else offset}"


def compute_MO_labels_and_indexes(log_data: LogData, config: Config, log_file):
    
    """Labels and indexes of the orbitals of ``MO_analysis.MO_list`` for one model.

    Labels such as ``homo-1`` or ``lumo+2`` are converted into orbital indexes (0-based)
    from the HOMO indexes; labels that do not exist in the model (homo+n, lumo-n, outside
    the kept orbitals) are dropped with a warning. Unrestricted calculations give one list
    per spin, with ``_alpha`` / ``_beta`` labels.

    Parameters
    ----------
    log_data : LogData
    config : Config
    log_file : str
        Name of the log, for the warning.

    Returns
    -------
    MO_labels, MO_indexes : list of list, or (None, None)
        One list per spin; None when the model has no orbital data.
    """
    MO_list_labels = config.output.include.wavefunction.MO_analysis.MO_list
    HO_ind = log_data.results.wavefunction.homo_indexes
    MO_number_kept = log_data.results.wavefunction.MO_number_kept

    
    # Skip if no data to compute
    if (
        MO_list_labels == None
        or HO_ind == None
        or MO_number_kept == None
    ):
        return None, None
    
    max_MO_index = MO_number_kept - 1
    warning = False
    
    # Parsing MO_list_labels
    label_data = []
    for label in MO_list_labels:
        m = re.fullmatch(r"(homo|lumo)([+-]?\d*)", label.lower())
        if m:
            label_data.append((m.group(1), int(m.group(2) or 0)))
        else:
            warning = True
    
    # Filtering invalid labels in MO_list
    valid_labels = []
    for (acronym,offset) in label_data:
        w = False
        if acronym == "homo" and offset > 0:
            w = True
        if acronym == "lumo" and offset < 0:
            w = True
        # same index as below: lumo+n is homo + n + 1
        shift = offset + (1 if acronym == "lumo" else 0)
        if HO_ind[0] + shift < 0 or HO_ind[0] + shift > max_MO_index:
            w = True
        if len(HO_ind) == 2 and (HO_ind[1] + shift < 0 or HO_ind[1] + shift > max_MO_index):
            w = True
            
        if w == True:
            warning = True
        else:
            valid_labels.append((acronym,offset))
    
    if warning:
        print(f"Warning: some labels are invalid for {log_file}. Labels kept: {valid_labels}")
        
    MO_labels = []
    MO_indexes = []
    
    if len(HO_ind) == 2:
        for i in range(2):
            spin = ['alpha', 'beta'][i]
            
            spin_labels = []
            spin_indexes = []
            for (acronym, offset) in valid_labels:
                spin_labels.append(f"{format_MO_label(acronym, offset)}_{spin}")
                index = HO_ind[i] + offset
                if acronym == "lumo":
                    index += 1
                spin_indexes.append(index)
            
            MO_labels.append(spin_labels)
            MO_indexes.append(spin_indexes)
    else:
        spin_labels = []
        spin_indexes = []
        
        for (acronym, offset) in valid_labels:
            spin_labels.append(format_MO_label(acronym, offset))
            index = HO_ind[0] + offset
            if acronym == "lumo":
                index += 1
            spin_indexes.append(index)
            
        MO_labels.append(spin_labels)
        MO_indexes.append(spin_indexes)
        
    return MO_labels, MO_indexes


def extract_simple_data(list_data_models: List[LogData], ref_data:LogData, computed_data: ComputedData):
    """Copy the data of the parsed models into ComputedData (one value per model).

    Molecule identity and atomic charges are taken from the reference log (YAML
    ``reference: true``), which is not necessarily the model used for the grids.

    Parameters
    ----------
    list_data_models : list of LogData
    ref_data : LogData
        Reference model.
    computed_data : ComputedData
        Filled in place.
    """
    
    # Extract simple data from the reference logfile
    computed_data.molecule.monoisotopic_mass = ref_data.molecule.monoisotopic_mass
    computed_data.molecule.inchi = ref_data.molecule.inchi
    computed_data.molecule.smi = ref_data.molecule.smi
    computed_data.molecule.formula = ref_data.molecule.formula
    computed_data.comp_details.general.package = ref_data.comp_details.general.package
    computed_data.results.wavefunction.Mulliken_partial_charges = ref_data.results.wavefunction.Mulliken_partial_charges
    computed_data.results.wavefunction.Hirshfeld_partial_charges = ref_data.results.wavefunction.Hirshfeld_partial_charges
    computed_data.results.wavefunction.CM5_partial_charges = ref_data.results.wavefunction.CM5_partial_charges
    
    # Exctract simple data from each logfile
    for i,log in enumerate(list_data_models):
        # Extract metadata
        computed_data.metadata.log_files[i] = log.metadata.log_file
        # Index of the YAML logfile entry (a log file may give several steps/models)
        computed_data.metadata.config_indexes[i] = log.metadata.config_idx
        
        # Extract Molecule Data
        computed_data.molecule.charges[i] = log.molecule.charge
        computed_data.molecule.multiplicity[i] = log.molecule.multiplicity
        computed_data.molecule.atoms_Z[i] = log.molecule.atoms_Z
        computed_data.molecule.connectivity.atom_pairs[i] = log.molecule.connectivity.atom_pairs
        
        # Extract General Computational Details
        computed_data.comp_details.general.job_type[i] = log.comp_details.general.job_type
        computed_data.comp_details.general.package_version[i] = log.comp_details.general.package_version
        computed_data.comp_details.general.last_theory[i] = log.comp_details.general.last_theory
        computed_data.comp_details.general.functional[i] = log.comp_details.general.functional
        computed_data.comp_details.general.dispersion[i] = log.comp_details.general.dispersion
        computed_data.comp_details.general.basis_set_name[i] = log.comp_details.general.basis_set_name
        computed_data.comp_details.general.basis_set_size[i] = log.comp_details.general.basis_set_size
        computed_data.comp_details.general.is_closed_shell[i] = log.comp_details.general.is_closed_shell
        computed_data.comp_details.general.integration_grid[i] = log.comp_details.general.integration_grid
        computed_data.comp_details.general.solvent[i] = log.comp_details.general.solvent
        computed_data.comp_details.general.scf_targets[i] = log.comp_details.general.scf_targets
        
        # Extract Geometry Computational Details
        computed_data.comp_details.geometry.geometric_targets[i] = log.comp_details.geometry.geometric_targets
        
        # Extract Frequency Computational Details
        computed_data.comp_details.freq.temperature[i] = log.comp_details.freq.temperature
        computed_data.comp_details.freq.anharmonicity[i] = log.comp_details.freq.anharmonicity
    
        # Extract Excited States Computational Details
        computed_data.comp_details.excited_states.nb_et_states[i] = log.comp_details.excited_states.nb_et_states
        
        # Extract Geometry Results
        computed_data.results.geometry.elements_3D_coords[i] = log.results.geometry.elements_3D_coords
        computed_data.results.geometry.geometric_values[i] = log.results.geometry.geometric_values
        computed_data.results.geometry.nuclear_repulsion_energy_from_xyz[i] = log.results.geometry.nuclear_repulsion_energy_from_xyz
        computed_data.results.geometry.OPT_DONE[i] = log.results.geometry.OPT_DONE

        # Extract Frequency Results
        computed_data.results.freq.vibrational_int[i] = log.results.freq.vibrational_int
        computed_data.results.freq.vibrational_freq[i] = log.results.freq.vibrational_freq
        computed_data.results.freq.vibrational_sym[i] = log.results.freq.vibrational_sym
        computed_data.results.freq.zero_point_energy[i] = log.results.freq.zero_point_energy
        computed_data.results.freq.electronic_thermal_energy[i] = log.results.freq.electronic_thermal_energy
        computed_data.results.freq.enthalpy[i] = log.results.freq.enthalpy
        computed_data.results.freq.free_energy[i] = log.results.freq.free_energy
        computed_data.results.freq.entropy[i] = log.results.freq.entropy
        
        # Extract Excited States Results
        computed_data.results.excited_states.et_sym[i] = log.results.excited_states.et_sym
        computed_data.results.excited_states.et_s2[i] = log.results.excited_states.et_s2
        computed_data.results.excited_states.et_eldips[i] = log.results.excited_states.et_eldips
        computed_data.results.excited_states.et_veldips[i] = log.results.excited_states.et_veldips
        computed_data.results.excited_states.et_magdips[i] = log.results.excited_states.et_magdips
        
        computed_data.results.excited_states.et_energies[i] = normalize_et_energies_format(log.results.excited_states.et_energies)
        computed_data.results.excited_states.et_oscs[i] = log.results.excited_states.et_oscs
        computed_data.results.excited_states.et_rot[i] = log.results.excited_states.et_rot
        computed_data.results.excited_states.et_transitions[i] = log.results.excited_states.et_transitions
        computed_data.results.excited_states.et_deexcitations[i] = log.results.excited_states.et_deexcitations
        computed_data.results.excited_states.soc[i] = log.results.excited_states.soc
        
        # Extract Wavefunction Results
        computed_data.results.wavefunction.total_molecular_energy[i] = log.results.wavefunction.total_molecular_energy
        computed_data.results.wavefunction.homo_indexes[i] = log.results.wavefunction.homo_indexes
        computed_data.results.wavefunction.MO_energies[i] = log.results.wavefunction.MO_energies
        computed_data.results.wavefunction.moments[i] = log.results.wavefunction.moments


def serialize_topology(atom_pairs, atoms_Z):
    """Canonical JSON of a topology (sorted bonds and atomic numbers), to group models with the same one."""
    data = {
        "atom_pairs": sorted([sorted(pair) for pair in atom_pairs]),
        "atoms_Z": atoms_Z,
    }
    return json.dumps(data, sort_keys=True)


def compute_molecule_representations(computed_data: ComputedData):
    
    """Group the models by molecular topology (same atoms and bonds).

    The first model of each group (the reference for its own group) gets the list of the
    group in ``derived.molecule.topology_groups``: one topology picture is drawn per group.
    """
    print("\n=================================================================")
    print(" Compute : Computing Molecule Representations")
    print("=================================================================")
    
    
    log_files = computed_data.metadata.log_files
    ref_id = computed_data.metadata.ref_log_file_idx

    all_atom_pairs = computed_data.molecule.connectivity.atom_pairs
    all_atoms_Z = computed_data.molecule.atoms_Z

    # Group by topology
    seen = defaultdict(list)
    for log_idx, log_file in enumerate(log_files):
        atom_pairs = all_atom_pairs[log_idx]
        atoms_Z = all_atoms_Z[log_idx]
        
        if atom_pairs is None or atoms_Z is None:
            continue

        key = serialize_topology(atom_pairs, atoms_Z)
        seen[key].append(log_idx)

    # Create groups of logfiles by equivalent topologies
    groups = []
    for group in seen.values():
        if ref_id in group:
            group.remove(ref_id)
            group.insert(0, ref_id)
        groups.append(group)
        
    for group in groups:
        base_log_idx = group[0]
        
        computed_data.derived.molecule.topology_groups[base_log_idx] = group
        
        sources = ", ".join([log_files[i] for i in group])
        print(f"Same molecule topology for : {sources}")
        


def compute_all_MO_labels_and_indexes(computed_data: ComputedData, list_data_models: List[LogData], config: Config):
    """Labels and indexes of the requested orbitals for every model."""
    print("\n=================================================================")
    print(" Compute : Computing MO Labels and Indexes")
    print("=================================================================")
    
    for i,log in enumerate(list_data_models):
        log_file = log.metadata.log_file
        
        # Compute MO_lables and MO_index if the required data is available
        # Otherwise MO_lables and MO_indexes both set to None
        MO_labels, MO_indexes = compute_MO_labels_and_indexes(log, config, log_file)
        
        computed_data.derived.wavefunction.MO_analysis.MO_labels[i] = MO_labels
        computed_data.derived.wavefunction.MO_analysis.MO_indexes[i] = MO_indexes
        
        if (
            MO_labels is not None
            and MO_indexes is not None
        ):
            print(f"MO Labels and MO Indexes done for {log_file}")


# Orbitals closer than this in energy (eV) are degenerate (the programs print 5 decimals in
# hartree, 3e-4 eV)
DEGENERATE_EV = 0.01


def degenerate_partners(energies, index, tol=DEGENERATE_EV):
    """Indexes of the other orbitals degenerate with orbital index (energies of one spin, eV)."""
    if energies is None or index is None or not 0 <= index < len(energies):
        return []
    return [k for k in range(len(energies)) if k != index and abs(energies[k] - energies[index]) < tol]


def show_degenerate_pairs(voxels, indexes, energies, X, Y, Z, direction, centre, grid_of):
    """Pictures of degenerate pairs of orbitals seen alike from the first point of view.

    Any combination of two degenerate orbitals is as valid as the pair given by the program,
    which is arbitrary (ORCA) or follows the axes of its frame (Gaussian). Seen from the side,
    one orbital of the pi pair of O2 was in the picture plane and the other pointed at the
    camera (two balls over the atoms, 26/11/2026). Each requested orbital of a degenerate pair
    is replaced by the combination cos t phi_i + sin t phi_j whose extent along the view,
    Q = sum phi^2 (d.(r - c))^2, equals that of its partner: for a pi pair seen from the side,
    two orbitals at 45 degrees to the view, both showing their lobes. The partner, when not
    requested, is computed by grid_of(index). Groups of three or more degenerate orbitals are
    left as they are.

    voxels: array (n, nx, ny, nz), the grids of indexes (modified in place); energies: of this
    spin (eV); X, Y, Z: grid (bohr); direction: unit vector to the camera; centre: bohr.
    Returns the pairs rotated, as (index, partner)."""
    d = np.asarray(direction, dtype=float)
    depth2 = (d[0] * (X - centre[0]) + d[1] * (Y - centre[1]) + d[2] * (Z - centre[2])) ** 2
    done, rotated = set(), []
    for pos, i in enumerate(indexes):
        partners = degenerate_partners(energies, i)
        if len(partners) != 1 or i in done:
            continue
        j = partners[0]
        phi_i = voxels[pos]
        phi_j = voxels[indexes.index(j)] if j in indexes else np.asarray(grid_of(j))
        q11, q22 = np.sum(phi_i ** 2 * depth2), np.sum(phi_j ** 2 * depth2)
        q12 = np.sum(phi_i * phi_j * depth2)
        t = 0.5 * np.arctan2(q22 - q11, 2 * q12) if abs(q22 - q11) + abs(q12) > 0 else 0.0
        # the rotation closest to the original pair (|t| <= 45 degrees)
        t = (t + np.pi / 4) % (np.pi / 2) - np.pi / 4
        c, s_ = np.cos(t), np.sin(t)
        new_i, new_j = c * phi_i + s_ * phi_j, -s_ * phi_i + c * phi_j
        voxels[pos] = new_i
        if j in indexes:
            voxels[indexes.index(j)] = new_j
        done.update((i, j))
        rotated.append((i, j))
    return rotated


def _view_and_centre(ref_data):
    """First camera direction, centre (bohr) and vertical of the picture of the reference
    (camera.choose_views)."""
    from quchemreport.visualization_engine import camera
    coords = np.asarray(ref_data.results.geometry.elements_3D_coords, dtype=float).reshape(-1, 3)
    view = camera.choose_views(coords, ref_data.molecule.atoms_Z, ref_data.molecule.connectivity.atom_pairs)[0]
    return view.direction, coords.mean(axis=0) / a0_to_A, view.up



# Relative deviation |mu(grid) - mu(log)| / max(|mu(log)|, 0.1 au) beyond which the transition
# density drawn from the printed amplitudes is said to be approximate (20 degrees give 0.35).
# Measured 27/11/2026 (validation/calculs_20261127/tdip): all the amplitudes X and Y
# (Gaussian IOp(9/40=4)) give 0.01-0.03; the default printing (X above 0.1) gives 1 to 7 on
# the bright states of acetone and HCFClBr, mostly by the missing Y amplitudes (TD-DFT).
TRANSITION_DIPOLE_TOLERANCE = 0.35
# Below this norm (e.bohr) the transition is taken as forbidden: no arrow
FORBIDDEN_DIPOLE = 1.0e-3


# A molecule is taken as chiral (magnetic transition dipoles drawn) when one of its states has
# a rotatory strength above this (10^-40 cgs; Gaussian prints 0.0000 for achiral molecules)
CHIRAL_ROTATORY_STRENGTH = 1.0e-3


def is_chiral(rotatory_strengths):
    """Whether the rotatory strengths of a log show a chiral molecule."""
    values = [abs(r) for r in (rotatory_strengths or []) if r is not None and np.isfinite(r)]
    return bool(values) and max(values) > CHIRAL_ROTATORY_STRENGTH


# Units of the rotatory strengths (cgs, as printed by Gaussian and ORCA: 10^-40 erg.esu.cm/G):
# e.bohr = 2.541746e-18 esu.cm and Bohr magneton = 9.2740101e-21 erg/G, so that a transition
# dipole of 1 e.bohr parallel to a magnetic one of 1 Bohr magneton gives R = 235.72 (the factor of
# Gaussian's "R(length) = -235.7 mu.m"), and D = |mu|^2 = 1 (e.bohr)^2 gives 64604.7.
EBOHR_CGS = 2.541746e-18
BOHR_MAGNETON_CGS = 9.2740101e-21
ROTATORY_UNIT = EBOHR_CGS * BOHR_MAGNETON_CGS * 1e40
DIPOLE_STRENGTH_UNIT = EBOHR_CGS ** 2 * 1e40
CM_TO_HARTREE = 1.0 / 219474.6313705


def rotatory_strengths_velocity(energies_cm, veldips, magdips):
    """Rotatory strengths in the velocity gauge (10^-40 cgs), R = ROTATORY_UNIT p.m / E, from the
    velocity transition dipoles p = <0|del|n> and the magnetic ones m (a.u., as cclib reads them,
    etveldips and etmagdips), and the energies E (cm-1, converted to hartree); None without them
    (cclib reads neither from ORCA).

    Why (27/11/2026, Thomas Cauchy): reviewers ask for both gauges. R(length) depends on the
    origin of the program, R(velocity) does not; their agreement tells whether the sign of R is
    reliable. cclib reads R(length) only (etrotats), but the velocity and magnetic dipoles it
    reads give R(velocity): on HCFClBr (10 states, TD-PBE0/6-31G(d)) the values printed by Gaussian
    are found to 0.034 at most, the rounding of the printed vectors (4 decimals; S3 21.607 for
    21.641 printed, S10 34.983 for 34.982, S1 -0.548 for -0.565), with the same sign for every
    state. A new cclib attribute was not needed. ORCA prints the same vectors (magnetic ones in
    atomic units, e hbar / m_e = 2 Bohr magnetons): with the cclib branch orca-magnetic-dipoles,
    which stores them in Bohr magnetons, the R(velocity) printed by ORCA is found to 0.002 (HCFClBr,
    TDA, 10 states)."""
    if energies_cm is None or veldips is None or magdips is None:
        return None
    n = min(len(energies_cm), len(veldips), len(magdips))
    if n == 0 or n != len(energies_cm):
        return None
    values = []
    for e, p, m in zip(energies_cm[:n], veldips[:n], magdips[:n]):
        p, m = np.asarray(p, dtype=float), np.asarray(m, dtype=float)
        if e is None or e <= 0 or not (np.all(np.isfinite(p)) and np.all(np.isfinite(m))):
            values.append(None)
            continue
        values.append(float(ROTATORY_UNIT * (p @ m) / (e * CM_TO_HARTREE)))
    return values


def dissymmetry_factor(rotatory_strength, dipole):
    """Dissymmetry (Kuhn) factor g = 4 R / D of a transition, from its rotatory strength
    (10^-40 cgs) and its electric transition dipole (e.bohr, D = |mu|^2); None when either is
    missing or the transition is forbidden."""
    if rotatory_strength is None or dipole is None:
        return None
    d = float(np.dot(dipole, dipole)) * DIPOLE_STRENGTH_UNIT
    if d <= 0 or not np.isfinite(rotatory_strength):
        return None
    return 4.0 * rotatory_strength / d


def dipole_angle(dipole, magnetic):
    """Angle (degrees) between the electric transition dipole and the magnetic one as drawn
    (compute.transition_dipole: acute for R > 0); None when either is missing or zero."""
    if dipole is None or magnetic is None:
        return None
    a, b = np.asarray(dipole, dtype=float), np.asarray(magnetic, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return None
    return float(np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1.0, 1.0))))


def transition_dipole(td_entry, log_dipole, symmetry, transitions, magnetic=None):
    """Transition density and dipole of one state, from an entry of calc_orb.TD.

    Turns the transition density of the entry, in place, into the charge transition density
    -rho_T (electrons of charge -1), with the phase that makes its dipole parallel to the
    dipole printed in the log: the arrow then goes from the negative to the positive regions
    of the picture. The sign of a transition density is arbitrary (phase of the state).

    Returns None for a spin-forbidden state of a restricted calculation (triplet: no charge
    transition density) or without signed amplitudes (ORCA without TDA prints only the
    weights), otherwise a dict: dipole (e.bohr, of the log when it prints one,
    else from the grid; None when forbidden), grid (dipole from the amplitudes), source
    ("log" or "grid"), deviation (see TRANSITION_DIPOLE_TOLERANCE, None without a log dipole),
    centre (bohr, mean position weighted by |rho_T|), magnetic (the magnetic transition dipole
    of the log, a.u., for a chiral molecule; else None).

    The magnetic dipole is given with the opposite sign of the one printed by Gaussian: the
    rotatory strength R(length) of Gaussian is -235.7 mu.m with the printed vectors (HCFClBr,
    10 states), so with -m an acute angle between the arrows means R > 0, as the "E-M Angle" of
    Gaussian. It is drawn when the phase of the state is that of the log (dipole of the log)."""
    rho_T = td_entry[3]
    if rho_T is None:       # no signed amplitudes (ORCA without TDA)
        return None
    mu_grid, centre = (np.asarray(v, dtype=float) for v in td_entry[4])
    if calc_orb.transition_density_factor(transitions) > 1.0 and spin_letter(symmetry) == "T":
        return None
    rho_T *= -1.0                       # charge density: its dipole is mu_grid
    exact = None if log_dipole is None else np.asarray(log_dipole, dtype=float)
    if exact is not None and np.linalg.norm(exact) > FORBIDDEN_DIPOLE and mu_grid @ exact < 0:
        rho_T *= -1.0
        mu_grid = -mu_grid
    deviation = None
    if exact is not None:
        deviation = float(np.linalg.norm(mu_grid - exact) / max(np.linalg.norm(exact), 0.1))
    dipole = exact if exact is not None else mu_grid
    m = None
    if magnetic is not None and exact is not None and np.all(np.isfinite(magnetic)):
        m = (-np.asarray(magnetic, dtype=float)).tolist()
    return {"dipole": None if np.linalg.norm(dipole) <= FORBIDDEN_DIPOLE else dipole.tolist(),
            "grid": mu_grid.tolist(), "source": "log" if exact is not None else "grid",
            "deviation": deviation, "centre": np.asarray(centre), "magnetic": m}

def compute_MO_diagrams(computed_data: ComputedData, ref_data: LogData, nproc, step):
    """Grids of the requested orbitals of the reference (one set per spin if unrestricted).

    Parameters
    ----------
    computed_data : ComputedData
    ref_data : LogData or None
        Model used for the grids (skipped with a warning when None).
    nproc : int
    step : float
        Grid step (bohr).
    """
    if ref_data is None:
        print("Warning: MO diagrams skipped: no log file with MO coefficients for the reference.")
        return
    ref_id = computed_data.metadata.ref_log_file_idx
    HO_ind = ref_data.results.wavefunction.homo_indexes
    MO_indexes = computed_data.derived.wavefunction.MO_analysis.MO_indexes[ref_id]
    MO_labels = computed_data.derived.wavefunction.MO_analysis.MO_labels[ref_id]
    
    # Skip if data missing
    if (
        HO_ind == None
        or MO_indexes == None
        or MO_labels == None
    ):
        return
    
    print("\n=================================================================")
    print(" Compute : Computing MO Diagrams")
    print("=================================================================")

    if len(HO_ind) == 2:
        print("Unrestricted calculation detected")
        
        for i in range(2):
            spin = ['alpha', 'beta'][i]
            
            MO_list = MO_indexes[i]
            discretized_MO_voxels, grid_X, grid_Y, grid_Z = calc_orb.MO(ref_data, MO_list, spin, grid_step=step, nproc=nproc)
            direction, centre, _ = _view_and_centre(ref_data)
            show_degenerate_pairs(discretized_MO_voxels, list(MO_list), ref_data.results.wavefunction.MO_energies[i],
                                  grid_X, grid_Y, grid_Z, direction, centre,
                                  lambda k, spin=spin: calc_orb.MO(ref_data, [k], spin, grid_step=step, nproc=nproc)[0][0])
            
            setattr(
                computed_data.derived.wavefunction.MO_analysis.MO_diagrams,
                spin,
                MODiagramsComputedData(
                    spin,
                    MO_labels[i],
                    discretized_MO_voxels,
                    grid_X,
                    grid_Y,
                    grid_Z
                )
            )
        
    else:
        
            MO_labels = MO_labels[0]
            # MO_list = MO_indexes[0]
            discretized_MO_voxels, grid_X, grid_Y, grid_Z = calc_orb.MO(ref_data, MO_labels, spin="none", grid_step=step, nproc=nproc)
            direction, centre, _ = _view_and_centre(ref_data)
            show_degenerate_pairs(discretized_MO_voxels, list(MO_indexes[0]), ref_data.results.wavefunction.MO_energies[0],
                                  grid_X, grid_Y, grid_Z, direction, centre,
                                  lambda k: calc_orb.MO(ref_data, [k], spin="none", grid_step=step, nproc=nproc)[0][0])
            
            computed_data.derived.wavefunction.MO_analysis.MO_diagrams.restricted = MODiagramsComputedData(
                "restricted",
                MO_labels,
                discretized_MO_voxels,
                grid_X,
                grid_Y,
                grid_Z
            )

def compute_MEP_worker_direct(ref_data, nproc, step):
    """Compute the MEP in the current process; None on MemoryError."""
    try:
        rho, MEP, grid_X, grid_Y, grid_Z, precomputed_rho_opt = calc_orb.Potential(ref_data, grid_step=step, nproc=nproc)
        return rho, MEP, grid_X, grid_Y, grid_Z, precomputed_rho_opt
    except MemoryError:
        return None

def compute_MEP_worker(ref_data, nproc, step, queue):
    """Compute the MEP in a child process and put the result (or None) in the queue."""
    try:
        rho, MEP, grid_X, grid_Y, grid_Z, precomputed_rho_opt = calc_orb.Potential(ref_data, grid_step=step, nproc=nproc)
        queue.put((rho, MEP, grid_X, grid_Y, grid_Z, precomputed_rho_opt))
    except MemoryError:
        queue.put(None)

USE_MULTIPROCESSING = False  # switch global (YAML, CLI, debug)


def compute_MEP_maps(computed_data, ref_data, nproc, step):
    
    """Molecular electrostatic potential of the reference and its extrema on the surface.

    The MEP grid (calc_orb.Potential) is stored with the density; its local minima and
    maxima on the 0.001 e-/bohr3 density isosurface (mep_surface) and the half-width of
    the colour scale of the picture, which depends on the charge, are stored too. A MEP that
    cannot be computed (missing data, electron count inconsistent with an ECP, memory) is
    skipped with a warning. With USE_MULTIPROCESSING, the computation runs in a child
    process, so that a process killed for lack of memory does not stop the report.

    Parameters
    ----------
    computed_data : ComputedData
    ref_data : LogData or None
        Model used for the grids.
    nproc : int
    step : float
        Grid step (bohr).

    Returns
    -------
    precomputed_rho_opt : numpy.ndarray or None
        Raw density of the reference on the grid, reused by the Fukui functions.
    """
    print("\n=================================================================")
    print(" Compute : Computing MEP Maps")
    print("=================================================================")
    
    if ref_data is None:
        print("Warning: MEP map skipped: no log file with MO coefficients for the reference.")
        return None
    problem = calc_orb.electron_count_error(ref_data)
    if problem:
        print(f"Warning: MEP map skipped: {problem}.")
        return None
    
    
    # We need to calculate the MEP map inside a child process because it may 
    # be killed by the operating system due to excessive memory consumption.
    if USE_MULTIPROCESSING:
        queue = mp.Queue()
        p = mp.Process(target=compute_MEP_worker, args=(ref_data, nproc, step, queue))
        p.start()
        
        try:
            # ? on lit d'abord la queue AVANT le join
            result = queue.get(timeout=600)
        except Exception as e:
            print(f"Failed to get result from child process: {e}")
            p.terminate()
            return None

        p.join()
        print("p.join ok")

        if p.exitcode == 0:
            try:
                result = queue.get_nowait()
                if result is None:
                    print("MemoryError caught inside child process when computing the MEP Map")
                    # TODO : TO IMPLEMENT : implement the gestion of process and step via the YAML
                    # print("Try changing the number of process or the step value")
                    return None
                rho, MEP, grid_X, grid_Y, grid_Z, precomputed_rho_opt = result
            except Exception:
                print("Failed to get result from child process")
                return None
        elif p.exitcode == -9:
            print("\nWarning : Child process killed by the operating system (signal 9)")
            print("Warning : This might be caused by the MEP Map computation exceeding available memory.")
            # TODO : TO IMPLEMENT : implement the gestion of process and step via the YAML
            # print("Try changing the number of process or the step value")
            return None
        else:
            print(f"\nWarning : Child process exited with code {p.exitcode}")
            print("Warning : This might be caused by the MEP Map computation exceeding available memory.")
            # TODO : TO IMPLEMENT : implement the gestion of process and step via the YAML
            # print("Try changing the number of process or the step value")
            return None

    else:
        try:
            result = compute_MEP_worker_direct(ref_data, nproc, step)
            if result is None:
                print("MemoryError (direct mode)")
                return None
            rho, MEP, grid_X, grid_Y, grid_Z, precomputed_rho_opt = result
        except Exception as e:
            print(f"Error during MEP computation: {e}")
            return None
    
    # Assign results
    computed_data.derived.wavefunction.MEP_maps.rho_voxels = rho
    computed_data.derived.wavefunction.MEP_maps.MEP_voxels = MEP
    computed_data.derived.wavefunction.MEP_maps.grid_X = grid_X
    computed_data.derived.wavefunction.MEP_maps.grid_Y = grid_Y
    computed_data.derived.wavefunction.MEP_maps.grid_Z = grid_Z

    # Minima and maxima of the MEP on the molecular surface (pictures and table of the report)
    try:
        coords = np.asarray(ref_data.results.geometry.elements_3D_coords, dtype=float).reshape(-1, 3) / a0_to_A
        computed_data.derived.wavefunction.MEP_maps.surface_isovalue = mep_surface.MEP_SURFACE_ISOVALUE
        computed_data.derived.wavefunction.MEP_maps.color_range = mep_surface.color_range(ref_data.molecule.charge or 0)
        extrema = mep_surface.mep_surface_extrema(rho, MEP, grid_X, grid_Y, grid_Z, coords)
        computed_data.derived.wavefunction.MEP_maps.surface_extrema = mep_surface.merge_rings(
            extrema, coords, _view_and_centre(ref_data)[2])
    except Exception as e:
        print(f"Warning: MEP surface extrema skipped: {e}")

    # Raw (non-renormalized) reference density, reused by the Fukui functions
    return precomputed_rho_opt



def compute_population_analysis_indices(computed_data: ComputedData):

    """Atoms whose charge deviates from the mean by more than one standard deviation.

    Done for each available charge model (Mulliken, Hirshfeld, CM5) of the reference; the
    atoms kept are the union over the models. Means and standard deviations are stored for
    the report.
    """
    charges_dict = {
        'Mulliken': computed_data.results.wavefunction.Mulliken_partial_charges,
        'Hirshfeld': computed_data.results.wavefunction.Hirshfeld_partial_charges,
        'CM5': computed_data.results.wavefunction.CM5_partial_charges,
    }

    means = {'Mulliken': None, 'Hirshfeld': None, 'CM5': None}
    stds = {'Mulliken': None, 'Hirshfeld': None, 'CM5': None}
    all_indices = set()

    for method, values in charges_dict.items():
        if not values:
            continue

        data = np.array(values)
        mean = np.mean(data)
        std = np.std(data)

        means[method] = mean
        stds[method] = std
        thres_max = mean + std
        thres_min = mean - std
        
        indices = {i for i, val in enumerate(data) if val < thres_min or val > thres_max}
        all_indices.update(indices)

    derived_pop_analysis = computed_data.derived.wavefunction.pop_analysis
    derived_pop_analysis.Mulliken_mean = means['Mulliken']
    derived_pop_analysis.Mulliken_std = stds['Mulliken']
    derived_pop_analysis.Hirshfeld_mean = means['Hirshfeld']
    derived_pop_analysis.Hirshfeld_std = stds['Hirshfeld']
    derived_pop_analysis.CM5_partial_mean = means['CM5']
    derived_pop_analysis.CM5_partial_std = stds['CM5']
    derived_pop_analysis.indices = sorted(all_indices)


def compute_CDFT_global_indices(computed_data: ComputedData, ref_data: LogData, log_SP_plus: LogData, log_SP_minus: LogData):
    
    """Conceptual DFT global indices and condensed Fukui functions of the reference.

    Electron affinity A, ionization potential I, electronegativity, hardness,
    electrophilicity and DeltaN, from the energies of the reference and of the single
    points with one more and one electron less, and condensed Fukui functions from their
    Mulliken and Hirshfeld charges (calc_orb.CDFT_Indices). Values that cannot be computed
    are None.

    Parameters
    ----------
    computed_data : ComputedData
    ref_data : LogData
        Reference log.
    log_SP_plus : LogData or None
        Single point with one more electron (N+1).
    log_SP_minus : LogData or None
        Single point with one electron less (N-1).
    """
    print("\n=================================================================")
    print(" Compute : Computing CDFT Global Indices")
    print("=================================================================")

    # Compute the CDFT Indices for SP Plus, Minus and Dual if the data is available 
    # Corresponding values are set to None if the necessary data are
    # not available inside calc_orb.CDFT_Indices to compute them
    A, I, Khi, Eta, Omega, DeltaN, fplus_lambda_mulliken, fminus_lambda_mulliken, \
        fdual_lambda_mulliken, fplus_lambda_hirshfeld, fminus_lambda_hirshfeld, \
        fdual_lambda_hirshfeld = calc_orb.CDFT_Indices(ref_data, log_SP_plus, log_SP_minus)
    
    computed_data.results.wavefunction.A = A
    computed_data.results.wavefunction.I = I
    computed_data.results.wavefunction.Khi = Khi
    computed_data.results.wavefunction.Eta = Eta
    computed_data.results.wavefunction.Omega = Omega
    computed_data.results.wavefunction.DeltaN = DeltaN
    
    computed_data.results.wavefunction.fplus_lambda_mulliken = fplus_lambda_mulliken.tolist() if fplus_lambda_mulliken is not None else None
    computed_data.results.wavefunction.fminus_lambda_mulliken = fminus_lambda_mulliken.tolist() if fminus_lambda_mulliken is not None else None
    computed_data.results.wavefunction.fdual_lambda_mulliken = fdual_lambda_mulliken.tolist() if fdual_lambda_mulliken is not None else None
    computed_data.results.wavefunction.fplus_lambda_hirshfeld = fplus_lambda_hirshfeld.tolist() if fplus_lambda_hirshfeld is not None else None
    computed_data.results.wavefunction.fminus_lambda_hirshfeld = fminus_lambda_hirshfeld.tolist() if fminus_lambda_hirshfeld is not None else None
    computed_data.results.wavefunction.fdual_lambda_hirshfeld = fdual_lambda_hirshfeld.tolist() if fdual_lambda_hirshfeld is not None else None


def compute_fukui_functions(computed_data: ComputedData, ref_data: LogData, log_SP_plus: LogData, log_SP_minus: LogData, step, precomputed_rho_opt=None):
    
    """Grids of the Fukui functions f+ (from SP plus), f- (from SP minus) and of the dual
    descriptor (when both exist).

    The single points must be at the geometry of the reference; the density of the
    reference computed for the MEP is reused when given.

    Parameters
    ----------
    computed_data : ComputedData
    ref_data : LogData or None
        Model used for the grids.
    log_SP_plus, log_SP_minus : LogData or None
        Single points with N+1 and N-1 electrons.
    step : float
    precomputed_rho_opt : numpy.ndarray, optional
        Density of the reference on the same grid.
    """
    print("\n=================================================================")
    print(" Compute : Computing Fukui Functions")
    print("=================================================================")
    
    if ref_data is None:
        print("Warning: Fukui functions skipped: no log file with MO coefficients for the reference.")
        return
    for model in (ref_data, log_SP_plus, log_SP_minus):
        problem = calc_orb.electron_count_error(model) if model is not None else None
        if problem:
            print(f"Warning: Fukui functions skipped ({model.metadata.log_file}): {problem}.")
            return
    
    precomp_rho_opt = precomputed_rho_opt
    
    # Fukui dicretization for SP Plus if available
    if log_SP_plus is not None:
        print("\n-----------------------------------------------------------------")
        print(" Compute : Computing Fukui Functions : SP Plus")
        print("-----------------------------------------------------------------")
        
        rho_voxels_SPp, grid_X_spp, grid_Y_spp, grid_Z_spp, precomp_rho_opt = calc_orb.Fukui(
            ref_data,
            log_SP_plus,
            label=None,
            grid_step=step,
            nproc=nproc,
            precomputed_rho_opt=precomp_rho_opt
        )
        computed_data.derived.reactivity.fukui_functions.SP_plus = FukuiData(
            rho_voxels_SPp,
            grid_X_spp,
            grid_Y_spp,
            grid_Z_spp
        )
    
    # Fukui dicretization for SP Minus if available
    if log_SP_minus is not None:
        print("\n-----------------------------------------------------------------")
        print(" Compute : Computing Fukui Functions : SP Minus")
        print("-----------------------------------------------------------------")
        
        rho_voxels_SPm, grid_X_spm, grid_Y_spm, grid_Z_spm, precomp_rho_opt = calc_orb.Fukui(
            ref_data,
            log_SP_minus,
            label=None,
            grid_step=step,
            nproc=nproc,
            precomputed_rho_opt=precomp_rho_opt
        )
        computed_data.derived.reactivity.fukui_functions.SP_minus = FukuiData(
            rho_voxels_SPm,
            grid_X_spm,
            grid_Y_spm,
            grid_Z_spm
        )
    
    # Fukui dicretization for Dual if available
    if (
        log_SP_plus is not None
        and log_SP_minus is not None
    ):
        print("\n-----------------------------------------------------------------")
        print(" Compute : Computing Fukui Functions : Dual")
        print("-----------------------------------------------------------------")
        
        rho_voxels_dual, grid_X_dual, grid_Y_dual, grid_Z_dual = calc_orb.Fdual(ref_data,  rho_voxels_SPp, rho_voxels_SPm, grid_step=step)
        computed_data.derived.reactivity.fukui_functions.dual = FukuiData(
            rho_voxels_dual,
            grid_X_dual,
            grid_Y_dual,
            grid_Z_dual
        )


def compute_reactivity(computed_data: ComputedData, validated_data: ValidatedData, ref_data: LogData, list_data_models: List[LogData], compute_flags: ComputeFlags, step, precomputed_rho_opt=None):
    
    # Skip if doing neither CDFT Indices or Fukui Funcions
    """Conceptual DFT indices and Fukui functions, when requested.

    The single points are found by their charge: SP plus has the charge of the reference
    minus one (one more electron), SP minus the charge plus one.

    Parameters
    ----------
    computed_data : ComputedData
    validated_data : ValidatedData
    ref_data : LogData or None
        Model used for the grids.
    list_data_models : list of LogData
    compute_flags : ComputeFlags
    step : float
    precomputed_rho_opt : numpy.ndarray, optional
    """
    if (
        compute_flags.compute_CDFT_global_indices == False
        and compute_flags.compute_fukui_functions == False
    ):
        return
    
    charge_ref = validated_data.charge_ref
    charges = validated_data.charges
    
    charge_SPm = charge_ref + 1
    charge_SPp = charge_ref - 1
    
    # Trying to find the index of the logfiles corresponding to SP Plus and Minus
    SPp_index = -1
    try:
        SPp_index = charges.index(charge_SPp)
    except ValueError:
        pass

    SPm_index = -1
    try:
        SPm_index: int = charges.index(charge_SPm)
    except ValueError:
        pass
    
    # Get the logfiles for the SP Plus and Minus if they were found
    log_SP_plus: LogData = None
    if (
        SPp_index >= 0
        and SPp_index < len(list_data_models)
    ):
        log_SP_plus = list_data_models[SPp_index]
    
    log_SP_minus: LogData = None
    if (
        SPm_index >= 0
        and SPm_index < len(list_data_models)
    ):
        log_SP_minus = list_data_models[SPm_index]
    
    
    # Compute CDFT global indices (energies and charges of the reference log itself)
    if compute_flags.compute_CDFT_global_indices:
        ref_log = list_data_models[validated_data.ref_log_file_idx]
        compute_CDFT_global_indices(computed_data, ref_log, log_SP_plus, log_SP_minus)

    # Compute Fukui Functions
    if compute_flags.compute_fukui_functions:
        compute_fukui_functions(computed_data, ref_data, log_SP_plus, log_SP_minus, step, precomputed_rho_opt=precomputed_rho_opt)

def select_excited_states_indices(computed_data: ComputedData, log_idx, selection_mode="dominant_only", n=5, osc_threshold=0.1, rot_threshold=10):
    """Indexes (0-based) of the excited states shown in the report.

    - ``"all_states"``: all the states;
    - ``"first_n"``: the first n states;
    - ``"dominant_only"``: the first five states, then the states with an oscillator
      strength above osc_threshold or a rotatory strength above rot_threshold (a missing
      criterion, e.g. no rotatory strengths in the log, is ignored).

    Parameters
    ----------
    computed_data : ComputedData
    log_idx : int
        Model index.
    selection_mode : str, optional
    n : int, optional
    osc_threshold : float, optional
    rot_threshold : float, optional
        In 10^-40 erg.esu.cm.

    Returns
    -------
    list of int

    Raises
    ------
    ValueError
        Unknown selection mode.
    """
    
    et_energies = computed_data.results.excited_states.et_energies[log_idx]
    oscs = computed_data.results.excited_states.et_oscs[log_idx]
    rots = computed_data.results.excited_states.et_rot[log_idx]
    
    if et_energies == None:
        return []
    # A missing criterion (e.g. no rotatory strengths in the log) is ignored
    # instead of discarding every state.
    oscs = oscs if isinstance(oscs, (list, tuple)) else []
    rots = rots if isinstance(rots, (list, tuple)) else []
    
    n_states = len(et_energies)

    if selection_mode == "all_states":
        return list(range(n_states))
    
    elif selection_mode == "first_n":
        return list(range(min(n, n_states)))

    elif selection_mode == "dominant_only":
        selected = []

        for i in range(n_states):
            if i < 5:
                selected.append(i)
                continue

            osc_i = oscs[i] if i < len(oscs) else None
            try:
                if osc_i is not None and float(osc_i) > osc_threshold:
                    selected.append(i)
                    continue
            except (ValueError, TypeError):
                pass

            if i >= len(rots):
                continue
            rot_i = rots[i]
            if rot_i == "N/A":
                selected.append(i)
                continue
            try:
                if abs(float(rot_i)) > rot_threshold:
                    selected.append(i)
            except (ValueError, TypeError):
                pass

        return sorted(set(selected))
    
    else:
        raise ValueError(f"Unknown selection_mode: {selection_mode}")


def compute_es_Abso_and_CD_spectrum(computed_data: ComputedData, log_idx, do_abso, do_cd):
    """Absorption and CD spectra of one TD model (Gaussian broadening, FWHM in cm-1).

    The range covers the transitions with 9000 cm-1 margins (5000 to 100000 cm-1). No
    spectrum is computed when all oscillator (or rotatory) strengths are zero.

    Parameters
    ----------
    computed_data : ComputedData
    log_idx : int
    do_abso, do_cd : bool
    """
    if do_abso and do_cd:
        print("Calculating the UV absorption and CD spectrum.")
    elif do_abso:
        print("Calculating the UV absorption spectrum.")
    elif do_cd:
        print("Calculating the CD spectrum.")
    else:
        return

    et_energies = computed_data.results.excited_states.et_energies[log_idx]
    et_oscs = computed_data.results.excited_states.et_oscs[log_idx]
    et_rotats = computed_data.results.excited_states.et_rot[log_idx]
    
    # Skip if no data
    if (
        et_energies is None
        or et_oscs is None
    ):
        return 
    
    # If all oscillators are 0, skip everything
    if et_oscs.count(0.0) == len(et_oscs):
        print("The oscillators strengths are always 0.0. The absorption spectrum won't be plotted.")
        return
    
    # If Rotational strength in calculation treat Circular Dichroism      
    if et_rotats == None:
        et_rotats = []

    if len(et_energies) == 0:
        return
    td_start, td_end, numpts = TD2UVvis.spectrum_grid(et_energies, FWHM)

    # Calculating the Gaussian broadening based on wavenumbers
    heights = [[x*2.174e8/FWHM for x in et_oscs]]
    xvalues, abso_spectrum = TD2UVvis.GaussianSpectrum(td_start,td_end,numpts,et_energies,heights,FWHM)
    computed_data.derived.excited_states.xvalues[log_idx] = xvalues
    computed_data.derived.excited_states.abso_spectrum[log_idx] = abso_spectrum
    
    if (
        len(et_rotats) == 0 
        or et_rotats.count(0.0) == len(et_rotats)
    ):
        print("The rotational strengths are always 0.0. The circular dichroism spectrum won't be plotted.")
        return
    
    heights = TD2UVvis.CDheights(et_energies, et_rotats, FWHM)
    xvalues, CD_spectrum = TD2UVvis.GaussianSpectrum(td_start,td_end,numpts,et_energies,[heights], FWHM)
    computed_data.derived.excited_states.xvalues[log_idx] = xvalues
    computed_data.derived.excited_states.CD_spectrum[log_idx] = CD_spectrum


def compute_es_transitions_and_EDD(computed_data: ComputedData, ref_data: LogData, config: Config, log_idx, step, nproc, restart=False, verbose=False):
    """Electron density differences, charge-transfer indices and transition dipoles of the
    selected excited states of one TD model.

    The orbitals of the excitation amplitudes are put on the grid (calc_orb.TD), which
    gives for each selected state the density difference, the Tozer Lambda index, the
    charge-transfer distance d_CT and charge q_CT, and the dipoles drawn in the pictures
    (ground state, overlap, charge-transfer and electric transition dipoles). States
    without excitation amplitudes in the log are skipped with a warning.

    Parameters
    ----------
    computed_data : ComputedData
    ref_data : LogData or None
        Model whose orbitals are used (see _td_orbitals).
    config : Config
    log_idx : int
    step : float
    nproc : int
    restart, verbose : bool, optional
    """
    output_dir = computed_data.metadata.output_dir
    
    et_energies = computed_data.results.excited_states.et_energies[log_idx]
    et_oscs = computed_data.results.excited_states.et_oscs[log_idx]
    et_rotats = computed_data.results.excited_states.et_rot[log_idx]
    et_transitions = computed_data.results.excited_states.et_transitions[log_idx]
    et_sym = computed_data.results.excited_states.et_sym[log_idx]
    moments = computed_data.results.wavefunction.moments[log_idx]
    et_magdips_raw = computed_data.results.excited_states.et_magdips[log_idx]
    et_eldips_raw = computed_data.results.excited_states.et_eldips[log_idx]
    et_veldips_raw = computed_data.results.excited_states.et_veldips[log_idx]
    
    # Skip if missing data
    if (
        et_energies is None
        or et_oscs is None
        or et_sym is None
    ):
        return

    if ref_data is None:
        print("Warning: transitions and EDD skipped: no log file with MO coefficients for the reference.")
        return

    # The dipoles only give arrows of the (optional) dipole pictures: their absence
    # (e.g. cclib reads no transition dipoles from ORCA) must not stop the EDD.
    if et_eldips_raw is None:
        print("Note: no electric transition dipoles in the log (e.g. ORCA): no ELDIP arrow.")
    if moments is None:
        print("Note: no ground state dipole moment in the log: no GSDIP arrow.")
    
    print("Calculating the electronic density differences.")
    
    if et_rotats is None:
        et_rotats = []
    if et_transitions is None:
        et_transitions = []
    # Set the transition list to process = T_list. By default all transtions are taken into account.
    T_list = []   

    # TOOD : TO REMOVE : N/A replace by None
    # computed_data.derived.excited_states.Tozer_lambda = ["N/A"] * len(et_energies)
    # computed_data.derived.excited_states.d_ct = ["N/A"]* len(et_energies) 
    # computed_data.derived.excited_states.q_ct = ["N/A"] * len(et_energies)
    # TODO : TO CHECK : NOT USED FOR NOW
    # computed_data.derived.excited_states.mu_ct = ["N/A"] * len(et_energies)
    # computed_data.derived.excited_states.e_barycenter = ["N/A"] * len(et_energies)
    # computed_data.derived.excited_states.hole_barycenter = ["N/A"] * len(et_energies)    
    
    
    ## Discretization of all MO used in the transitions
    TD_output = None
    grid_X = None
    grid_Y = None
    grid_Z = None
    missing = [k + 1 for k, t in enumerate(et_transitions) if len(t) == 0]
    if missing:
        print(f"Warning: no excitation amplitudes in the log for the state(s) {missing}: "
              "no EDD, Lambda, d_CT and q_CT for them.")
    try:
        TD_output, grid_X, grid_Y, grid_Z = calc_orb.TD(
            ref_data, et_transitions, grid_step=step, nproc=nproc, verbose=verbose,
            deexcitations=computed_data.results.excited_states.et_deexcitations[log_idx])
    except MemoryError :
        sys.stderr.write('\n\nERROR: Memory Exception during discretization of MO used in the transitions\n')
        et_transitions = []
    
    computed_data.derived.excited_states.TD_output[log_idx] = TD_output
    computed_data.derived.excited_states.grid_X[log_idx] = grid_X
    computed_data.derived.excited_states.grid_Y[log_idx] = grid_Y
    computed_data.derived.excited_states.grid_Z[log_idx] = grid_Z
    
    # Ground state dipole drawn from the origin of the moments: (end, start).
    # cclib: moments[0] = origin (bohr; 0 for Gaussian, centre of mass for ORCA),
    # moments[1] = dipole (Debye, drawn with 1 D = 1 bohr).
    gs_eldip = None
    if moments is not None and len(moments) > 1:
        gs_eldip = (np.array(moments[0]) + np.array(moments[1]), np.array(moments[0]))
    
    selection_mode = config.output.include.excited_states.selection.mode
    first_n = config.output.include.excited_states.selection.first_n
    selected_indices = select_excited_states_indices(computed_data, log_idx, selection_mode=selection_mode, n=first_n)
    
    if (len(et_transitions)> 0):
    
        computed_data.derived.excited_states.Tozer_lambda[log_idx] = [None] * len(et_transitions)
        computed_data.derived.excited_states.d_ct[log_idx] = [None] * len(et_transitions)
        computed_data.derived.excited_states.q_ct[log_idx] = [None] * len(et_transitions)
        computed_data.derived.excited_states.all_data_dip[log_idx] = [None] * len(et_transitions)
        computed_data.derived.excited_states.transition_dipoles[log_idx] = [None] * len(et_transitions)
        
        for k in selected_indices:
            if TD_output is None or TD_output[k] is None:
                continue
            
            if (len(et_rotats) > 0) and (abs(et_rotats[k]) > 10.):
                #calculate only the elect and magnetic dipole for chiral compounds
                
                # TODO : TO CHECK : et_magdips not used
                if et_magdips_raw != None:
                    et_magdips = (np.array(et_magdips_raw[k]), np.array([0,0,0]))

            # Transition density and dipole (charge density, phase of the log's dipole)
            eldip = et_eldips_raw[k] if et_eldips_raw is not None and k < len(et_eldips_raw) else None
            magdip = (et_magdips_raw[k] if et_magdips_raw is not None and k < len(et_magdips_raw)
                      and is_chiral(et_rotats) else None)
            tdm = transition_dipole(TD_output[k], eldip, et_sym[k], et_transitions[k], magdip)
            computed_data.derived.excited_states.transition_dipoles[log_idx][k] = tdm

            print("Generating transition dipole for selected transition:", k+1),
            ct_dip = TD_output[k][2][3:]
            data_dip = {"GSDIP" : gs_eldip, "CTDIP" : ct_dip}
            if gs_eldip is None:
                del data_dip["GSDIP"]
            # Transition dipole drawn from the centre of the transition density (it does not
            # depend on the origin; the origin of the program, e.g. the centre of nuclear charge
            # of Gaussian, was near Br in HCFClBr, far from the transition)
            if tdm is not None and tdm["dipole"] is not None:
                data_dip["ELDIP"] = (tdm["centre"] + np.asarray(tdm["dipole"]), tdm["centre"])
        
            # TODO : TO CHECK : VELDIP disabled for now in data_dip
            if et_veldips_raw != None:
                et_veldips = (np.array(et_veldips_raw[k]), np.array([0,0,0]))
                #data_dip["VELDIP"] = et_veldips # Disabled for now
            
            computed_data.derived.excited_states.all_data_dip[log_idx][k] = data_dip

            
            # Extract the calculated values of the tozer_lambda, d_CT, Q_CT
            computed_data.derived.excited_states.Tozer_lambda[log_idx][k] = TD_output[k][1]
            computed_data.derived.excited_states.d_ct[log_idx][k] = TD_output[k][2][0]
            computed_data.derived.excited_states.q_ct[log_idx][k] = TD_output[k][2][1]
            
            # TODO : TO CHECK : NOT USED FOR NOW
            # # Extract calculated values of Mu_CT and e- barycenter and hole barycenter
            # computed_data.derived.excited_states.mu_ct[log_idx][k] = TD_output[k][2][2]
            # computed_data.derived.excited_states.e_barycenter[log_idx][k] = TD_output[k][2][3].tolist()
            # computed_data.derived.excited_states.hole_barycenter[log_idx][k] = TD_output[k][2][4].tolist()
            
    print("Computing Excited States Transitions and EDD done")


def compute_nto_states(computed_data, log, ref_data, log_idx, config):
    """Natural transition orbital pairs of the selected states of one TD model.

    Only the NTO coefficients are built here (data_enrichment.nto); their grids are
    computed one pair at a time when the pictures are drawn. The NTO are skipped, with a
    warning, when the orbitals of ref_data are not those of the TD calculation (atoms,
    charge, multiplicity, geometry, functional, basis set, HOMO and orbital energies must
    match), since the amplitudes would then be applied to other orbitals.

    Parameters
    ----------
    computed_data : ComputedData
    log : LogData
        TD model.
    ref_data : LogData or None
        Model whose orbitals are used.
    log_idx : int
    config : Config
    """
    name = log.metadata.log_file
    if ref_data is None or not getattr(ref_data.metadata, "discretizable", False):
        print(f"Warning: NTO skipped for {name}: reference orbital coefficients and basis are required.")
        return
    ref_wf, td_wf = ref_data.results.wavefunction, log.results.wavefunction
    if not ref_wf.MO_coefs or not ref_wf.homo_indexes:
        print(f"Warning: NTO skipped for {name}: reference MO coefficients are missing.")
        return
    # Rotation, atom order, method, spin and orbital energies must be compatible.
    try:
        ref_xyz = np.asarray(ref_data.results.geometry.elements_3D_coords, dtype=float)
        td_xyz = np.asarray(log.results.geometry.elements_3D_coords, dtype=float)
        compatible = (ref_data.molecule.atoms_Z == log.molecule.atoms_Z
                      and ref_data.molecule.charge == log.molecule.charge
                      and ref_data.molecule.multiplicity == log.molecule.multiplicity
                      and ref_xyz.shape == td_xyz.shape and ref_xyz.size > 0
                      and np.allclose(ref_xyz, td_xyz, atol=1e-4, rtol=0))
        for attr in ("functional", "dispersion", "basis_set_md5"):
            a, b = getattr(ref_data.comp_details.general, attr), getattr(log.comp_details.general, attr)
            if a and b and a != b:
                compatible = False
        if td_wf.homo_indexes is not None and ref_wf.homo_indexes != td_wf.homo_indexes:
            compatible = False
        if ref_wf.MO_energies and td_wf.MO_energies:
            if len(ref_wf.MO_energies) != len(td_wf.MO_energies):
                compatible = False
            else:
                for a, b in zip(ref_wf.MO_energies, td_wf.MO_energies):
                    count = min(len(a), len(b))
                    if count and not np.allclose(a[:count], b[:count], atol=1e-3, rtol=0):
                        compatible = False
    except (TypeError, ValueError):
        compatible = False
    if not compatible:
        print(f"Warning: NTO skipped for {name}: TD and reference orbitals/geometries are incompatible.")
        return
    print(f"NTO for {name}: assuming the TD calculation uses the reference canonical orbitals in the same order and phases.")
    transitions = log.results.excited_states.et_transitions
    if not transitions:
        print(f"Warning: NTO skipped for {name}: no excitation amplitudes were parsed.")
        return
    # Same state selection as the transitions/EDD so that both sections cover the same states
    selection = config.output.include.excited_states.selection
    indices = [i for i in select_excited_states_indices(computed_data, log_idx,
                                                        selection_mode=selection.mode,
                                                        n=selection.first_n)
               if i < len(transitions)]
    options = config.output.include.excited_states.natural_transition_orbitals
    states = []
    for i in indices:
        try:
            pairs, norm = build_nto_pairs(transitions[i], ref_wf.MO_coefs, ref_wf.homo_indexes,
                                          options.max_pairs, options.min_weight)
            complete = norm >= X_NORM_COMPLETE_THRESHOLD
            states.append({"state": i + 1, "pairs": pairs, "x_norm": norm,
                           "n_amplitudes": len(transitions[i]), "complete": complete})
            if not complete:
                print(f"Warning: NTO for {name}, state {i + 1}: printed X amplitudes are incomplete "
                      f"(sum X^2 = {norm:.3f} < {X_NORM_COMPLETE_THRESHOLD}); weights and shapes are approximate.")
        except (ValueError, IndexError, TypeError) as exc:
            print(f"Warning: NTO skipped for {name}, state {i + 1}: {exc}")
    computed_data.derived.excited_states.nto_pairs[log_idx] = states
    computed_data.derived.excited_states.nto_reference = ref_data


def compute_excited_states(computed_data: ComputedData, list_data_models: List[LogData], ref_data: LogData, config: Config, compute_flags: ComputeFlags, step):
    
    """Spectra, transitions, density differences and NTO of every TD model (type td).

    Parameters
    ----------
    computed_data : ComputedData
    list_data_models : list of LogData
    ref_data : LogData or None
    config : Config
    compute_flags : ComputeFlags
    step : float
    """
    do_abso = compute_flags.compute_es_Abso_spectrum
    do_CD = compute_flags.compute_es_CD_spectrum
    do_transitions_and_EDD = compute_flags.compute_es_transitions_and_EDD
    do_nto = compute_flags.compute_es_nto
    
    # Skip if no output enabled
    if (
        do_abso == False
        and do_CD == False
        and do_transitions_and_EDD == False
        and do_nto == False
    ):
        return
    
    print("\n=================================================================")
    print(" Compute : Computing Excited States")
    print("=================================================================")
    
    for log_idx,log in enumerate(list_data_models):
        log_file = log.metadata.log_file
        jobs = log.comp_details.general.job_type
        
        if 'td' not in jobs:
            continue

        orbitals = _td_orbitals(log, ref_data) if (do_nto or do_transitions_and_EDD) else ref_data

        if do_nto:
            compute_nto_states(computed_data, log, orbitals, log_idx, config)
        
        if (do_abso or do_CD):
            print("\n-----------------------------------------------------------------")
            print(f" Compute : Computing Excited States : Absorption and CD Spectrum")
            print(f" (Source : {log_file})")
            print("-----------------------------------------------------------------")
            
            compute_es_Abso_and_CD_spectrum(computed_data, log_idx, do_abso, do_CD)
        
        if (do_transitions_and_EDD):
            print("\n-----------------------------------------------------------------")
            print(f" Compute : Computing Excited States : Transition and EDD")
            print(f" (Source : {log_file})")
            print("-----------------------------------------------------------------")
            
            compute_es_transitions_and_EDD(computed_data, orbitals, config, log_idx, step, nproc)


def _td_orbitals(log: LogData, ref_data: LogData):
    """Orbitals used for the EDD and NTO of a TD model.

    The excitation amplitudes refer to the orbitals of the TD calculation itself (ORCA
    redoes the SCF: other phases and slightly other energies than the reference), so the TD
    model is used when it has its orbitals at the geometry of the reference, orientation
    included (the pictures are drawn with the atoms of the reference); otherwise the
    reference.

    Returns
    -------
    LogData or None
    """
    from quchemreport.quality_check_engine.conformity import same_geometry
    if (ref_data is not None and log is not ref_data and _has_orbitals(log)
            and getattr(log.metadata, "discretizable", False) and same_geometry(ref_data, log)):
        print(f"EDD and NTO from the orbitals of {log.metadata.log_file}.")
        return log
    return ref_data


def compute_optimized_es_emi_and_CPL_spectrum(computed_data: ComputedData, log_idx, emi_state, do_emi, do_CPL):
    
    """Emission and CPL spectra of one optimized excited state model.

    The emitting state is ``excited_state_number`` of the YAML file; one Gaussian band is
    drawn at its energy, from its oscillator (emission) and rotatory (CPL) strengths.

    Parameters
    ----------
    computed_data : ComputedData
    log_idx : int
    emi_state : int
        Emitting state (1-based).
    do_emi, do_CPL : bool
    """
    et_transitions = computed_data.results.excited_states.et_transitions[log_idx]
    
    # Skip if data is missing
    if (
        et_transitions is None
        or emi_state is None
    ):
        return
    
    emi_index = emi_state - 1
    if (
        emi_index < 0
        or emi_index >= len(et_transitions)
    ):
        print("Incoherent excited state optimization detected. Problem with root number")
        print(f"Emission state (user input) = {emi_state} ; expected to be within [1, {len(et_transitions)}]")
        return
    
    emi_energy = computed_data.results.excited_states.et_energies[log_idx][emi_index]
    emi_osc = computed_data.results.excited_states.et_oscs[log_idx][emi_index]
    et_rot = computed_data.results.excited_states.et_rot[log_idx]
    
    if et_rot is not None:
        emi_rotat = et_rot[emi_index]
    else:
        emi_rotat = 0.0

    td_start, td_end, numpts = TD2UVvis.spectrum_grid([emi_energy], FWHM)
    
    if emi_osc == 0.0 :
        print("The oscillator strength is 0.0. The emission spectra won't be computed.")
    else:
        # Calculating the Gaussian broadening based on wavenumbers
        heights = [[emi_osc*2.174e8/FWHM]]
        xvalues, emi_spectrum = TD2UVvis.GaussianSpectrum(td_start,td_end,numpts,[emi_energy],heights,FWHM)
        computed_data.derived.optimized_excited_states.xvalues[log_idx] = xvalues
        computed_data.derived.optimized_excited_states.emi_spectrum[log_idx] = emi_spectrum
        print("Computing Emission Spectrum done")
    
    # If no Rotational strength in calculation, skip the CPL Spectrum
    if emi_rotat  == 0.0 :
        print("The rotational strength is 0.0. The emission circular dichroism spectra won't be computed.")
        return
    else:
        # Calculating the Gaussian broadening for the CPL Spectrum
        heights = TD2UVvis.CDheights([emi_energy], [emi_rotat], FWHM)
        xvalues, CPL_spectrum = TD2UVvis.GaussianSpectrum(td_start,td_end,numpts,[emi_energy],[heights], FWHM)
        computed_data.derived.optimized_excited_states.CPL_spectrum[log_idx] = CPL_spectrum
        computed_data.derived.optimized_excited_states.xvalues[log_idx] = xvalues
        print("Computing CPL Spectrum done")


# Radiative rate from the oscillator strength (Strickler-Berg, refractive index 1):
# k_r [s-1] = f * nu^2 / STRICKLER_BERG with nu in cm-1.
STRICKLER_BERG = 1.499


def compute_phosphorescence(computed_data: ComputedData, log_idx, emi_state):
    """Phosphorescence of an optimized triplet state from its spin-orbit coupled sublevels.

    Needs an ORCA excited state optimization of a triplet with ``dosoc true`` (type opt_es):
    at the optimized geometry, the three sublevels of the triplet (spin-orbit coupled states
    whose main component is that triplet) carry small oscillator strengths towards the ground
    state, which give the phosphorescence (vertical emission, one Gaussian band per sublevel).
    The radiative rate of each sublevel follows Strickler-Berg (STRICKLER_BERG); the sublevels,
    a fraction of cm-1 apart, are assumed in thermal equilibrium (room temperature), so the
    mean rate gives the radiative lifetime. Nothing is done when the emitting state is not a
    triplet or the log has no SOC data.

    Parameters
    ----------
    computed_data : ComputedData
    log_idx : int
        Model index (type opt_es).
    emi_state : int
        ``excited_state_number`` of the YAML file (1-based, among all the states of the log).
    """
    soc = computed_data.results.excited_states.soc[log_idx]
    et_sym = computed_data.results.excited_states.et_sym[log_idx]
    if soc is None or emi_state is None or not et_sym or not 0 < emi_state <= len(et_sym):
        return
    if spin_letter(et_sym[emi_state - 1]) != "T":
        return
    label = f"T{state_number(et_sym, emi_state - 1)}"
    sublevels = [{"energy": s["energy"], "fosc": s["fosc"],
                  "k_r": s["fosc"] * s["energy"] ** 2 / STRICKLER_BERG}
                 for s in soc["states"][1:] if s["components"] and s["components"][0][1] == label]
    if not sublevels:
        print(f"Warning: no spin-orbit coupled sublevel of {label} found: no phosphorescence.")
        return
    k_mean = sum(s["k_r"] for s in sublevels) / len(sublevels)
    energies = [s["energy"] for s in sublevels]
    td_start, td_end, numpts = TD2UVvis.spectrum_grid(energies, FWHM)
    heights = [[s["fosc"] * 2.174e8 / FWHM for s in sublevels]]
    xvalues, spectrum = TD2UVvis.GaussianSpectrum(td_start, td_end, numpts, energies, heights, FWHM)
    computed_data.derived.optimized_excited_states.phosphorescence[log_idx] = {
        "state": label, "sublevels": sublevels, "k_r": k_mean,
        "lifetime": 1.0 / k_mean if k_mean > 0 else None,
        "xvalues": xvalues, "spectrum": spectrum}
    print(f"Phosphorescence of {label}: {1e7 / min(energies):.0f} nm, mean k_r = {k_mean:.3g} s-1")


def compute_optimized_es_transitions_and_EDD(computed_data: ComputedData, ref_data: LogData, emi_state, log_idx, report_type, step, nproc, restart=False, verbose=False):
    """Density difference, charge-transfer indices and dipole of the emitting state of one
    optimized excited state model (skipped for text reports).

    Parameters
    ----------
    computed_data : ComputedData
    ref_data : LogData or None
        Model whose orbitals are used: the optimized excited state itself when it has them,
        otherwise the ground-state reference (approximate, with a warning).
    emi_state : int
        Emitting state (1-based).
    log_idx : int
    report_type : str
    step : float
    nproc : int
    restart, verbose : bool, optional
    """
    if report_type == 'text':
        return
    
    et_transitions = computed_data.results.excited_states.et_transitions[log_idx]
    
    # Skip if data is missing
    if (
        et_transitions is None
        or emi_state is None
    ):
        return
    
    emi_index = emi_state - 1
    if (
        emi_index < 0
        or emi_index >= len(et_transitions)
    ):
        print("Incoherent excited state optimization detected. Problem with root number")
        print(f"Emission state (user input) = {emi_state} ; expected to be within [1, {len(et_transitions)}]")
        return

    emi_transition = et_transitions[emi_index]
    
    if ref_data is None:
        print("Warning: emission EDD skipped: no log file with MO coefficients for the reference.")
        return
    
    # Calculating Electronic density difference discretization and transition property (Tozer lambda and charge tranfer)
    print("Calculating the emission electronic density difference of optimized excited states.")
    
    computed_data.derived.optimized_excited_states.Tozer_lambda[log_idx] = [None] * len(et_transitions)
    computed_data.derived.optimized_excited_states.d_ct[log_idx] = [None] * len(et_transitions)
    computed_data.derived.optimized_excited_states.q_ct[log_idx] = [None] * len(et_transitions)
    computed_data.derived.optimized_excited_states.all_data_dip[log_idx] = [None] * len(et_transitions)
    
    ## Discretization of all MO used in the transition
    TD_output = None
    grid_X = None
    grid_Y = None
    grid_Z = None
    
    try:
        # Compute the TD_ouput with only the data for the emi_transition (only for the user-given state)
        deexc = computed_data.results.excited_states.et_deexcitations[log_idx]
        TD_output, grid_X, grid_Y, grid_Z = calc_orb.TD(
            ref_data, [emi_transition], grid_step=step, nproc=nproc,
            deexcitations=[deexc[emi_index]] if deexc and emi_index < len(deexc) else None)
    except MemoryError :
        sys.stderr.write('\n\nERROR: Memory Exception during discretization of MO used in the transitions\n')
    
    computed_data.derived.optimized_excited_states.TD_output[log_idx] = TD_output
    computed_data.derived.optimized_excited_states.grid_X[log_idx] = grid_X
    computed_data.derived.optimized_excited_states.grid_Y[log_idx] = grid_Y
    computed_data.derived.optimized_excited_states.grid_Z[log_idx] = grid_Z
    
    print(f"\nComputing Optimized Excited State transition for S{emi_state} done.")
    
    if TD_output is not None and TD_output[0] is None:
        print("Warning: no excitation amplitudes for the emitting state: no emission EDD.")
        TD_output = None

    if TD_output is not None:
        # Using TD_output[0] instead of the index of the transition as TD_output only contain the transition of index emi_index here
        ct_dip = TD_output[0][2][3:]
        data_dip = {"CTDIP" : ct_dip}
        computed_data.derived.optimized_excited_states.all_data_dip[log_idx][emi_index] = data_dip
        # Transition density and dipole of the emission (same transition, from the excited state)
        eldips = computed_data.results.excited_states.et_eldips[log_idx]
        magdips = computed_data.results.excited_states.et_magdips[log_idx]
        et_sym = computed_data.results.excited_states.et_sym[log_idx]
        magdip = (magdips[emi_index] if magdips is not None and emi_index < len(magdips)
                  and is_chiral(computed_data.results.excited_states.et_rot[log_idx]) else None)
        tdm = transition_dipole(TD_output[0], eldips[emi_index] if eldips is not None and emi_index < len(eldips) else None,
                                et_sym[emi_index] if et_sym else None, emi_transition, magdip)
        computed_data.derived.optimized_excited_states.transition_dipoles[log_idx] = [None] * len(et_transitions)
        computed_data.derived.optimized_excited_states.transition_dipoles[log_idx][emi_index] = tdm

    if TD_output is None:
        return

    # Extract the calculated values of the tozer_lambda, d_CT, Q_CT
    # Using TD_output[0] instead of the index of the transition as TD_output only contain the transition of index emi_index here
    computed_data.derived.optimized_excited_states.Tozer_lambda[log_idx][emi_index] = TD_output[0][1]
    computed_data.derived.optimized_excited_states.d_ct[log_idx][emi_index] = TD_output[0][2][0]
    computed_data.derived.optimized_excited_states.q_ct[log_idx][emi_index] = TD_output[0][2][1]
    
    # TODO : TO CHECK : NOT USED FOR NOW
    # # Extract calculated values of Mu_CT and e- barycenter and hole barycenter
    # computed_data.derived.optimized_excited_states.mu_ct[emi_index] = TD_output[emi_index][2][2]
    # computed_data.derived.optimized_excited_states.e-_barycenter[emi_index] = TD_output[emi_index][2][3].tolist()
    # computed_data.derived.optimized_excited_states.hole_barycenter[emi_index] = TD_output[emi_index][2][4].tolist()
    
    print(f"Computing Optimized Excited State EDD for S{emi_state} done")


def compute_optimized_excited_states(computed_data: ComputedData, list_data_models: List[LogData], ref_data: LogData, config: Config, report_type, compute_flags: ComputeFlags, step):
    
    """Spectra, transitions and density differences of every optimized excited state model
    (types opt_es, freq_es).

    Parameters
    ----------
    computed_data : ComputedData
    list_data_models : list of LogData
    ref_data : LogData or None
    config : Config
    report_type : str
    compute_flags : ComputeFlags
    step : float
    """
    do_emi = compute_flags.compute_optimized_es_emi_spectrum
    do_CPL = compute_flags.compute_optimized_es_CPL_spectrum
    do_transitions_and_EDD = compute_flags.compute_optimized_es_transitions_and_EDD
    
    # Skip if no output enabled
    if (
        do_emi == False
        and do_CPL == False
        and do_transitions_and_EDD == False
    ):
        return
    
    print("\n=================================================================")
    print(" Compute : Computing Optimized Excited States")
    print("=================================================================")
    
    for log_idx,log in enumerate(list_data_models):
        log_file = log.metadata.log_file
        jobs = log.comp_details.general.job_type
        
        if not any("_es" in jb for jb in jobs):
            continue
        
        emi_state = config.logfiles[log.metadata.config_idx].excited_state_number
        
        if (do_emi or do_CPL):
            print("\n-----------------------------------------------------------------")
            print(f" Compute : Computing Optimized Excited States : Absorption and CPL Spectrum")
            print(f" (Source : {log_file})")
            print("-----------------------------------------------------------------")
            
            compute_optimized_es_emi_and_CPL_spectrum(computed_data, log_idx, emi_state, do_emi, do_CPL)
            if do_emi:
                compute_phosphorescence(computed_data, log_idx, emi_state)   
        
        
        if do_transitions_and_EDD:
            print("\n-----------------------------------------------------------------")
            print(f" Compute : Computing Optimized Excited States : Transition and EDD")
            print(f" (Source : {log_file})")
            print("-----------------------------------------------------------------")

            # The excitation amplitudes of this log refer to its own orbitals, at the optimized
            # excited state geometry: use them when the log has them (pop=full gfprint).
            orbitals = log if _has_orbitals(log) else ref_data
            if orbitals is log:
                print(f"Emission EDD from the orbitals and geometry of {log_file}.")
            elif ref_data is not None:
                print(f"Warning: {log_file} has no MO coefficients or basis set (add pop=full gfprint to the "
                      "excited state optimization): the emission EDD, Lambda, d_CT and q_CT use the orbitals "
                      f"and geometry of the ground-state reference {ref_data.metadata.log_file} (approximate).")
            computed_data.derived.optimized_excited_states.orbitals_log_idx[log_idx] = next(
                (i for i, model in enumerate(list_data_models) if model is orbitals), None)

            compute_optimized_es_transitions_and_EDD(computed_data, orbitals, emi_state, log_idx, report_type, step, nproc)


def _has_orbitals(model: LogData):
    """Whether a model has what the grids need: MO coefficients, basis set and AO names."""
    general = model.comp_details.general
    mo_coefs = model.results.wavefunction.MO_coefs
    return (mo_coefs is not None and len(mo_coefs) in (1, 2)
            and general.basis_set is not None and general.ao_names is not None)


def compute_data(list_data_models: List[LogData], validated_data: ValidatedData, config: Config, compute_flags: ComputeFlags, step = obk_step):
    """Run all the computations of the report.

    Parameters
    ----------
    list_data_models : list of LogData
        Parsed models.
    validated_data : ValidatedData
        Conformity results (reference, model used for the grids).
    config : Config
    compute_flags : ComputeFlags
        Computations to run.
    step : float, optional
        Grid step (bohr), obk_step by default.

    Returns
    -------
    ComputedData
    """
    global nproc
    if config.resources.nproc is not None:
        nproc = config.resources.nproc
    print(f"Grid computations with nproc = {nproc}" + (" (resources.nproc)" if config.resources.nproc else " (default)"))
    
    # Init the ComputedData object
    len_logfiles = len(list_data_models)
    computed_data: ComputedData = init_computed_data(len_logfiles)
    
    # Manage output dir
    # folder chosen at launch (execution_engine.output_folder)
    output_dir = config.output.report_folder()
    os.makedirs(output_dir, exist_ok=True)
    computed_data.metadata.output_dir = output_dir
    
    # Get report type
    report_type = config.output.verbosity
    
    # Copy idx for reference logfile
    computed_data.metadata.ref_log_file_idx = validated_data.ref_log_file_idx
    ref_data = validated_data.data_for_discretization
    
    # Extract and aggregate data from each log file (simple data from the reference log)
    extract_simple_data(list_data_models, list_data_models[validated_data.ref_log_file_idx], computed_data)
    
    # Compute molecule representations
    if compute_flags.compute_molecule_representations:
        compute_molecule_representations(computed_data)
    
    # Compute MO_labels and MO_indexes for the user-defined MO_list
    if compute_flags.compute_all_MO_labels_and_indexes:
        compute_all_MO_labels_and_indexes(computed_data, list_data_models, config)
    
    # Compute MO Diagram Data
    if compute_flags.compute_MO_diagrams:
        compute_MO_diagrams(computed_data, ref_data, nproc, step)
    
    precomputed_rho_opt = None
    
    # Compute MEP Maps
    if compute_flags.compute_MEP_maps:
        precomputed_rho_opt = compute_MEP_maps(computed_data, ref_data, nproc, step)

    # Compute indices for the Population Analysis
    if compute_flags.compute_population_analysis_indices:
        compute_population_analysis_indices(computed_data)
    
    # Compute Reactivity Section
    compute_reactivity(computed_data, validated_data, ref_data, list_data_models, compute_flags, step, precomputed_rho_opt=precomputed_rho_opt)
    
    # Compute Excited States
    compute_excited_states(computed_data, list_data_models, ref_data, config, compute_flags, step)

    # Compute Optimized Excited State
    compute_optimized_excited_states(computed_data, list_data_models, ref_data, config, report_type, compute_flags, step)
        
    
    return computed_data
