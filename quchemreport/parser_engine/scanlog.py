"""Reading of the log files of quantum chemistry programs into LogData models.

A log file may contain several calculations (Gaussian Link1 jobs, ORCA compound jobs): it is
first split into steps, written to a temporary folder, and each completed step is read on its
own. A step is read with cclib (all the results) and Open Babel (molecule: formula,
connectivity, InChI, SMILES), then copied, section by section, into a LogData model
(utility_services.log_data). Values that cclib does not give for a log are left to None.

Gaussian and ORCA are supported; GAMESS and NWChem are accepted without splitting and untested.
A few values are read directly in the log, because cclib does not give them: the sum of
electronic and thermal energies, and the functional of ORCA calculations.
"""

import os
import re
import sys
import traceback
import json
import pickle
import hashlib
import cclib
import openbabel.pybel as pybel
import numpy as np
import openbabel as ob
import sklearn.preprocessing

from quchemreport.config.config import Config
from quchemreport.utility_services.log_data import LogData
from quchemreport.parser_engine.orca_soc import read_orca_soc

# constants
CstBohr2Ang = 0.52917721092
CstHartree2eV = 27.21138505
CstHartree2cm1 = 219474.6313708
scanlog_version = "1.0.2"

class ScanlogException(Exception):
    """Error while reading a log file (message for the user)."""
    pass


def split_logfile_gaussian(logfile, log_storage_path, verbose):
    """Split a Gaussian log file into its completed steps (one per "Normal termination").

    Each step is written as ``<name>_step_<k>.log`` in log_storage_path, with the Gaussian
    copyright line that cclib needs to recognize the program. A last step without normal
    termination (interrupted job) is ignored with a warning.

    Parameters
    ----------
    logfile : str
    log_storage_path : str
        Folder of the step files.
    verbose : bool

    Returns
    -------
    list of str
        Paths of the step files (empty when no step terminated normally).
    """
    log_files = []
    
    TERM = "Normal termination"
    
    with open(logfile, 'r') as log_fd:
        lines = log_fd.readlines()
    nbl = len(lines)
    
    file_cnt = 0
    base_fname = os.path.basename(logfile).rsplit('.', 1)[0]
    log_pat = os.path.join(log_storage_path, "%s_step_%s.log" % (base_fname, "%d"))
    cur_log = log_pat % file_cnt
    
    if verbose:
        print(">>> Processing", cur_log, "...")
        
    # FLAG to add copyright at the beginning of each step.
    cur_log_fd = open(cur_log, "w")
    file_start = True
    trailing_job = False  # a new job started after the last normal termination
    for cur_l, line in enumerate(lines):
        if file_start:
            cur_log_fd.write(" Copyright (c) 1988,1990,1992,1993,1995,1998,2003,2009,2013,\n")
            cur_log_fd.write("            Gaussian, Inc.  All Rights Reserved.\n")
            file_start = False
            trailing_job = False
            
        cur_log_fd.write(line)
        if " Initial command:" in line or " Entering Link 1" in line:
            trailing_job = True
        if line.find(TERM) > -1 :
            file_start = True
            trailing_job = False
            if verbose:
                print("=> ",line)
            log_files.append(cur_log)
            cur_log_fd.close()
            file_cnt += 1
            if nbl > (cur_l + 1) :
                cur_log = log_pat % file_cnt
                if verbose:
                    print(">>> Processing", cur_log, "...")
                cur_log_fd = open(cur_log, "w")
                
    if not cur_log_fd.closed :
        cur_log_fd.close()
    if trailing_job and log_files:
        print(f"Warning: the last step of '{logfile}' has no '{TERM}' (interrupted job?); it is ignored.")
        
    return log_files


def split_logfile_gamess(logfile, log_storage_path, verbose):
    """One step per "TERMINATED NORMALLY" of a GAMESS log, without splitting (untested)."""
    log_files = []
    
    TERM = "TERMINATED NORMALLY"
    
    with open(logfile, 'r') as log_fd:
        lines = log_fd.readlines()
        
    for line in lines:
        if line.find(TERM) > -1 :
            if verbose:
                print("=> ",line)
            log_files.append(logfile)
            
    return log_files


def split_logfile_orca(logfile, log_storage_path, verbose):
    """Split an ORCA log file into its completed steps (one per "Timings for individual
    modules").

    Each step is written as ``<name>_step_<k>.log`` with the ORCA header and the program
    version, so that cclib recognizes it. A last step without termination is ignored with a
    warning.

    Parameters
    ----------
    logfile : str
    log_storage_path : str
    verbose : bool

    Returns
    -------
    list of str
    """
    log_files = []
    
    TERM = "Timings for individual modules:"
    program_version = ""
    
    with open(logfile, 'r') as log_fd:
        lines = log_fd.readlines()
    nbl = len(lines)

    file_cnt = 0
    base_fname = os.path.basename(logfile).rsplit('.', 1)[0]
    log_pat = os.path.join(log_storage_path, "%s_step_%s.log" % (base_fname, "%d"))
    cur_log = log_pat % file_cnt
    
    if verbose:
        print(">>> Processing", cur_log, "...")
        
    # FLAG to add software name and version at the beginning of each step.
    cur_log_fd = open(cur_log, "w")
    file_start = True
    trailing_job = False  # a new job started after the last normal termination
    for cur_l, line in enumerate(lines):
        if file_start:
            cur_log_fd.write("\nO   R   C   A\n")
            if program_version != "" :
                cur_log_fd.write("\n"+program_version+"\n")
            file_start = False
            trailing_job = False
        if "CARTESIAN COORDINATES (ANGSTROEM)" in line:
            trailing_job = True

        if "Program Version" in line:
            program_version = line

        cur_log_fd.write(line)
        if line.find(TERM) > -1 :
            file_start = True
            trailing_job = False
            if verbose:
                print("=> ",line)
            log_files.append(cur_log)
            cur_log_fd.close()
            file_cnt += 1
            if nbl > (cur_l + 1) :
                cur_log = log_pat % file_cnt
                if verbose:
                    print(">>> Processing", cur_log, "...")
                cur_log_fd = open(cur_log, "w")
                
    if not cur_log_fd.closed :
        cur_log_fd.close()
    if trailing_job and log_files:
        print(f"Warning: the last step of '{logfile}' has no '{TERM}' (interrupted job?); it is ignored.")
        
    return log_files


def split_logfile(logfile, solver, log_storage_path="", verbose=False):
    """Split a log file into its completed steps, according to the program.

    Parameters
    ----------
    logfile : str
    solver : str
        Program: gaussian, orca (gamess, nwchem: untested).
    log_storage_path : str, optional
    verbose : bool, optional

    Returns
    -------
    list of str
        Paths of the step files.

    Raises
    ------
    ScanlogException
        Unsupported program, or the file could not be split.
    """
    try:
        if verbose:
            print(">>> SOLVER:", solver)

        def unsupported_solver(*args, **kwargs):
            raise ScanlogException(f"Unsupported solver : {solver}.")

        dispatch = {
            "gaussian": split_logfile_gaussian,
            ### TODO: GAMESS not tested, accepted here only for Riken
            ### DB insertion purpose (only OPT mono step)
            "gamess": split_logfile_gamess,
            ### TODO: NWChem not tested
            ### we assume normal termination already tested
            "nwchem": lambda *_: [logfile],
            "orca": split_logfile_orca,
        }

        handler = dispatch.get(solver, unsupported_solver)
        log_files = handler(logfile, log_storage_path, verbose)

        if verbose:
            print(">>> Steps :", log_files, "\n")

        return log_files

    except ScanlogException as err:
        raise err
    except Exception:
        if verbose:
            traceback.print_exc()
        raise ScanlogException("File splitting failed.")


def nuclear_repulsion_energy(data, slice_id=-1):
    """Nuclear repulsion energy of a geometry of a cclib result, in hartree.

    The coordinates are rounded to 1e-5 Angstrom, so that the same geometry written by two
    calculations gives the same value: it identifies the conformer in the conformity tests
    (conformity.same_nre).

    Parameters
    ----------
    data : cclib ccData
    slice_id : int, optional
        Geometry of the calculation (last one by default; 0 for the starting geometry).

    Returns
    -------
    float
    """
    nre = 0.0
    for i in range(data.natom):
        ri = np.array([float("%.5f" % k) for k in data.atomcoords[slice_id][i]])
        zi = data.atomnos[i]
        for j in range(i + 1, data.natom):
            rj = np.array([float("%.5f" % k) for k in data.atomcoords[slice_id][j]])
            zj = data.atomnos[j]
            d = np.linalg.norm(ri - rj)
            nre += zi * zj / d
    return float("%.5f" % (nre * CstBohr2Ang))


def _try_key_insertion(res_json, key, obj, obj_path=[], nullable=True):
    # case : dictionary
    """Former helper to copy a value into a dict (unused)."""
    if obj.__class__ == dict :
        try:
            if obj_path:
                d = obj.copy()
                for k in obj_path:
                    d = d[k]
                res_json[key] = d
        except Exception as e:
            if not nullable:
                raise ScanlogException("Fatal : error occured for required key %s" % key)
            # else error occured but key is not required
    # case : simple object
    elif obj != 'N/A':
        res_json[key] = obj
    elif not nullable:        
        raise ScanlogException("Fatal : key %s is N/A but is required" % key)
    # else obj is 'N/A' ans is ignored

def try_update_value(
    data_model: LogData,
    attr_path: str,
    obj_or_dict,
    dict_path: list = None,
    nullable: bool = True
):
    """Copy a value, or a value of a nested dict, into an attribute of a LogData model.

    Parameters
    ----------
    data_model : LogData
    attr_path : str
        Dotted path of the attribute, e.g. ``"comp_details.general.functional"``.
    obj_or_dict : object or dict
        The value, or the dict (cclib JSON, cclib metadata) that contains it.
    dict_path : list of str, optional
        Keys of the value in obj_or_dict; the value is obj_or_dict itself when omitted.
    nullable : bool, optional
        When True (default), a missing key or a value refused by the model leaves the
        attribute unchanged; when False, it raises.

    Raises
    ------
    AttributeError
        attr_path is not an attribute of the model (programming error).
    ScanlogException, ValueError
        Missing or invalid value with nullable False.
    """
    try:
        # Extract value from nested dict if dict_path is provided
        value = obj_or_dict
        if dict_path:
            for key in dict_path:
                value = value[key]

        # Traverse the attr_path in data_model to get the final target
        parts = attr_path.split(".")
        target = data_model
        for attr in parts[:-1]:
            if not hasattr(target, attr):
                raise AttributeError(attr)
            target = getattr(target, attr)

        final_attr = parts[-1]
        if not hasattr(target, final_attr):
            raise AttributeError(final_attr)

        setattr(target, final_attr, value)

    except KeyError as e:
        if nullable:
            return
        raise ScanlogException(f"Missing key {e} in provided data and nullable=False for path {dict_path}")

    except AttributeError as e:
        raise AttributeError(f"Fatal: Invalid attribute path '{e}' in 'MODEL_OBJECT.{attr_path}'")

    except Exception as e:
        if not nullable:
            raise ValueError(f"Fatal: Failed to assign value to 'MODEL_OBJECT.{attr_path}': {e}")



def general_param_subsection(data_model: LogData, data_json, data, obdata):
    """Fill ``comp_details.general``: program and version, methods, functional and dispersion,
    basis set (pickled with its MD5 hash, to compare basis sets), atomic orbitals, closed or
    open shell, solvent, SCF targets and core electrons.
    """
    all_unique_theory = np.array([])
    try:
        all_unique_theory = np.unique(data.metadata['methods'])
        if len(all_unique_theory) > 1:
            theo_array = np.array(data.metadata['methods'])
            _, theo_indices = np.unique(theo_array, return_index=True)
            theo_indices.sort()
            theo_array = theo_array[theo_indices]
        else:
            theo_array = all_unique_theory
    except:
        theo_array = 'N/A'
        
    if theo_array.__class__ != str :
        if len(theo_array) > 0:
            theo_array = theo_array.tolist() if (theo_array != 'N/A').any() else 'N/A'
        else:
            theo_array = 'N/A'
            
    if len(all_unique_theory) > 0:
        all_unique_theory = all_unique_theory.tolist() if (all_unique_theory != 'N/A').any() else 'N/A'
    else:
        all_unique_theory = 'N/A'
        
    methods = data.metadata.get('methods', ['N/A'])

    try_update_value(data_model, "comp_details.general.package", data.metadata, ['package'])
    if data_model.comp_details.general.package != None:
        data_model.comp_details.general.package = data_model.comp_details.general.package.lower()
    try_update_value(data_model, "comp_details.general.package_version", data.metadata, ['package_version'])
    try_update_value(data_model, "comp_details.general.all_unique_theory", all_unique_theory)
    
    if len(methods) > 0:
        try_update_value(data_model, "comp_details.general.last_theory", methods[-1])
        
    try_update_value(data_model, "comp_details.general.list_theory", theo_array)
    try_update_value(data_model, "comp_details.general.functional", data.metadata, ['functional'])
    try_update_value(data_model, "comp_details.general.dispersion", data.metadata, ['dispersion'])
    try_update_value(data_model, "comp_details.general.basis_set_name", data.metadata, ['basis_set'])
    
    # basis set Pickle version
    try:
        basis_str = pickle.dumps(data.gbasis, protocol=0)
        basis_hash = hashlib.md5(basis_str).hexdigest()
        try_update_value(data_model, "comp_details.general.basis_set", basis_str.decode()) # "%s"  % basis_str[2:-1])
        try_update_value(data_model, "comp_details.general.basis_set_md5", basis_hash)
    except:
        pass
    
    try_update_value(data_model, "comp_details.general.basis_set_size", data_json, ['properties', 'orbitals', 'basis number'])
    try_update_value(data_model, "comp_details.general.ao_names", data_json, ['atoms', 'orbitals', 'names'])
    
    try:
        # A ROHF/ROKS open shell (multiplicity > 1) also has a single set of MO energies
        data_model.comp_details.general.is_closed_shell = repr(data.mult == 1
                                                            and (len(data.moenergies) == 1
                                                                 or np.allclose(*data.moenergies, atol=1e-6)))
    except:
        pass
    
    # TODO : 'integration grid' not found in data by cclib (is it because of version differences or just not present in all formats?)
    try_update_value(data_model, "comp_details.general.integration_grid", data_json, ['properties', 'integration grid'])
    # Implicit solvent, as cclib reads it (metadata solvent_model, solvent_name, solvent_params).
    # Until 27/11/2026 the keys 'solvent' and 'scrf' were read, which cclib does not write: the
    # solvent never reached the report.
    data_model.comp_details.general.solvent, data_model.comp_details.general.solvent_reaction_field = \
        solvent_description(data.metadata)
    try_update_value(data_model, "comp_details.general.scf_targets", data_json, ['optimization', 'scf', 'targets'])
    try_update_value(data_model, "comp_details.general.core_electrons_per_atoms", data_json, ['atoms', 'core electrons'])


def split_deexcitations(excited_states):
    """Moves the de-excitation amplitudes (Y) of et_transitions to et_deexcitations.

    cclib stores the de-excitations, when the log prints them (Gaussian with IOp(9/40), Q-Chem),
    in etsecs with the orbitals reversed: from the virtual to the occupied orbital. They are
    kept apart here, as (occupied, virtual, Y), so that et_transitions holds the excitations X
    only, as all its users expect (density differences, NTO, table of the main excitations),
    and the transition density can use X + Y (calc_orb.TD). An entry is a de-excitation when
    it goes from a higher to a lower orbital of the same spin (the occupied orbitals of the
    ground state come first)."""
    transitions = excited_states.et_transitions
    if not transitions:
        return
    kept, deexcitations = [], []
    for state in transitions:
        x, y = [], []
        for start, end, c in state:
            if start[1] == end[1] and start[0] > end[0]:
                y.append((tuple(end), tuple(start), c))
            else:
                x.append((tuple(start), tuple(end), c))
        kept.append(x)
        deexcitations.append(y)
    excited_states.et_transitions = kept
    excited_states.et_deexcitations = deexcitations


def solvent_description(metadata):
    """(description, model) of the implicit solvent read by cclib: ("water (SMD-IEFPCM,
    dielectric constant 78.36)", "SMD-IEFPCM"), or ("none (gas phase)", None)."""
    model = metadata.get("solvent_model")
    if not model:
        return "none (gas phase)", None
    name = metadata.get("solvent_name")
    epsilon = (metadata.get("solvent_params") or {}).get("epsilon")
    details = model + (f", dielectric constant {epsilon:.2f}" if epsilon is not None else "")
    return f"{name or 'unnamed solvent'} ({details})", model


def geometry_param_subsection(data_model: LogData, data_json, data, obdata):
    """Fill ``comp_details.geometry`` (convergence targets of the optimization)."""
    try_update_value(data_model,"comp_details.geometry.geometric_targets", data_json, ['optimization', 'geometric targets'])


def freq_param_subsection(data_model: LogData, data_json, data, obdata):
    """Fill ``comp_details.freq`` (temperature, anharmonicity)."""
    try_update_value(data_model, "comp_details.freq.temperature", data_json, ['properties', 'temperature'])
    try_update_value(data_model, "comp_details.freq.anharmonicity", data_json, ['vibrations', 'anharmonicity constants'])


def td_param_subsection(data_model: LogData, data_json, data, obdata):
    """Fill ``comp_details.excited_states`` (number of excited states)."""
    # Number of states from the cclib attribute etenergies: the JSON entry used before
    # ('transitions' / 'electronic transitions') gave 3 instead of 20 states for an ORCA TD
    # with triplets and SOC, and 1 instead of 10 for an ORCA excited-state optimization.
    etenergies = getattr(data, "etenergies", None)
    if etenergies is not None and len(etenergies):
        data_model.comp_details.excited_states.nb_et_states = len(etenergies)
    ## TODO:
    # data_model.comp_details.excited_states.TDA = 'N/A' # TODO: test Tamm Damcoff approx.
    # data_model.comp_details.excited_states.et_sym_constraints = 'N/A/'


def wavefunction_results_subsection(data_model: LogData, data_json, data, obdata):
    """Fill ``results.wavefunction``: orbitals, total energy, charges, moments, SCF values.

    The MO energies, symmetries and coefficients are kept up to the first virtual orbital
    above 10 eV, and at least 30 virtual orbitals above the HOMO, to keep the model small.
    Energies that Gaussian cannot print (``**********``, e.g. the 1s orbital of iodine in an
    all-electron basis set) are NaN in cclib: they become -inf, so that the energies stay
    usable. When the MO data are incomplete, a warning is printed and no grid can be made
    from this log.
    """
    try_update_value(data_model, "results.wavefunction.homo_indexes", data_json, ['properties', 'orbitals', 'homos'])
    try_update_value(data_model, "results.wavefunction.MO_energies", data_json, ['properties', 'orbitals', 'energies'])
    try_update_value(data_model, "results.wavefunction.MO_sym", data_json, ['properties', 'orbitals', 'molecular orbital symmetry'])
    # MO_number, MO_energies, MO_sym, MO_coefs

    try:
        try_update_value(data_model, "results.wavefunction.MO_number", data_json, ['properties', 'orbitals', 'MO number'], 
                            nullable=False) # not nullable in this context, exception catched.
        # Energies too low for the printed format (Gaussian "**********", e.g. 1s of iodine in an
        # all-electron basis set) are NaN in cclib and null in its JSON, which made the whole
        # MO_energies list invalid (None): they become -inf, in every spin.
        for moen in data.moenergies:
            moen[np.isnan(moen)] = -np.inf
        w_cut = np.where(data.moenergies[-1] > 10.)
        b_cut = min(max(w_cut[0][0] if len(w_cut[0]) > 0 else 0,
                        data.homos.max() + 31),
                    len(data.moenergies[-1]))
        try_update_value(data_model, "results.wavefunction.MO_number_kept", int(b_cut))

        # prune energies and sym
        try_update_value(data_model, "results.wavefunction.MO_energies", [[float(e) for e in moen[:b_cut]] for moen in data.moenergies])
        
        # Problem if there is no MO_sym like in ORCA !
        # TODO : Check if correct logique (if there is no MO_sys, while do we use a MO_Sym value to calculate it ?)
        if data_json.get('properties', {}).get('orbitals', {}).get('molecular orbital symmetry'):
            try_update_value(data_model, "results.wavefunction.MO_sym", [mosym[:b_cut] for mosym in data_json['properties']['orbitals']['molecular orbital symmetry']])
        # if "MO_sym" in section : 
        #     try_update_value(data_model, "results.wavefunction.MO_sym", [mosym[:b_cut] for mosym in section["MO_sym"]])
        
        # prune mocoeffs (Sparse version removed)
        mo_coefs = []
        # take last mocoeffs  (-2 with alpha/beta or -1)
        nb_coef = -2 if len(data.moenergies) == 2 else -1
        for a in data.mocoeffs[nb_coef:]:
            mo_coefs.append(a.tolist())
            
        # data insertion into JSON
        data_model.results.wavefunction.MO_coefs = mo_coefs
    except Exception as e:
        # partial MO data: no orbital pictures can be made from this log
        print(f"Warning: MO data of {getattr(data, 'quchemreport_logfile', '?')} could not be read "
              f"({type(e).__name__}: {e}); no orbital grids from this log.")
    
    try_update_value(data_model, "results.wavefunction.total_molecular_energy", data_json, ['properties', 'energy', 'total'])
    
    # eV to Hartree conversion
    try: 
        try_update_value(data_model, "results.wavefunction.total_molecular_energy", data_json['properties']['energy']['total'] / CstHartree2eV)
    except:
        ## TODO: pb with SP
        pass # SP ? failure ?
    
    try_update_value(data_model, "results.wavefunction.Mulliken_partial_charges", data_json, ['properties', 'partial charges', 'mulliken'])
    
    try: 
        data_model.results.wavefunction.moments = data.moments 
    except:
        pass # Dipolar moments not present depending on the verbosity of log. 
    
    try:
        data_model.results.wavefunction.SCF_values = data_json['optimization']['scf']['values'][-1][-1] 
    except:
        pass
    
    try_update_value(data_model, "results.wavefunction.virial_ratio", data_json, ['optimization', 'scf', 'virialratio'])
    
    # TODO: # data_model.results.wavefunction.Hirshfeld_partial_charges  = 'N/A' # see scanlog
    try:
        data_model.results.wavefunction.Hirshfeld_partial_charges = data.atomcharges["hirshfeld"].tolist()
    except:
        pass
    # CM5 charges (Gaussian pop=hirshfeld prints them with the Hirshfeld ones; ORCA has none).
    # Not read before 27/11/2026, although the report has a CM5 column.
    try:
        data_model.results.wavefunction.CM5_partial_charges = data.atomcharges["cm5"].tolist()
    except (AttributeError, KeyError):
        pass


def geom_results_subsection(data_model: LogData, data_json, data, obdata):
    """Fill ``results.geometry``: nuclear repulsion energy, optimization status, coordinates
    (Angstrom) and geometric convergence values.
    """
    try_update_value(data_model, "results.geometry.nuclear_repulsion_energy_from_xyz", nuclear_repulsion_energy(data))
    try_update_value(data_model, "results.geometry.OPT_DONE", data_json, ['optimization', 'done'])
    try_update_value(data_model, "results.geometry.elements_3D_coords", data_json, ['atoms', 'coords', '3d'])
    try_update_value(data_model, "results.geometry.geometric_values", data_json, ['optimization', 'geometric values'])


def _electronic_thermal_energy(logfile):
    """Last sum of electronic and thermal energies of a Gaussian or ORCA log, in hartree.

    Read in the log because cclib does not give it.

    Parameters
    ----------
    logfile : str

    Returns
    -------
    float or None
        None when the log has no thermochemistry.
    """
    import re
    if not logfile or not os.path.isfile(logfile):
        return None
    patterns = (r"Sum of electronic and thermal Energies=\s*(-?\d+\.\d+)",    # Gaussian
                r"Total thermal energy\s+(-?\d+\.\d+)\s+Eh")                 # ORCA
    value = None
    with open(logfile, errors="replace") as fd:
        for line in fd:
            for pattern in patterns:
                found = re.search(pattern, line)
                if found:
                    value = float(found.group(1))
    return value


_ORCA_FUNCTIONAL_NAMES = {name.lower() for name in (
    "HFS", "LDA", "LSD", "VWN", "VWN3", "VWN5", "PWLDA", "BP86", "BP", "BLYP", "OLYP", "GLYP", "XLYP", "PW91",
    "mPWPW", "mPWLYP", "PBE", "RPBE", "revPBE", "PWP", "B97-D", "B97-D3", "B97-3c", "TPSS", "M06L", "M06-L",
    "SCAN", "rSCAN", "r2SCAN", "r2SCAN-3c", "B97M-V", "B97M-D3BJ", "B97M-D4", "B1LYP", "B3LYP", "B3LYP/G",
    "O3LYP", "X3LYP", "B1P", "B3P", "B3PW", "PW1PW", "mPW1PW", "mPW1LYP", "PBE0", "revPBE0", "PBEh-3c",
    "HSE06", "TPSSh", "TPSS0", "M06", "M062X", "M06-2X", "MN15", "MN15L", "PW6B95", "BHandHLYP", "r2SCANh",
    "r2SCAN0", "r2SCAN50", "wB97", "wB97X", "wB97X-D3", "wB97X-D3BJ", "wB97X-D4", "wB97X-V", "wB97M-V",
    "wB97M-D3BJ", "wB97M-D4", "wB97X-3c", "CAM-B3LYP", "LC-BLYP", "LC-PBE", "LC-wPBE", "B2PLYP", "B2GP-PLYP",
    "mPW2PLYP", "DSD-BLYP", "DSD-PBEP86", "DSD-PBEB95", "PWPB95", "wB2PLYP", "wB2GP-PLYP", "RSX-QIDH",
    "PBE-QIDH", "PBE0-DH", "B3LYP-D3", "PBE0-D3")}


def _orca_functional(logfile, keywords=None):
    """Functional of an ORCA calculation, read in the log (cclib does not give it for ORCA).

    The functional keyword of the input is followed by the definition printed by ORCA
    (exchange, correlation, HF exchange and DFT scalings, range separation, perturbative
    correction), so that two calculations with the same keyword but modified parameters
    are seen as different levels of theory.

    Parameters
    ----------
    logfile : str
    keywords : list of str, optional
        Keywords of the input line (cclib metadata["keywords"]).

    Returns
    -------
    str or None
        e.g. ``"B3LYP (Exchange B88, Correlation LYP, HF exchange 0.2, ...)"``; None
        when the log has no DFT block.
    """
    import re
    if not logfile or not os.path.isfile(logfile):
        return None
    fields = (("Exchange", r"^\s*Exchange Functional\s+Exchange\s+\.+\s*(\S+)"),
              ("Correlation", r"^\s*Correlation Functional\s+Correlation\s+\.+\s*(\S+)"),
              ("HF exchange", r"ScalHFX\s+\.+\s*(\S+)"),
              ("DF-GGA-X", r"ScalDFX\s+\.+\s*(\S+)"),
              ("DF-GGA-C", r"ScalDFC\s+\.+\s*(\S+)"),
              ("range separation", r"RangeSepMu\s+\.+\s*(\S+)"),
              ("perturbative correction", r"Perturbative correction\s+\.+\s*(\S+)"))
    found = {}
    with open(logfile, errors="replace") as fd:
        for line in fd:
            for name, pattern in fields:
                match = re.search(pattern, line)
                if match and name not in found:
                    found[name] = match.group(1)
    if "Exchange" not in found:
        return None

    def value(text):
        """Number formatted with %g, or the text unchanged."""
        try:
            return "%g" % float(text)
        except ValueError:
            return text

    if value(found.get("perturbative correction", "0")) == "0":
        found.pop("perturbative correction", None)
    definition = ", ".join(f"{name} {value(found[name])}" for name, _ in fields if name in found)
    names = [k for k in (keywords or []) if str(k).lower() in _ORCA_FUNCTIONAL_NAMES]
    return f"{names[0]} ({definition})" if names else definition


def freq_results_subsection(data_model: LogData, data_json, data, obdata):
    """Fill ``results.freq``: thermochemistry (entropy, enthalpy, free energy, zero-point
    correction), frequencies, IR and Raman intensities, symmetries, displacements,
    polarizabilities, anharmonicity.
    """
    try_update_value(data_model, "results.freq.entropy", data_json, ['properties', 'entropy'])
    try:
        try_update_value(data_model, "results.freq.entropy", float("%.9f" % data_json['properties']['entropy']))
    except:
        pass
    
    try_update_value(data_model, "results.freq.enthalpy", data_json, ['properties', 'enthalpy'])
    try_update_value(data_model, "results.freq.free_energy", data_json, ['properties', 'energy', 'free energy'])
    # cclib: zero-point vibrational energy correction (hartree/particle), not E + ZPE
    try_update_value(data_model, "results.freq.zero_point_energy", data_json, ['properties', 'energies', 'zero-point correction'])
    # Sum of electronic and thermal energies: not provided by cclib, read in the log
    # (hartree) by logfile_to_obj
    try_update_value(data_model, "results.freq.vibrational_freq", data_json, ['vibrations', 'frequencies'])
    try_update_value(data_model, "results.freq.vibrational_int", data_json, ['vibrations', 'intensities', 'IR'])
    
    # here NWChem
    try:
        data_model.results.freq.polarizabilities = data.polarizabilities[0].tolist()
    except:
        pass
    
    try_update_value(data_model, "results.freq.vibrational_sym", data_json, ['vibrations', 'vibration symmetry'])
    try_update_value(data_model, "results.freq.vibration_disp", data_json, ['vibrations', 'displacement'])
    
    # here Gaussian
    try_update_value(data_model, "results.freq.vibrational_anharms", data_json, ['vibrations', 'anharmonicity constants'])
    try_update_value(data_model, "results.freq.vibrational_raman", data_json, ['vibrations', 'intensities', 'raman'])

def td_results_subsection(data_model: LogData, data_json, data, obdata):
    """Fill ``results.excited_states``: energies, oscillator and rotatory strengths, symmetries,
    excitation amplitudes and transition dipoles of the excited states.
    """
    # Energies in cm-1 from the cclib attribute etenergies, for every program. The cclib JSON
    # gives ORCA's energies as a dict of spectra in hartree (e.g. "ABSORPTION SPECTRUM VIA
    # TRANSITION ELECTRIC DIPOLE MOMENTS"), which were then read as cm-1: until 24/11/2026 the
    # excited state tables and spectra of ORCA reports had wavelengths of millions of nm.
    if getattr(data, "etenergies", None) is not None:
        try_update_value(data_model, "results.excited_states.et_energies", [float(e) for e in data.etenergies])
    else:
        try_update_value(data_model, "results.excited_states.et_energies", data_json, ['transitions', 'electronic transitions'])
    try_update_value(data_model, "results.excited_states.et_oscs", data_json, ['transitions', 'oscillator strength'])
    try_update_value(data_model, "results.excited_states.et_sym",  data_json, ['transitions', 'symmetry'])
    try_update_value(data_model, "results.excited_states.et_transitions", data_json, ['transitions', 'one excited config'])
    split_deexcitations(data_model.results.excited_states)
    
    # here NWChem
    try_update_value(data_model, "results.excited_states.et_rot", data_json, ['transitions', 'rotatory strength'])
    try_update_value(data_model, "results.excited_states.et_eldips", data_json, ['transitions', 'electic transition dipoles'])
    try_update_value(data_model, "results.excited_states.et_veldips", data_json, ['transitions', 'velocity-gauge electric transition dipoles'])
    try_update_value(data_model, "results.excited_states.et_magdips", data_json, ['transitions', 'magnetic transition dipoles'])


def molecule_section(data_model: LogData, data_json, data, obdata, verbose=False):
    """Fill ``molecule``: chirality, valences, InChI, SMILES, monoisotopic mass and bonds (Open
    Babel), then formula, numbers of atoms, charge, multiplicity, atomic numbers and the
    starting geometry and energy (cclib).

    Raises
    ------
    ScanlogException
        Open Babel could not read the molecule.
    """
    if "isChiral" in dir(obdata.OBMol) and "GetValence" in obdata.atoms[0].OBAtom:
        # openbabel 2
        data_model.molecule.chirality = obdata.OBMol.IsChiral()
        data_model.molecule.atoms_valence = [at.OBAtom.GetValence() for at in obdata.atoms]
    else:
        # openbabel 3
        data_model.molecule.chirality = obdata.OBMol.HasChiralityPerceived()
        # GetTotalValence() with implicit H, else GetExplicitValence()
        data_model.molecule.atoms_valence = [at.OBAtom.GetTotalValence() for at in obdata.atoms]
    # Start OpenBabel (all are mandatory)
    
    try:
        data_model.molecule.inchi = obdata.write("inchi").strip() # remove trailing \n
        data_model.molecule.smi = obdata.write("smi").split()[0]
        data_model.molecule.can = obdata.write("can").split()[0]
        data_model.molecule.monoisotopic_mass = obdata.OBMol.GetExactMass() # in Dalton
        
        connectivity_atom_pairs = []
        connectivity_bond_orders = []
        for i, a1 in enumerate(obdata.atoms):
            for j, a2 in enumerate(obdata.atoms):
                b = a1.OBAtom.GetBond(a2.OBAtom)
                if b is not None:
                    connectivity_atom_pairs.append((i, j))
                    connectivity_bond_orders.append(b.GetBondOrder())
        data_model.molecule.connectivity.atom_pairs = connectivity_atom_pairs
        data_model.molecule.connectivity.bond_orders = connectivity_bond_orders
    except:
        if verbose:
            traceback.print_exc()            
        raise ScanlogException("Reading mandatory data failed (Openbabel)")
    # End OpenBabel
    
    try_update_value(data_model, "molecule.formula", data_json, ['formula'])
    # CRITICAL TODO: formula versus inchi formula
    try_update_value(data_model, "molecule.nb_atoms", data_json, ['properties', 'number of atoms'])
    try_update_value(data_model, "molecule.nb_heavy_atoms", data_json, ['atoms', 'elements', 'heavy atom count'])
    try_update_value(data_model, "molecule.charge", data_json, ['properties', 'charge'])
    try_update_value(data_model, "molecule.multiplicity", data_json, ['properties', 'multiplicity'])
    try_update_value(data_model, "molecule.atoms_Z", data_json, ['atoms', 'elements', 'number'])
    
    # try_update_value(data_model, "molecule.atoms_masses", data.atommasses.tolist())
    # try_update_value(data_model, "molecule.nuclear_spins", data.nuclearspins.tolist())
    # try_update_value(data_model, "molecule.atoms_Zeff", data.atomzeff.tolist())
    # try_update_value(data_model, "molecule.nuclear_QMom", data.nuclearqmom.tolist())
    # try_update_value(data_model, "molecule.nuclear_gfactors", data.nucleargfactors.tolist())
    try_update_value(data_model, "molecule.starting_geometry", data.atomcoords[0,:,:].tolist())
    ## TODO: pb with SP
    try_update_value(data_model, "molecule.starting_energy", data_json, ["optimization", "scf", "scf energies"]) # in eV
    try:
        # eV to Hartree conversion
        try_update_value(data_model, "molecule.starting_energy", data_json["optimization"]["scf"]["scf energies"][0] / CstHartree2eV)
    except ScanlogException:
        pass #  SP ?
    try_update_value(data_model, "molecule.starting_nuclear_repulsion", nuclear_repulsion_energy(data, 0))


def parameters_section(data_model: LogData, data_json, data, obdata):
    # subsection : General parameters
    """Fill the ``comp_details`` section (computational details)."""
    general_param_subsection(data_model, data_json, data, obdata)
    # subsection : Geometry 
    geometry_param_subsection(data_model, data_json, data, obdata)
    # subsection : Thermochemistry and normal modes
    freq_param_subsection(data_model, data_json, data, obdata)
    # subsection :  Excited states
    td_param_subsection(data_model, data_json, data, obdata)


def results_section(data_model: LogData, data_json, data, obdata):
    # subsection : Wavefunction
    """Fill the ``results`` section."""
    wavefunction_results_subsection(data_model, data_json, data, obdata)
    # subsection : Geometry
    geom_results_subsection(data_model, data_json, data, obdata)
    # subsection : Thermochemistry and normal modes
    freq_results_subsection(data_model, data_json, data, obdata)
    # subsection : Excited states
    td_results_subsection(data_model, data_json, data, obdata)


def metadata_section(logfile, data_model: LogData, data_json, data, obdata):
    """Fill the ``metadata`` section: parser version, name of the step file, and whether the
    log has MO coefficients (``discretizable``).
    """
    data_model.metadata.parser_version = scanlog_version
    data_model.metadata.log_file = os.path.basename(logfile)
    data_model.metadata.discretizable = False
    if data_model.results.wavefunction.MO_coefs != None:
        data_model.metadata.discretizable = True


def full_report(logfile, data_json, data, obdata, verbose=False):
    """Build the LogData model of one step from its cclib and Open Babel data.

    Parameters
    ----------
    logfile : str
        Step file.
    data_json : dict
        cclib result as JSON (cclib writejson).
    data : cclib ccData
    obdata : pybel.Molecule
    verbose : bool, optional

    Returns
    -------
    LogData
    """
    data_model = LogData()
    
    # section : Molecule
    molecule_section(data_model, data_json, data, obdata, verbose=verbose)
    # section : Computational details
    parameters_section(data_model, data_json, data, obdata)
    # section : Results
    results_section(data_model, data_json, data, obdata)
    # section : Metadata
    metadata_section(logfile, data_model, data_json, data, obdata)
    
    return data_model


def last_geometry_xyz(data):
    """XYZ block of the last geometry read by cclib (atomnos, atomcoords in Angstrom), in the
    format of cclib's XYZ writer, for Open Babel.

    Temporary workaround (27/11/2026, Thomas Cauchy), to be removed once cclib is fixed: then
    call data.writexyz() again, as before. Why: the XYZ writer of cclib takes the comment of a
    geometry as metadata["comments"][index], but the Gaussian parser stores one comment per job
    (its title section), not one per geometry. It worked by chance for the last geometry while
    its index was -1; since cclib commit 58814731 (25/04/2026, "XYZ: remove -1 as sentinel") the
    index is n_geometries - 1 and every Gaussian optimization raises IndexError: no report could
    be made with cclib master. The cclib of /applis (master of 15/11/2025, 53594d0e) is not
    affected. Fix proposed to cclib: branch xyz-writer-comments (commit c374f344), see README.
    The block is built here from the two attributes every cclib version gives, in the format of
    the cclib writer (same atom lines; the title line is not used by Open Babel)."""
    table = cclib.parser.utils.PeriodicTable()
    coords = data.atomcoords[-1]
    lines = [str(len(data.atomnos)), "Last geometry"]
    lines += ["{:3s} {:15.10f} {:15.10f} {:15.10f}".format(table.element[int(z)], *xyz)
              for z, xyz in zip(data.atomnos, coords)]
    return "\n".join(lines) + "\n"


def logfile_to_obj(logfile, verbose=False):
    # reading with cclib
    """Read one step file into a LogData model.

    cclib reads the log; Open Babel reads the last geometry (from cclib's XYZ). The sum of
    electronic and thermal energies and, for ORCA, the functional are read in the log.

    Parameters
    ----------
    logfile : str
        Step file (split_logfile).
    verbose : bool, optional

    Returns
    -------
    LogData
    """
    data = cclib.parser.ccopen(logfile).parse()
    data.quchemreport_logfile = os.path.basename(logfile)  # for messages only
    data_json = json.loads(data.writejson())
    # openbabel sur XYZ
    obdata = pybel.readstring("xyz", last_geometry_xyz(data))
    # construct raw data object 
    data_model = full_report(logfile, data_json, data, obdata, verbose=verbose)
    thermal = _electronic_thermal_energy(logfile)
    if thermal is not None:
        data_model.results.freq.electronic_thermal_energy = thermal
    # cclib reads no functional from ORCA logs: without it the 'theory' test could not
    # see two different functionals
    general = data_model.comp_details.general
    if general.functional is None and general.package == "orca":
        general.functional = _orca_functional(logfile, data.metadata.get("keywords"))
    # Spin-orbit coupling: not read by cclib (see parser_engine.orca_soc)
    if general.package == "orca":
        data_model.results.excited_states.soc = read_orca_soc(logfile)
    _unrestricted_states(logfile, data_model)
    return data_model


# Name of the multiplicity in the state symmetries (as cclib writes 'Singlet-A', 'Triplet-B2')
MULTIPLICITY_NAMES = {1: "Singlet", 2: "Doublet", 3: "Triplet", 4: "Quartet", 5: "Quintet"}
_S2_GAUSSIAN = re.compile(r"^ Excited State\s+\d+:.*<S\*\*2>=\s*(-?\d+\.\d+)")
_S2_ORCA = re.compile(r"^STATE\s+\d+:\s+E=.*<S\*\*2>\s*=\s*(-?\d+\.\d+)")


def _unrestricted_states(logfile, data_model):
    """Multiplicity names and <S^2> of the excited states of an open-shell reference.

    cclib gives no multiplicity for these states ('3.002-?Sym' for Gaussian, the multiplicity
    estimated from <S^2>; '???' for ORCA), so they had no S/T label and their spin
    contamination did not appear in the report (O2, 26/11/2026). A TD-DFT from an open-shell
    reference only gives excitations that keep the spin projection: the states are named after
    the multiplicity of the reference (triplet O2: T1, T2...), and <S^2>, read in the log
    (the last values: an optimization prints one TD per step), shows how far each one is from
    a pure spin state."""
    es = data_model.results.excited_states
    mult = data_model.molecule.multiplicity
    if not es.et_sym or not mult or mult == 1 or any("Singlet" in str(x) or "Triplet" in str(x) for x in es.et_sym):
        return
    name = MULTIPLICITY_NAMES.get(mult)
    if name is None:
        return
    spatial = [str(x).split("-", 1)[1] if "-" in str(x) else "" for x in es.et_sym]
    es.et_sym = [f"{name}-{sym}" if sym and sym not in ("?Sym", "???") else name for sym in spatial]
    pattern = _S2_ORCA if data_model.comp_details.general.package == "orca" else _S2_GAUSSIAN
    try:
        with open(logfile, errors="replace") as fd:
            values = [float(m.group(1)) for m in map(pattern.match, fd) if m]
    except OSError:
        return
    n = len(es.et_sym)
    if len(values) >= n:
        es.et_s2 = values[-n:]


def quality_check_lvl2(data_model: LogData, solver, verbose=False):
    """Mark the model as archivable when it has a total energy (``metadata.archivable``; kept
    from the database version of the parser, not used by the report).
    """
    qual = "True"
    qual2 = "True"
    
    # if not "basis_set_md5" in res_json["comp_details"]["general"].keys():
    #     qual = "False"
    if data_model.results.wavefunction.total_molecular_energy == None:
        qual = "False"
    # if OPT then res_json["results"]["wavefunction"]["MO_coefs"] needed
    # if 'OPT' in res_json["comp_details"]["general"]["job_type"]:
    #     if "MO_coefs" in res_json["results"]["wavefunction"].keys():
    #         qual2 = "True"
            # # If only OPT then qual = False (not archivable) ???
            # if len(res_json["comp_details"]["general"]["job_type"]) == 1:
            #     qual = "False"
    # TODO: mandatory job_type, package & package_version for import policy
    
    data_model.metadata.archivable = qual
    data_model.metadata.archivable_for_new_entry = qual2
    
    if verbose:
        print(">>> START QC lvl2 <<<")
        print("File:", data_model.metadata.log_file)
        print("Job type:", data_model.comp_details.general.job_type)
        print("Archivable:", data_model.metadata.archivable)
        print("Archivable for new entry:", data_model.metadata.archivable_for_new_entry)
        print(">>> END QC lvl2 <<<\n")


def process_logfile(config: Config, logfile, logfile_idx, log_storage_path="", verbose=False):
    """Split and read one log file of the YAML file.

    Every step of the file gets the type given in the YAML file (a note is printed when
    the file has several steps) and the index of its YAML entry (``metadata.config_idx``).

    Parameters
    ----------
    config : Config
    logfile : str
    logfile_idx : int
        Index of the log file in ``config.logfiles``.
    log_storage_path : str, optional
        Folder of the step files.
    verbose : bool, optional

    Returns
    -------
    log_files : list of str
        Step files.
    data_list : list of LogData
        One model per step.

    Raises
    ------
    ScanlogException
        No completed calculation in the file (message naming the missing termination
        line), or a step that cclib or Open Babel cannot read.
    """
    solver = config.common_solver
    
    log_files = split_logfile(
        logfile, solver, log_storage_path=log_storage_path, verbose=verbose
    )
    if not log_files:
        termination_markers = {
            "gaussian": "Normal termination",
            "orca": "Timings for individual modules:",
            "gamess": "TERMINATED NORMALLY",
        }
        marker = termination_markers.get(solver)
        if marker:
            raise ScanlogException(
                f"No completed {solver} calculation found in '{logfile}': "
                f"the termination marker '{marker}' is missing. "
                "The calculation may have been interrupted or the output file may be incomplete; "
                "check the end of the log and the calculation status."
            )
        raise ScanlogException(f"No completed calculation found in '{logfile}'.")
    data_list = []
    
    if len(log_files) > 1:
        print(f"Note: '{logfile}' contains {len(log_files)} steps; each one is read as a separate log "
              f"with the type {config.logfiles[logfile_idx].type} given in the YAML file.")
    for step, log in enumerate(log_files):
        try:
            data_model: LogData = logfile_to_obj(log, verbose=verbose)
        except ScanlogException:
            raise
        except Exception as err:
            if verbose:
                traceback.print_exc()
            raise ScanlogException(
                f"Reading of '{logfile}' (step {step + 1}/{len(log_files)}) failed in cclib or Open Babel: "
                f"{type(err).__name__}: {err}. Check this log file (or remove it from the YAML file)."
            ) from err
        data_model.metadata.config_idx = logfile_idx
        # Name shown in the report and messages: the file given in the YAML file, not the
        # temporary step file (OPT_step_0.log), with the step for multi-step logs
        data_model.metadata.log_file = os.path.basename(logfile) + (
            f" (step {step + 1})" if len(log_files) > 1 else "")
        data_model.comp_details.general.job_type = config.logfiles[logfile_idx].type   
        # job_type_guess(res_json) # jobtype logique replaced and metadata.discretizable moved over to metadata section
        quality_check_lvl2(data_model, solver, verbose=verbose)
        data_list.append(data_model)
        
    return (log_files, data_list)


def process_logfile_list(config: Config, log_storage_path="", verbose=False):    
    """Split and read all the log files of the YAML file, in their order.

    Parameters
    ----------
    config : Config
    log_storage_path : str, optional
    verbose : bool, optional

    Returns
    -------
    log_files : list of str
    data_list : list of LogData
        The parsed models, several per multi-step log file.
    """
    logfilelist = [str(l.path.resolve()) for l in config.logfiles]
    
    data_list = []
    log_files = []
    
    for logfile_idx, logfile in enumerate(logfilelist):
        l, d = process_logfile(
            config, logfile, logfile_idx, log_storage_path=log_storage_path, verbose=verbose
        )
        log_files += l
        data_list += d

    # Two YAML entries with the same file name (OPT.log of two folders): add the folder
    names = [os.path.basename(f) for f in logfilelist]
    for model in data_list:
        logfile = logfilelist[model.metadata.config_idx]
        if names.count(os.path.basename(logfile)) > 1:
            model.metadata.log_file = os.path.join(os.path.basename(os.path.dirname(logfile)),
                                                   model.metadata.log_file)

    return (log_files, data_list)
