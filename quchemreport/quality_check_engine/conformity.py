#-*- coding: utf-8 -*-

from dataclasses import dataclass
import sys
import numpy as np
from typing import Any, List, Optional, Tuple

from quchemreport.config.config import Config
from quchemreport.utility_services.log_data import LogData
from quchemreport.utility_services.validated_data import ValidatedData

@dataclass
class ExtractedData:
    nb_files: int
    packages: List[str]
    formulas: List[str]
    formulas_nocharges: List[str]
    functionals: List[str]
    basis_sets: List[str]
    ao_names: List[List[str]]
    MO_coeff: List[List[List[float]]]
    job_types: List[List[str]]
    nres: List[float]
    nres_noES: List[float]
    charges: List[int]
    charges_noSP: List[int]
    charges_SP: List[int]
    multiplicities: List[int]
    multiplicities_noSP: List[int]
    list_data_models: List[LogData]
    verbose: bool

@dataclass
class ReferenceData:
    charge_ref: int
    multiplicity_ref: int
    MO_coeff_ref: List
    basis_sets_ref: List
    ao_names_ref: List
    package_ref: str
    data_for_discretization: object

def extract_data_models(list_data_models: List[LogData], verbose=False):
    packages = []
    formulas = []
    functionals = []
    ao_names = []
    basis_sets = []
    MO_coeff = []
    job_types = []
    nres = []
    charges = []
    multiplicities = []

    nres_noES = []
    charges_noSP = []
    charges_SP = []
    multiplicities_noSP = []
    
    # Create list of key values for conformity tests
    for data_model in list_data_models: 
        data_model: LogData = data_model
        packages.append(data_model.comp_details.general.package.lower())
        formulas.append(data_model.molecule.formula)
        nres.append(data_model.results.geometry.nuclear_repulsion_energy_from_xyz)
        
        # TODO : Check if logic is correct (Jobtypes with only ES or jobtypes among wich ES is present)
        if (
            'opt_es' in data_model.comp_details.general.job_type
            or 'freq_es' in data_model.comp_details.general.job_type
        ):
            print ('Detected optimized excited state in :', data_model.metadata.log_file)
        else :
            nres_noES.append(data_model.results.geometry.nuclear_repulsion_energy_from_xyz)
            
        if data_model.comp_details.general.functional != None:
            functionals.append(data_model.comp_details.general.functional)
        else:
            functionals.append("ERROR reading functionals")
            
        # basis sets, ao_names and MO coeffs are not mandatory but necessary for the discretization process
        if data_model.comp_details.general.basis_set != None:
            basis_sets.append(data_model.comp_details.general.basis_set)
        if data_model.comp_details.general.ao_names != None:
            ao_names.append(data_model.comp_details.general.ao_names)
        
        if data_model.results.wavefunction.MO_coefs != None:
            MO_coeff.append(data_model.results.wavefunction.MO_coefs) 
        else: 
            MO_coeff.append([])   # Problem we would like to add N/A is no MO to keep the same index hax jf[i] but test on len after   
            
        jt = data_model.comp_details.general.job_type
        charges.append(data_model.molecule.charge)
        if ('sp' not in jt):
            charges_noSP.append(data_model.molecule.charge)
        if ('sp' in jt):
            charges_SP.append(data_model.molecule.charge)
        job_types.append(jt) 
        
        multiplicities.append(data_model.molecule.multiplicity)
        if ('sp' not in jt):
            multiplicities_noSP.append(data_model.molecule.multiplicity)
            
    # Problem in the formula: the charges appear and direct comparison fails
    # example : [u'C4H7NO', u'C4H7NO', u'C4H7NO+', u'C4H7NO', u'C4H7NO-']
    # Clean the charges in the formula to compare them later
    formulas_nocharges = []
    # If no charge, get formula, 
    # If monocationic remove last caracter (+) because 1 is implicit
    # If charge > 1, remove length of charge and the plus sign
    # If charge is negative, length of charge include the minus sign

    for i, charge in enumerate(charges) :
        if (int(charge) == 0) :
            formulas_nocharges.append(formulas[i])
        elif (int(charge) == 1) or (int(charge) < -1) :
            formulas_nocharges.append(formulas[i][:-len(str(charges[i]))])
        elif (int(charge) > 1) :
            formulas_nocharges.append(formulas[i][:-(len(str(charges[i]))+1)])
        elif (int(charge) == -1) :
            formulas_nocharges.append(formulas[i][:-(len(str(charges[i]))-1)])
            
    return ExtractedData(
        nb_files=len(list_data_models),
        packages=packages,
        formulas=formulas,
        formulas_nocharges=formulas_nocharges,
        functionals=functionals,
        basis_sets=basis_sets,
        ao_names=ao_names,
        MO_coeff=MO_coeff,
        job_types=job_types,
        nres=nres,
        nres_noES=nres_noES,
        charges=charges,
        charges_noSP=charges_noSP,
        charges_SP=charges_SP,
        multiplicities=multiplicities,
        multiplicities_noSP=multiplicities_noSP,
        list_data_models=list_data_models,
        verbose=verbose,
    )

def test_package(data: ExtractedData, config: Config):
    packages = data.packages

    if (
        not packages
        or any(pk != packages[0] for pk in packages[1:])
    ):
        print("ERROR: You need to provide a set of logfiles for a unique conformer : Too many different packages")
        print(packages)
        print("Program will exit.")
        sys.exit(1)

    if packages[0] != config.common_solver:
        print("WARNING: You need to provide a set of logfiles for a unique conformer : Packages are different from the specified common solver ")
        print(f"packages: {packages} ; common_solver: {config.common_solver}")
        

def test_formula(data: ExtractedData):
    # Checking if data_models has more than 1 file in order to process parenting and uniformity test
    if (data.nb_files) < 1 :
        return
    
    # Test on cleaned formulas. There should be only one value
    if (len(set(data.formulas_nocharges))) > 1:
        print("ERROR: You need to provide a set of logfiles for a unique conformer : Too many different formulas")
        print(data.formulas_nocharges)
        print("Program will exit.")
        sys.exit(1)
    elif (len(set(data.formulas_nocharges))) == 1: 
        print("\tDetected formula:                                ... ", set(data.formulas_nocharges))
        print("\tSame formula for all logfiles:                   ...  Test OK")
    elif (len(set(data.formulas_nocharges))) == 0:
        print("ERROR: There is no correct formulas in the list. Program will exit. Contact admin")
        sys.exit(1)


def test_theory(data: ExtractedData):
    # Tests on theory (same functional and same basis set)
    # print("Detected number of basis sets : ", len(set(data.basis_sets)))
    # print("Detected number of functionals : ", len(set(data.functionals)))

    if (len(set(data.functionals)) > 1) or (len(set(data.basis_sets)) > 1): 
        print("\tDetected number of basis sets:                   ... ", len(set(data.basis_sets)))
        print("\tDetected functionals:                            ... ", data.functionals)
        print("ERROR: You need to provide a set of logfiles with the same level of theory")
        print("That means a unique functional and basis set.")
        print("Program will exit.")
        sys.exit(1)
    else:
        print("\tDetected number of basis sets:                   ... ", len(set(data.basis_sets)))
        print("\tDetected functionals:                            ... ", data.functionals)
        print("\tSame functional and basis set for all logfiles:  ...  Test OK")
        
    


# Relative tolerance on the nuclear repulsion energy (NRE) of a same geometry. The NRE is
# computed from coordinates rounded to 1e-5 A: a rotated copy of the same geometry
# (other orientation) changes it by up to ~1.3e-6 (relative), a 0.001 A change of
# one bond length by >= 1.6e-5 (azulene, C10H4Br2O2S2, phenanthrene).
NRE_RTOL = 5e-6


def same_nre(a, b, rtol=NRE_RTOL):
    return a is not None and b is not None and abs(a - b) <= rtol * max(abs(a), abs(b))


def test_nuclear_repulsion(data: ExtractedData):
    # Test on NRE to ensure that there is only one conformer
    nres = [nre for nre in data.nres_noES if nre is not None]
    if nres and not same_nre(min(nres), max(nres)):
        print("\tSame geometries (no optimized excited state):    ... ", len(set(data.nres_noES)))
        print("\tDetected NRE (with optimized excited state):     ... ",data.nres)
        print("ERROR: You need to provide a set of logfiles for a unique conformer without counting the optimized excited states.")
        print("Program will exit.")
        sys.exit(1)
    else:
        print("\tSame conformer for all logfiles:                 ...  Test OK")
    


SINGLE_POINTS_ONLY = (
    "ERROR: all the log files are single points (type sp). The report needs the optimization "
    "(type opt, or [opt, freq]) of the ground state as reference; single points of the cation "
    "and anion (Fukui functions, CDFT indices) must be computed at this optimized geometry. "
    "Program will exit.")


def test_charge(data: ExtractedData):
    if (len(set(data.charges_noSP))) > 1:
        print("\tDetected charges (No single points):             ... ", len(set(data.charges_noSP)))
        print("\tDetected charges (with single points):           ... ",data.charges)
        print("ERROR: You need to provide a set of logfiles with the same charge without counting the single points.")
        print("Program will exit.")
        sys.exit(1)
    elif  (len (set(data.charges_noSP))) == 0 :
        print(SINGLE_POINTS_ONLY)
        sys.exit(1)
    else:
        print("\tSame charge for all logfiles:                    ...  Test OK")
    


def test_multiplicity(data: ExtractedData):
    # Test on multiplicity to ensure that there is only one oxydation state whitout counting the single points
    if (len(set(data.multiplicities_noSP))) > 1:
        print("\tDetected multiplicities (No single points):      ... ", len(set(data.multiplicities_noSP)))
        print("\tDetected multiplicities (with single points):    ... ",data.multiplicities)

        print("ERROR: You need to provide a set of logfiles with the same electronic state.")
        print("Program will exit.")
        sys.exit(1)
    elif  (len(set(data.multiplicities_noSP))) == 0 :
        print(SINGLE_POINTS_ONLY)
        sys.exit(1)
    else:
        print("\tSame multiplicity:                               ...  Test OK")
    


def test_ground_state_presence(data: ExtractedData):
    # Test on Ground State information in case of multiples log files
    job_types = data.job_types
    has_ground_state = any(
        "opt" in jt or "freq" in jt for jt in job_types
    )

    if not has_ground_state:
        print("ERROR: In a list of logfiles, one OPT or FREQ of a Ground State should be provided. Program will exit.")
        sys.exit(1)
    else:
        print("\tGround State:                                    ...  Test OK")


def run_selected_tests(data: ExtractedData, config: Config, verbose=False):
    # Checking logfiles packages
    test_package(data, config)
    
    tests_map = {
        "formula": test_formula,
        "theory": test_theory,
        "nuclear_repulsion": test_nuclear_repulsion,
        "charge": test_charge,
        "multiplicity": test_multiplicity,
        "ground_state_optimization": test_ground_state_presence,
    }

    checks = config.quality_control.checks

    for attr, test_func in tests_map.items():
        if getattr(checks, attr, False):
            test_func(data)
        else:
            if verbose:
                print(f"Skipping test: {attr} (disabled)")


def compare_refs(data: ExtractedData, ref_idx):
    # TODO : TO COMPLETE
    return True


def select_reference_log(data: ExtractedData, config: Config):
    ref_idx = []
    for idx, logfile in enumerate(config.logfiles):
        if logfile.reference:
            ref_idx.append(idx)

    # Check if there is at least one reference log file
    if (len(ref_idx) == 0):
        print("\nNo reference log file found. You need to specify a reference log file in the config file.")
        print("Program will exit.")
        sys.exit(1)
    
    # Check if multiple reference log files are specified
    if (len(ref_idx) > 1):
        print("\nMultiple reference log files found. You need to specify a single reference log file in the config file.")
        print("Program will exit.")
        sys.exit(1)
    
    print("\tReference:                                       ...  Test OK")
    return ref_idx[0]


def reference_model_index(data: ExtractedData, config_reference_idx):
    """Index of the reference among the parsed models (a YAML log file may give several
    steps/models): first step of the reference log file that has MO coefficients, else its first step."""
    steps = [i for i, model in enumerate(data.list_data_models)
             if model.metadata.config_idx == config_reference_idx]
    if not steps:
        print("ERROR: the reference log file gave no usable data. Program will exit.")
        sys.exit(1)
    with_mo = [i for i in steps if len(data.MO_coeff[i]) in (1, 2)]
    return with_mo[0] if with_mo else steps[0]


def same_geometry(model_a, model_b, atol=1e-4):
    """Same atoms at the same cartesian coordinates (A), orientation included: the grids
    of one log are drawn with the atoms of the other one."""
    a = model_a.results.geometry.elements_3D_coords
    b = model_b.results.geometry.elements_3D_coords
    if a is None or b is None or model_a.molecule.atoms_Z != model_b.molecule.atoms_Z:
        return False
    a, b = np.array(a, dtype=float).reshape(-1, 3), np.array(b, dtype=float).reshape(-1, 3)
    return a.shape == b.shape and np.allclose(a, b, rtol=0, atol=atol)


def select_reference_data(data: ExtractedData, reference_id, verbose: bool = False):
    """reference_id is the index of the reference model (see reference_model_index).

    The grids (MO, MEP, Fukui, EDD...) use the reference itself when it has MO
    coefficients and a basis set; otherwise the first other log of the same
    charge, multiplicity, geometry (NRE) and solvent that has them."""
    models = data.list_data_models
    ref_model = models[reference_id]
    charge_ref = data.charges[reference_id]
    multiplicity_ref = data.multiplicities[reference_id]
    nres_ref = data.nres[reference_id]
    solvent_ref = ref_model.comp_details.general.solvent
    package_ref = data.packages[reference_id]

    def usable(i):
        general = models[i].comp_details.general
        return (len(data.MO_coeff[i]) in (1, 2)
                and general.basis_set is not None and general.ao_names is not None
                and data.charges[i] == charge_ref
                and data.multiplicities[i] == multiplicity_ref
                and same_geometry(models[i], ref_model)
                and general.solvent == solvent_ref)

    data_for_discretization = None
    MO_coeff_ref = []
    basis_sets_ref = []
    ao_names_ref = []
    for i in [reference_id] + [j for j in range(len(models)) if j != reference_id]:
        if usable(i):
            data_for_discretization = models[i]
            MO_coeff_ref = data.MO_coeff[i]
            basis_sets_ref = models[i].comp_details.general.basis_set
            ao_names_ref = models[i].comp_details.general.ao_names
            if i != reference_id:
                print(f"Note: the reference log has no MO coefficients or basis set; the grids use "
                      f"{models[i].metadata.log_file} (same charge, multiplicity, geometry, orientation and solvent).")
            elif verbose:
                print("A reference data has been selected for the discretization process:", models[i].metadata.log_file)
            break
    if data_for_discretization is None:
        print("Warning: no log file with MO coefficients and basis set matches the reference "
              "(same charge, multiplicity, solvent and cartesian coordinates, orientation included; "
              "add pop=full and gfprint to the reference calculation): no MO, MEP, Fukui or EDD pictures.")

    return ReferenceData(
        package_ref=package_ref,
        charge_ref=charge_ref,
        multiplicity_ref=multiplicity_ref,
        MO_coeff_ref=MO_coeff_ref,
        basis_sets_ref=basis_sets_ref,
        ao_names_ref=ao_names_ref,
        data_for_discretization=data_for_discretization,
    )


def test_basis_and_mo_coeff(reference_data: ReferenceData, verbose: bool = False):
    # Discretization depends on Orbkit and needs basis sets and MO coefficients.
    # Orbkit can not work if the file have both sphericals and cartesians basis set functions.
    discret_proc = False
    cartesian = False
    spherical = False
    # Define booelan to not repeat MO discretization process
    mo_viz_done = False

    # Test on Basis set and MO coefficients to ensure that the discretization process is doable
    # Than tests if basis set is cartesian or spherical or mixed
    if (len(set(reference_data.basis_sets_ref))) == 0 or (len(reference_data.MO_coeff_ref)) == 0 :
        print("There is no basis set or MO coefficients in your logfiles. MO and EDD pictures cannot be generated.")
        discret_proc = False
    elif (len(set(reference_data.basis_sets_ref))) != 0 and (len(reference_data.MO_coeff_ref)) != 0:
        if verbose:
            print("MO coefficients and basis sets detected. MO and EDD pictures can be generated." )
        # Basis set names for D and F orbitals
        ao = []
        for i, orbitals in enumerate(reference_data.ao_names_ref) :
            ao.append(reference_data.ao_names_ref[i].split('_')[-1])
        ao_DF = []
        for i, orbitals in enumerate(ao) :
            # Isolate D and F atomic orbitals names
            # If cartesian, D may not appear in the name. We keep  the X coordinates for test
            if ("D" in ao[i]) or ("XX" in ao[i]) or ("F" in ao[i]) or ("XY" in ao[i]):
                ao_DF.append(ao[i])    
        for i, orbitals in enumerate(ao_DF) :        
            if ("XX" in ao_DF[i]):
                cartesian = True
            elif ("+" in ao_DF[i]) or reference_data.package_ref == 'orca':
                spherical = True

        # Test if there is no D or F functions
        if (len(ao_DF)) == 0 :
            for i, orbitals in enumerate(ao) :
            # Enumerate again to test if spherical or cartesian based on the p orbitals
                if ("PX" in ao[i]) :
                    cartesian = True
            # See which test could isolate a spherical p orbital
            #    else :
            #        spherical = True
            # Only s functions (e.g. He, H2 with 6-31G or STO-3G): cartesian and spherical
            # functions are then the same
            if not cartesian and all(name.rstrip().endswith("S") for name in ao):
                cartesian = True
                if verbose:
                    print("Basis set with s functions only")
        if (cartesian is True) and (spherical is True) :
            # Mixed basis sets (Gaussian 6-31G(2df): Cartesian d, spherical f; 5D 10F) are
            # handled shell by shell by the qmckl backend, not by orbkit.
            from quchemreport.data_enrichment.calc_orb import _use_qmckl
            discret_proc = _use_qmckl()
            if discret_proc and verbose:
                print("Mixed Cartesian/spherical basis set detected")
        elif (cartesian is True) and (spherical is False) :
            discret_proc = True
            if verbose:
                print("Cartesian basis set detected")
        elif (cartesian is False) and (spherical is True) :
            discret_proc = True
            if verbose:
                print("Spherical basis set detected")
        else :
            print("ERROR: The basis set is neither cartesian nor spherical nor mixed. Contact admin.")
            sys.exit(1)
        if discret_proc is True :
            print("All discretization tests OK.")
        else :
            print("The basis set is a mixture of cartesian and spherical atomic orbitals: the orbkit backend cannot discretize it (use the default qmckl backend).")
            
    return discret_proc, mo_viz_done


### CONFORMITY/UNIFORMITY TESTS
# Conformity and Uniformity between log files depend on chosen JSON keys
# All log files must correpond to the same formula and theory (functional and basis set)
# Same formulas, same theory and same NRE (nuclear repulsion energy) == same conformer
# If same conformer but different charges = Fukui process
# If same conformer but different NRE and job type = OPT_ES or FREQ_ES, related excited state
def tests(list_data_models: List[LogData], config: Config):
    verbose = config.logging.level == "debug"
    
    extracted: ExtractedData = extract_data_models(list_data_models, verbose=verbose)

    run_selected_tests(extracted, config, verbose=verbose)
    
    # Test references (YAML index, then index among the parsed models)
    reference_id = reference_model_index(extracted, select_reference_log(extracted, config))
    
    # End of uniformity / conformity tests. 
    # At this point : we have a set of log files of a unique conformer
    # Or related excited state or related Single point
    print("All conformity tests OK.")

    # PARENTING AND INFORMATION TESTS
    ### DISCRETIZATION TESTS
    print("\nStarting discretization tests.")
    reference_data: ReferenceData = select_reference_data(extracted, reference_id, verbose=verbose) # TODO : pass reference_id
    discret_proc, mo_viz_done = test_basis_and_mo_coeff(reference_data, verbose=verbose)
    
    validatedData = ValidatedData(
        job_types=extracted.job_types,
        nres_noES=extracted.nres_noES,
        charges=extracted.charges,
        charge_ref=reference_data.charge_ref,
        discret_proc=discret_proc,
        mo_viz_done=mo_viz_done,
        data_for_discretization=reference_data.data_for_discretization,
        ref_log_file_idx=reference_id
    )
    return validatedData



