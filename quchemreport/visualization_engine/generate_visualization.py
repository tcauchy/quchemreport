"""Pictures of the report, from the ComputedData object.

generate_visualization draws, for the outputs asked in the YAML file (ComputeFlags), the 3D
pictures (visu_mayavi), the spectra (visu_plots) and their text files (visu_txt), in the output
folder. File names follow the convention of docs/architecture.md: ``img-log<k>-<content>.png``
with k the model index plus one; pictures of the reference only (Fukui functions) have no
number. The report reads the same names (report_generator.generate_report).
"""

from collections import defaultdict
import json
import sys
from quchemreport.config.config import Config
from quchemreport.data_enrichment.compute_flags import ComputeFlags
from quchemreport.utility_services.computed_data import ComputedData, config_logfile
from quchemreport.visualization_engine import visu_mayavi, visu_plots, visu_txt
from quchemreport.data_enrichment import calc_orb
from quchemreport.utility_services.parameters import obk_step
from quchemreport.utility_services.naming import state_label, state_number
from quchemreport.data_enrichment.mep_surface import MEP_SURFACE_ISOVALUE


def generate_molecule_representation(
    computed_data: ComputedData, 
    nuclear_repulsion_energy_from_xyz, elements_3D_coords, atom_pairs, atoms_Z,
    log_idx, sources="N/A"
):
    
    """Topology picture of one model (``img-log<k>-TOPOLOGY.png``).

    Parameters
    ----------
    computed_data : ComputedData
    nuclear_repulsion_energy_from_xyz : float
        Only checked for presence (a model without geometry gives no picture).
    elements_3D_coords, atom_pairs, atoms_Z :
        Geometry, bonds and atomic numbers of the model.
    log_idx : int
        Model index.
    sources : str, optional
        Log files of the group, for the message.
    """
    output_dir = computed_data.metadata.output_dir
    
    if (
        nuclear_repulsion_energy_from_xyz is not None
        and elements_3D_coords is not None
        and atom_pairs is not None
        and atoms_Z is not None
    ):
        visu_mayavi.viz_topology(
            elements_3D_coords,
            atom_pairs,
            atoms_Z,
            file_name=output_dir+f"img-log{log_idx+1}"
        )
        print(f"Picture generated for one topology (Source: {sources})")

def output_molecule_representations(computed_data: ComputedData):
    """One topology picture per group of models with the same topology."""
    log_files = computed_data.metadata.log_files

    all_nuclear_repulsion_energy_from_xyz = computed_data.results.geometry.nuclear_repulsion_energy_from_xyz
    all_elements_3D_coords = computed_data.results.geometry.elements_3D_coords
    all_atom_pairs = computed_data.molecule.connectivity.atom_pairs
    all_atoms_Z = computed_data.molecule.atoms_Z

    # Generate an image by groups of topology
    groups = computed_data.derived.molecule.topology_groups
    for group in groups:
        
        # Skip if no data
        if group is None:
            continue
        
        # Idx of the base log of the group
        base_log_idx = group[0]

        nuclear_repulsion_energy_from_xyz = all_nuclear_repulsion_energy_from_xyz[base_log_idx]
        elements_3D_coords = all_elements_3D_coords[base_log_idx]
        atom_pairs = all_atom_pairs[base_log_idx]
        atoms_Z = all_atoms_Z[base_log_idx]

        sources = ", ".join([log_files[i] for i in group])
        generate_molecule_representation(
            computed_data,
            nuclear_repulsion_energy_from_xyz,
            elements_3D_coords,
            atom_pairs,
            atoms_Z,
            base_log_idx,
            sources=sources
        )


def output_MO_diagrams_clean(computed_data: ComputedData):
    
    """Pictures of the requested orbitals of the reference (one set per spin if unrestricted)."""
    ref_id = computed_data.metadata.ref_log_file_idx
    output_dir = computed_data.metadata.output_dir
    
    MO_data = computed_data.derived.wavefunction.MO_analysis.MO_diagrams
    elements_3D_coords = computed_data.results.geometry.elements_3D_coords[ref_id]
    atom_pairs = computed_data.molecule.connectivity.atom_pairs[ref_id]
    atoms_Z = computed_data.molecule.atoms_Z[ref_id]
    
    if (
        MO_data.alpha is not None
        or MO_data.beta is not None
        or MO_data.restricted is not None
    ):
        print("\nGenerating MO Diagrams ...")
    
    # Generate visualization of the MO Diagrams for unrestricted calculation
    if MO_data.alpha is not None:
            visu_mayavi.viz_MO(
                MO_data.alpha.discretized_MO_voxels, 
                MO_data.alpha.grid_X, 
                MO_data.alpha.grid_Y, 
                MO_data.alpha.grid_Z,
                elements_3D_coords,
                atom_pairs,
                atoms_Z,
                file_name=output_dir+f"img-log{ref_id + 1}", 
                labels=MO_data.alpha.labels
            )
            print("Generation of the image for Alpha MO Diagram done")
            
    if MO_data.beta is not None:
        visu_mayavi.viz_MO(
            MO_data.beta.discretized_MO_voxels, 
            MO_data.beta.grid_X, 
            MO_data.beta.grid_Y, 
            MO_data.beta.grid_Z,
            elements_3D_coords,
            atom_pairs,
            atoms_Z,
            file_name=output_dir+f"img-log{ref_id + 1}", 
            labels=MO_data.beta.labels
        )
        print("Generation of the image for Beta MO Diagram done")
    
    # Generate visualization of the MO Diagram for restricted calculation
    if MO_data.restricted is not None:
        visu_mayavi.viz_MO(
            MO_data.restricted.discretized_MO_voxels, 
            MO_data.restricted.grid_X, 
            MO_data.restricted.grid_Y, 
            MO_data.restricted.grid_Z, 
            elements_3D_coords,
            atom_pairs,
            atoms_Z,
            file_name=output_dir+f"img-log{ref_id + 1}", 
            labels=MO_data.restricted.labels
        )
        print("Generation of the image for restricted MO Diagram done")


def output_MEP_maps(computed_data: ComputedData):
    
    """MEP picture and colour bar of the reference, with the extrema markers and the colour range."""
    ref_id = computed_data.metadata.ref_log_file_idx
    output_dir = computed_data.metadata.output_dir
    
    elements_3D_coords = computed_data.results.geometry.elements_3D_coords[ref_id]
    atom_pairs = computed_data.molecule.connectivity.atom_pairs[ref_id]
    atoms_Z = computed_data.molecule.atoms_Z[ref_id]
    
    # Skip if data is missing
    if (
        elements_3D_coords is None
        or atom_pairs is None
        or atoms_Z is None
    ):
        return
    
    rho = computed_data.derived.wavefunction.MEP_maps.rho_voxels
    MEP = computed_data.derived.wavefunction.MEP_maps.MEP_voxels
    grid_X = computed_data.derived.wavefunction.MEP_maps.grid_X
    grid_Y = computed_data.derived.wavefunction.MEP_maps.grid_Y
    grid_Z = computed_data.derived.wavefunction.MEP_maps.grid_Z
    
    if (
        rho is not None
        and MEP is not None
        and grid_X is not None
        and grid_Y is not None
        and grid_Z is not None
    ):
        print("\nGenerating MEP Map ...")
        visu_mayavi.viz_Potential(
            rho,
            MEP,
            grid_X,
            grid_Y,
            grid_Z,
            elements_3D_coords,
            atom_pairs,
            atoms_Z,
            file_name=output_dir+f"img-log{ref_id + 1}",
            isovalue=computed_data.derived.wavefunction.MEP_maps.surface_isovalue or MEP_SURFACE_ISOVALUE,
            extrema=computed_data.derived.wavefunction.MEP_maps.surface_extrema,
            color_range=computed_data.derived.wavefunction.MEP_maps.color_range,
        )
        print("Generation of the MEP Map image done")


def output_fukui_functions(computed_data: ComputedData):
    
    """Pictures of f+ (SP plus), f- (SP minus) and of the dual descriptor, when computed."""
    ref_id = computed_data.metadata.ref_log_file_idx
    output_dir = computed_data.metadata.output_dir
    
    elements_3D_coords = computed_data.results.geometry.elements_3D_coords[ref_id]
    atom_pairs = computed_data.molecule.connectivity.atom_pairs[ref_id]
    atoms_Z = computed_data.molecule.atoms_Z[ref_id]
    
    # Skip if data is missing
    if (
        elements_3D_coords is None
        or atom_pairs is None
        or atoms_Z is None
    ):
        return
    
    print("")
    
    # Generate the output of the fukui functions for the SP Plus if the data is available
    f_plus  = computed_data.derived.reactivity.fukui_functions.SP_plus
    if f_plus is not None:
        
        delta_rho_SPp = f_plus.rho_voxels
        grid_X_SPp = f_plus.grid_X
        grid_Y_SPp = f_plus.grid_Y
        grid_Z_SPp = f_plus.grid_Z
        
        if (
            delta_rho_SPp is not None
            and grid_X_SPp is not None
            and grid_Y_SPp is not None
            and grid_Z_SPp is not None
        ):
            print("Generating the image for Fukui SP Plus ...")
            visu_mayavi.viz_Fukui(
                delta_rho_SPp,
                grid_X_SPp,
                grid_Y_SPp,
                grid_Z_SPp,
                elements_3D_coords,
                atom_pairs,
                atoms_Z,
                file_name=output_dir+"img",
                labels="SP_plus"
            )
            print("Generation of the image for Fukui SP Plus done")
    
    # Generate the output of the fukui functions for the SP Minus if the data is available
    f_minus = computed_data.derived.reactivity.fukui_functions.SP_minus
    if f_minus is not None:
        
        delta_rho_SPm = f_minus.rho_voxels
        grid_X_SPm = f_minus.grid_X
        grid_Y_SPm = f_minus.grid_Y
        grid_Z_SPm = f_minus.grid_Z
        
        if (
            delta_rho_SPm is not None
            and grid_X_SPm is not None
            and grid_Y_SPm is not None
            and grid_Z_SPm is not None
        ):
            print("Generating the image for Fukui SP Minus ...")
            visu_mayavi.viz_Fukui(
                delta_rho_SPm,
                grid_X_SPm,
                grid_Y_SPm,
                grid_Z_SPm,
                elements_3D_coords,
                atom_pairs,
                atoms_Z,
                file_name=output_dir+"img",
                labels="SP_minus"
            )
            print("Generation of the image for Fukui SP Minus done")
    
    # Generate the output of the fukui functions for the Dual if the data is available
    f_dual = computed_data.derived.reactivity.fukui_functions.dual
    if f_dual is not None:
        
        delta_rho_dual = f_dual.rho_voxels
        grid_X_dual = f_dual.grid_X
        grid_Y_dual = f_dual.grid_Y
        grid_Z_dual = f_dual.grid_Z
        
        if (
            delta_rho_dual is not None
            and grid_X_dual is not None
            and grid_Y_dual is not None
            and grid_Z_dual is not None
        ):
            print("Generating the image for Fukui Dual ...")
            visu_mayavi.viz_Fdual(
                delta_rho_dual,
                grid_X_dual,
                grid_Y_dual,
                grid_Z_dual,
                elements_3D_coords,
                atom_pairs,
                atoms_Z,
                file_name=output_dir+"img"
            )
            print("Generation of the image for Fukui Dual done")


def output_es_absorption_and_CD_spectrum(computed_data: ComputedData, do_cd, do_abso):
    
    """Absorption and CD spectra (pictures and text files) of every TD model.

    Parameters
    ----------
    computed_data : ComputedData
    do_cd, do_abso : bool
    """
    output_dir = computed_data.metadata.output_dir
    log_files = computed_data.metadata.log_files
    
    print("")
    
    for log_idx, log_file in enumerate(log_files):
    
        et_energies = computed_data.results.excited_states.et_energies[log_idx]
        et_rotats = computed_data.results.excited_states.et_rot[log_idx]
        et_oscs = computed_data.results.excited_states.et_oscs[log_idx]
        
        abso_spectrum = computed_data.derived.excited_states.abso_spectrum[log_idx]
        CD_spectrum = computed_data.derived.excited_states.CD_spectrum[log_idx]
        xvalues = computed_data.derived.excited_states.xvalues[log_idx]
        
        # Skip if no data
        if (
            abso_spectrum is None
            or len(abso_spectrum) == 0
            or xvalues is None
            or et_energies is None
        ):
            continue
    
        
        # Output calculated CD spectra in text files and figures
        if (
            do_cd 
            and CD_spectrum is not None 
            and len(CD_spectrum) != 0
            and et_rotats is not None
        ):
            print(f"Generating Circular dichroism spectrum done for {log_file} ...")
            # Write both Absorption and Circular dichroism spectra in text file to compare with experimental data
            visu_txt.textfile_CD(xvalues, abso_spectrum, CD_spectrum, file_name=output_dir+f"log{log_idx + 1}-",)                        
            # IF CD plot CD spectrum 
            visu_plots.absoCD(et_energies, et_rotats, xvalues, CD_spectrum, file_name=output_dir+"img"+f"-log{log_idx + 1}",)
            print(f"Circular dichroism spectrum done for {log_file}.")
        
        # Output calculated Absorption spectra in text files and figures
        if (
            do_abso
            and et_oscs is not None
        ):
            print(f"Generating Absorption spectrum done for {log_file}.")
            # If no CD write only Absorption spectrum in text file to compare with experimental data
            # CD_spectrum is None when all the rotatory strengths are 0 (achiral molecule)
            if (
                not do_cd
                or CD_spectrum is None
                or len(CD_spectrum) == 0
            ):
                visu_txt.textfile_UV(xvalues, abso_spectrum, file_name=output_dir+f"log{log_idx + 1}-")       

            # Plotting UV spectrum 
            visu_plots.absoUV(et_energies, et_oscs, xvalues, abso_spectrum, file_name=output_dir+"img"+f"-log{log_idx + 1}",)
            print(f"Absorption spectrum done for {log_file}.")


def output_es_EDD_OIF_DIP(computed_data: ComputedData):
    
    """Density difference, overlap and dipole pictures of the selected states of every TD model."""
    ref_id = computed_data.metadata.ref_log_file_idx
    output_dir = computed_data.metadata.output_dir
    log_files = computed_data.metadata.log_files
    
    elements_3D_coords = computed_data.results.geometry.elements_3D_coords[ref_id]
    atom_pairs = computed_data.molecule.connectivity.atom_pairs[ref_id]
    atoms_Z = computed_data.molecule.atoms_Z[ref_id]
    
    for log_idx, _ in enumerate(log_files):
    
        et_transitions = computed_data.results.excited_states.et_transitions[log_idx]
        et_oscs = computed_data.results.excited_states.et_oscs[log_idx]
        et_rotats = computed_data.results.excited_states.et_rot[log_idx]
        et_sym = computed_data.results.excited_states.et_sym[log_idx]
        
        TD_output = computed_data.derived.excited_states.TD_output[log_idx]
        grid_X = computed_data.derived.excited_states.grid_X[log_idx]
        grid_Y = computed_data.derived.excited_states.grid_Y[log_idx]
        grid_Z = computed_data.derived.excited_states.grid_Z[log_idx]
        all_data_dip = computed_data.derived.excited_states.all_data_dip[log_idx]
        Tozer_lambda = computed_data.derived.excited_states.Tozer_lambda[log_idx]
        
        # Skip if data is missing
        if (
            elements_3D_coords is None
            or atom_pairs is None
            or atoms_Z is None
            or et_transitions is None
            or et_oscs is None
            or et_rotats is None
            or et_sym is None
            or TD_output is None
            or grid_X is None
            or grid_Y is None
            or grid_Z is None
            or all_data_dip is None
            or Tozer_lambda is None
        ):
            continue
        
        # Skip if et_transitions is empty
        if (len(et_transitions) == 0):
            continue
        
        print("")
        
        for k, _ in enumerate(et_transitions):
            
            # Skip if no Tozer_lambda computed
            # Only print the visualization of the user-selected excited state
            if (
                Tozer_lambda[k] is None
            ):
                continue
            
            print("Excited State EDD visualization in progress for the transition:", k+1)
            discretized_EDD_voxels = [TD_output[k][0]]
            visu_mayavi.viz_EDD(
                discretized_EDD_voxels,
                grid_X,
                grid_Y,
                grid_Z,
                elements_3D_coords,
                atom_pairs,
                atoms_Z,
                et_sym[k],
                file_name=output_dir+"img"+f"-log{log_idx + 1}", labels=[state_number(et_sym, k)]
            ) 
            
            # Transition density with its transition dipole (every allowed state; the overlap
            # picture Oif it replaces was only drawn for chiral molecules)
            tdms = computed_data.derived.excited_states.transition_dipoles[log_idx]
            tdm = tdms[k] if tdms is not None else None
            if tdm is not None:
                print("Transition density visualization in progress for the transition:", k+1)
                visu_mayavi.viz_TDEN([TD_output[k][3]], grid_X, grid_Y, grid_Z,
                                     elements_3D_coords, atom_pairs, atoms_Z, et_sym[k],
                                     file_name=output_dir + "img" + f"-log{log_idx + 1}",
                                     labels=[state_number(et_sym, k)],
                                     vectors=[tdm["dipole"], tdm.get("magnetic")])
                visu_mayavi.viz_TDIP(elements_3D_coords, atom_pairs, atoms_Z, et_sym[k], tdm["dipole"],
                                     tdm["centre"], magnetic=tdm.get("magnetic"),
                                     file_name=output_dir + "img" + f"-log{log_idx + 1}",
                                     label=state_number(et_sym, k))
            data_dip = all_data_dip[k]
            if data_dip is not None:
                print("Generating transition dipole images for selected transition:", k+1),
                visu_mayavi.viz_dip(
                    data_dip,
                    elements_3D_coords,
                    atom_pairs,
                    atoms_Z,
                    et_sym[k],
                    file_name=output_dir+"img"+f"-log{log_idx + 1}",
                    labels=[state_number(et_sym, k)]
                )


def output_optimized_es_absorption_and_CD_spectrum(computed_data: ComputedData, config: Config, do_emi, do_CPL):
    
    """Emission and CPL spectra (pictures and text files) of every optimized excited state.

    Parameters
    ----------
    computed_data : ComputedData
    config : Config
    do_emi, do_CPL : bool
    """
    log_files = computed_data.metadata.log_files
    job_types = computed_data.comp_details.general.job_type
    output_dir = computed_data.metadata.output_dir
    
    for log_idx,log_file in enumerate(log_files):
        
        jobs = job_types[log_idx]
        
        if not any("_es" in jb for jb in jobs):
            continue
        
        emi_state = config_logfile(config, computed_data, log_idx).excited_state_number
        if emi_state is None:
            # Invalid emi_state so it will stop when checking the emi_index
            emi_state = -1
        
        et_energy = computed_data.results.excited_states.et_energies[log_idx]
        et_osc = computed_data.results.excited_states.et_oscs[log_idx]
        et_rot = computed_data.results.excited_states.et_rot[log_idx]

        # Skip if data is missing
        if (
            et_energy is None
            or et_osc is None
        ):
            continue

        emi_index = emi_state - 1
        if emi_index < 0 or emi_index >= len(et_energy):
            print("Incoherent excited state optimization detected. Problem with root number")
            print(f"Emission state (user input) = {emi_state} ; expected to be within [1, {len(et_energy)}]")
            continue
        
        emi_energy = et_energy[emi_index]
        emi_osc = et_osc[emi_index]
        
        if et_rot is not None:
            emi_rotat = et_rot[emi_index]
        else:
            emi_rotat = 0.0
        
        xvalues = computed_data.derived.optimized_excited_states.xvalues[log_idx]
        emi_spectrum = computed_data.derived.optimized_excited_states.emi_spectrum[log_idx]
        CPL_spectrum = computed_data.derived.optimized_excited_states.CPL_spectrum[log_idx]
        
        # Skip if no data
        if (
            emi_spectrum is None
            or len(emi_spectrum) == 0
            or xvalues is None
        ):
            continue
        
        if (
            do_emi
        ):
            if (
                do_CPL
                and CPL_spectrum is not None
                and len(CPL_spectrum) != 0
            ):
                # Write both Emission and CPL spectra in text file to compare with experimental data
                visu_txt.textfile_emiCD(xvalues, emi_spectrum, CPL_spectrum, file_name=output_dir+f"log{log_idx + 1}-") 
            # If no CPL write only Emission spectrum in text file to compare with experimental data
            else :
                visu_txt.textfile_emiUV(xvalues, emi_spectrum, file_name=output_dir+f"log{log_idx + 1}-")   
            
            # Plotting emiUV spectrum 
            visu_plots.emiUV([emi_energy], [emi_osc], xvalues, emi_spectrum, file_name=output_dir+f"img-log{log_idx + 1}")
            print(f"Generating Emission spectrum done for {log_file}.")            
        
        if (
            do_CPL
            and CPL_spectrum is not None
            and len(CPL_spectrum) != 0
        ):
            # plot CPL spectrum 
            visu_plots.emiCD([emi_energy], [emi_rotat], xvalues, CPL_spectrum, file_name=output_dir+f"img-log{log_idx + 1}")
            print(f"CPL spectrum done for {log_file}.") 


def output_optimized_es_EDD_DIP(computed_data: ComputedData, config: Config):
    
    """Density difference and dipole pictures of the emitting state of every optimized excited state."""
    ref_id = computed_data.metadata.ref_log_file_idx
    output_dir = computed_data.metadata.output_dir
    log_files = computed_data.metadata.log_files
    job_types = computed_data.comp_details.general.job_type
    
    print("")
    
    for log_idx,log_file in enumerate(log_files):
        
        jobs = job_types[log_idx]
        
        if not any("_es" in jb for jb in jobs):
            continue
    
        # Draw the molecule of the log whose orbitals gave the EDD (opt_es log or reference)
        geom_idx = computed_data.derived.optimized_excited_states.orbitals_log_idx[log_idx]
        if geom_idx is None:
            geom_idx = ref_id
        elements_3D_coords = computed_data.results.geometry.elements_3D_coords[geom_idx]
        atom_pairs = computed_data.molecule.connectivity.atom_pairs[geom_idx]
        atoms_Z = computed_data.molecule.atoms_Z[geom_idx]
    
        et_transitions = computed_data.results.excited_states.et_transitions[log_idx]
        emi_state = config_logfile(config, computed_data, log_idx).excited_state_number
        
        # Skip if data is missing
        if (
            et_transitions is None
            or emi_state is None
        ):
            continue
        
        emi_index = emi_state - 1
        if (
            emi_index < 0
            or emi_index >= len(et_transitions)
        ):
            continue
    
        et_sym = computed_data.results.excited_states.et_sym[log_idx]
        et_rot = computed_data.results.excited_states.et_rot[log_idx]
        
    
        TD_output = computed_data.derived.optimized_excited_states.TD_output[log_idx]
        grid_X = computed_data.derived.optimized_excited_states.grid_X[log_idx]
        grid_Y = computed_data.derived.optimized_excited_states.grid_Y[log_idx]
        grid_Z = computed_data.derived.optimized_excited_states.grid_Z[log_idx]
        all_data_dip = computed_data.derived.optimized_excited_states.all_data_dip[log_idx]
        
        # Skip if data is missing
        if (
            elements_3D_coords is None
            or atom_pairs is None
            or atoms_Z is None
            or et_sym is None
            or TD_output is None
            or grid_X is None
            or grid_Y is None
            or grid_Z is None
            or all_data_dip is None
        ):
            continue
        
        
        if et_rot is not None:
            emi_rotat=et_rot[emi_index]
        else:
            emi_rotat = 0.0
        
        # Using TD_output[0] instead of the index of the transition as TD_output only contain the transition of index emi_index here.
        # Emission goes from the excited state to the ground state: the electron moves back from
        # the arrival region of the absorption (rho_excited - rho_ground > 0) to its departure
        # region. The density difference is drawn as rho_ground - rho_excited, so that the
        # positive surface is where the electron arrives in the emission, as for the absorption.
        # Why (27/11/2026, Thomas Cauchy): with the sign of the absorption, the pictures had to
        # be read backwards to interpret the emission.
        discretized_EDD_voxels = [-TD_output[0][0]]
        
        # file img-log<i+1>-emi-EDD-S<n>.png (T<n> for a triplet, <n> for other states), as in the report
        print(f"Optimized Excited State EDD visualization in progress for the state {emi_state}")
        visu_mayavi.viz_EDD(
            discretized_EDD_voxels,
            grid_X,
            grid_Y,
            grid_Z,
            elements_3D_coords,
            atom_pairs,
            atoms_Z,
            et_sym[emi_index], 
            file_name=output_dir+f"img-log{log_idx + 1}-emi", labels=[state_number(et_sym, emi_index)]
        ) 
        
        tdms = computed_data.derived.optimized_excited_states.transition_dipoles[log_idx]
        tdm = tdms[emi_index] if tdms is not None else None
        if tdm is not None:
            # The sign of a transition density is arbitrary: the same picture serves the
            # absorption and the emission (no reversal, unlike the density difference)
            visu_mayavi.viz_TDEN([TD_output[0][3]], grid_X, grid_Y, grid_Z,
                                 elements_3D_coords, atom_pairs, atoms_Z, et_sym[emi_index],
                                 file_name=output_dir + f"img-log{log_idx + 1}-emi",
                                 labels=[state_number(et_sym, emi_index)],
                                 vectors=[tdm["dipole"], tdm.get("magnetic")])
            visu_mayavi.viz_TDIP(elements_3D_coords, atom_pairs, atoms_Z, et_sym[emi_index], tdm["dipole"],
                                 tdm["centre"], magnetic=tdm.get("magnetic"),
                                 file_name=output_dir + f"img-log{log_idx + 1}-emi",
                                 label=state_number(et_sym, emi_index))

        if (emi_rotat != 0.0):
            data_dip = all_data_dip[emi_index]
            if data_dip is not None:
                # charge-transfer arrow reversed as the density difference (emission)
                data_dip = {k: ((v[1], v[0]) if k == "CTDIP" else v) for k, v in data_dip.items()}
                visu_mayavi.viz_dip(
                    data_dip,
                    elements_3D_coords,
                    atom_pairs,
                    atoms_Z,
                    et_sym[emi_index],
                    file_name=output_dir+f"img-log{log_idx + 1}-emi",
                    labels=[state_number(et_sym, emi_index)]
                )
                print(f"Generating transition dipole images for selected transition: {emi_state}")


def output_nto_images(computed_data: ComputedData, config: Config):
    """Pictures of the NTO pairs (hole and electron, colours output.colors.NTO) of the selected states,
    computed one pair at a time on the common grid (calc_orb.nto_pair).
    """
    ref = computed_data.derived.excited_states.nto_reference
    if ref is None or config.output.verbosity == "text":
        return
    for log_idx, states in enumerate(computed_data.derived.excited_states.nto_pairs):
        et_sym = computed_data.results.excited_states.et_sym[log_idx]
        for state in states or []:
            # file img-log<i+1>-NTO-S<n>-pair<p> (T<n> for a triplet), as in the report
            label = state_label(et_sym, state["state"] - 1) if et_sym else str(state["state"])
            for pair_idx, pair in enumerate(state["pairs"], 1):
                prefix = (computed_data.metadata.output_dir +
                          f"img-log{log_idx+1}-NTO-{label}-pair{pair_idx}")
                print(f"Generating NTO images: log {log_idx+1}, state {state['state']}, pair {pair_idx}")
                # Common grid, every k-th point (smooth images, like the MOs), evaluated directly
                data, x, y, z = calc_orb.nto_pair(ref, pair["hole"], pair["electron"],
                                                  grid_step=obk_step,
                                                  nproc=config.resources.nproc or 20,
                                                  reduction=config.output.image_grid_reduction)
                visu_mayavi.viz_MO(data, x, y, z, ref.results.geometry.elements_3D_coords,
                                   ref.molecule.connectivity.atom_pairs, ref.molecule.atoms_Z,
                                   file_name=prefix, labels=["hole", "electron"], kind="NTO",
                                   reduction=1)  # already reduced by nto_pair


def output_soc_pictures(computed_data: ComputedData):
    """Jablonski-Perrin diagram of every model with spin-orbit couplings (ORCA dosoc), and the
    phosphorescence spectrum of the optimized triplets."""
    output_dir = computed_data.metadata.output_dir
    for log_idx, soc in enumerate(computed_data.results.excited_states.soc):
        if soc and soc["socme"]:
            visu_plots.jablonski(soc, file_name=output_dir + f"img-log{log_idx + 1}")
    for log_idx, phospho in enumerate(computed_data.derived.optimized_excited_states.phosphorescence):
        if phospho:
            visu_plots.phospho([s["energy"] for s in phospho["sublevels"]],
                               [s["fosc"] for s in phospho["sublevels"]],
                               phospho["xvalues"], phospho["spectrum"],
                               file_name=output_dir + f"img-log{log_idx + 1}")


def generate_visualization(computed_data: ComputedData, config: Config, compute_flags: ComputeFlags):
    
    """Draw all the pictures of the report.

    Parameters
    ----------
    computed_data : ComputedData
    config : Config
    compute_flags : ComputeFlags
        Pictures to draw.
    """
    print("\n=================================================================")
    print(" Visualization")
    print("=================================================================")

    # Every geometry of the molecule is seen from the side chosen for the reference
    ref_id = computed_data.metadata.ref_log_file_idx
    if ref_id is not None:
        visu_mayavi.set_reference_geometry(computed_data.results.geometry.elements_3D_coords[ref_id],
                                           computed_data.molecule.connectivity.atom_pairs[ref_id],
                                           computed_data.molecule.atoms_Z[ref_id])

    # Generate molecule representation
    if compute_flags.compute_molecule_representations:
        output_molecule_representations(computed_data)
    
    # Generate the outputs for the MO Diagrams
    if compute_flags.compute_MO_diagrams:
        output_MO_diagrams_clean(computed_data)
    
    # Generate the outputs for the MEP maps
    if compute_flags.compute_MEP_maps :
        output_MEP_maps(computed_data)
    
    # Generate the outputs for the fukui functions
    if compute_flags.compute_fukui_functions :
        output_fukui_functions(computed_data)
    
    # Generate the output for the Absorption and CD spectrum of the excited states
    do_abso = compute_flags.compute_es_Abso_spectrum
    do_cd = compute_flags.compute_es_CD_spectrum
    if (do_abso or do_cd):
        output_es_absorption_and_CD_spectrum(computed_data, do_cd, do_abso)
    
    # Generate the outputs for the EDD, OIF and DIP of the excited states
    if compute_flags.compute_es_transitions_and_EDD :
        output_es_EDD_OIF_DIP(computed_data)

    if compute_flags.compute_es_nto:
        output_nto_images(computed_data, config)
    
    # Generate the output for the Absorption and CD spectrum of the optimized excited states
    do_emi = compute_flags.compute_optimized_es_emi_spectrum
    do_CPL = compute_flags.compute_optimized_es_CPL_spectrum
    if (do_emi or do_CPL):
        output_optimized_es_absorption_and_CD_spectrum(computed_data, config, do_emi, do_CPL)
    
    # Generate the outputs for the transitions of the optimized excited states
    if compute_flags.compute_optimized_es_transitions_and_EDD :
        output_optimized_es_EDD_DIP(computed_data, config)

    # Spin-orbit couplings and phosphorescence (ORCA dosoc)
    if compute_flags.compute_es_transitions_and_EDD or compute_flags.compute_optimized_es_emi_spectrum:
        output_soc_pictures(computed_data)
    
