"""Execution of a report, from the configuration to the PDF file.

run() calls the stages in order (see docs/architecture.md): reading of the log files,
conformity tests, computations, pictures and report.
"""

import json
import tempfile
import traceback
from typing import List

from quchemreport.config.config import Config
from quchemreport.data_enrichment.compute_flags import resolve_compute_flags
from quchemreport.parser_engine.scanlog import process_logfile_list
import quchemreport.quality_check_engine.conformity as conformity
from quchemreport.utility_services.computed_data import ComputedData
from quchemreport.data_enrichment.compute import compute_data
from quchemreport.visualization_engine.generate_visualization import generate_visualization
from quchemreport.report_generator.generate_report import generate_report


def run(config: Config):
    # Config implicitly validated when the object is created
    """Build the report described by a configuration.

    The log files are split into steps in a temporary folder; a stage that finds an
    inconsistency stops the program with a message.

    Parameters
    ----------
    config : Config
        Validated configuration (YAML_loader.load_yaml_config).

    Raises
    ------
    ValueError
        No log file could be read.
    """
    from quchemreport.visualization_engine import visu_mayavi
    visu_mayavi.set_image_size(config.output.image_size)
    visu_mayavi.set_image_grid_reduction(config.output.image_grid_reduction,
                                         config.output.image_grid_reduction_sensitive)
    nto = config.output.include.excited_states.natural_transition_orbitals
    if nto.grid_step is not None or nto.nproc is not None:
        print("Note: natural_transition_orbitals.grid_step / nproc are ignored; the NTO images use "
              "the common grid with output.image_grid_reduction and resources.nproc.")

    tmpdirname = tempfile.mkdtemp()
        
    # Parses the logfiles
    print("Parsing input files...")
    list_logs_files, list_data_models  = process_logfile_list(config, log_storage_path=tmpdirname)
    
    if (len(list_data_models)) == 0:
        print("There is no valid data. Program will exit.")
        raise ValueError("No valid data detected in the provided log files.")
    print(len(list_data_models), "valid files detected.")
    
    # Check the quality of the data
    print('\nStarting conformity tests.')
    validated_data = conformity.tests(list_data_models, config)
    
    # Resolve which compute function to do based on the user-selection of outputs
    compute_flags = resolve_compute_flags(config)
    
    # Compute all data needed for the visualization and the generation of the report 
    computed_data: ComputedData = compute_data(list_data_models, validated_data, config, compute_flags)
    
    # Generate the visualizations based on the computed data
    generate_visualization(computed_data, config, compute_flags)
    
    # Generate the final report based on the computed data
    generate_report(computed_data, config, "clean")
    
    return
