"""Command line of QuChemReport: ``python cli.py -c report.yaml``.

Builds the report described by the YAML file. Errors are printed and the program exits with
status 1; the traceback is printed when ``logging.level`` is debug.
"""

import argparse
import traceback
import sys

import quchemreport.execution_engine.engine as engine
from quchemreport.config.YAML_loader import load_yaml_config

def main():
    """Read the YAML file given with -c/--config and run the report."""
    parser = argparse.ArgumentParser(description="Command-line interface for running quchemreport.")
    parser.add_argument("-c", "--config", required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    try:
        config = load_yaml_config(args.config)
    except Exception as e:
        print(e)
        sys.exit(1)
    
    try:
        engine.run(config)
    except Exception as e:
        print("engine error: ",e)
        if config.logging.level == "debug":
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

