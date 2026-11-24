"""Reading of the YAML configuration file."""

import yaml
from pydantic import ValidationError

from quchemreport.config.config import Config

def load_yaml_config(path: str) -> Config:
    """Read the YAML configuration file and validate it.

    Parameters
    ----------
    path : str
        Path of the YAML file.

    Returns
    -------
    Config

    Raises
    ------
    FileNotFoundError
        The file does not exist.
    ValueError
        The file is not valid YAML, or an option is wrong (the message lists the options).
    """
    try:
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing the YAML file {path}: {e}")

    try:
        return Config(**config_dict)
    except ValidationError as e:
        errors = e.errors()
        messages = "; ".join([f"{err['loc']}: {err['msg']}" for err in errors])
        raise ValueError(f"Validation Error | {messages}")
