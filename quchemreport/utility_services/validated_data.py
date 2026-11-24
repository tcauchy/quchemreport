"""Result of the conformity tests (quality_check_engine.conformity.tests)."""

from dataclasses import dataclass
from typing import List, Union
from quchemreport.utility_services.log_data import LogData

@dataclass
class ValidatedData:
    """Data validated by the conformity tests and passed to the computations.

    ``job_types``, ``charges``: per parsed model; ``nres_noES``: nuclear repulsion energies
    without the optimized excited states; ``charge_ref``: charge of the reference;
    ``discret_proc``: whether grid computations are possible; ``ref_log_file_idx``: index of
    the reference model; ``data_for_discretization``: model used for the grids (None if no
    log has the basis set and the MO coefficients).
    """
    job_types: List[List[str]]
    nres_noES: List[float]
    charges: List[int]
    charge_ref: int
    discret_proc: bool
    mo_viz_done: bool
    ref_log_file_idx: int
    data_for_discretization: LogData