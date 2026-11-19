"""Names of the excited-state images, shared by visualization_engine and report_generator."""


def spin_letter(state_symmetry):
    """'S' for a singlet state, 'T' for a triplet, '' otherwise (e.g. unrestricted
    Gaussian states, labelled '3.002-?Sym' by cclib).

    state_symmetry is the symmetry string of ONE state (cclib etsyms element:
    'Singlet-A', 'Triplet-B2', 'Singlet' for ORCA...)."""
    sym = str(state_symmetry)
    if "Singlet" in sym:
        return "S"
    if "Triplet" in sym:
        return "T"
    return ""
