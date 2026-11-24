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


def state_number(state_symmetries, k):
    """Number of the state k (0-based, in the order of the log, i.e. of the energies) within
    its multiplicity: the n of S<n> / T<n>. A TD with singlets and triplets (ORCA: triplets
    true; Gaussian: 50-50) gives T1, S1, T2... instead of the global rank (T1, S2, T3...).
    States without a multiplicity (unrestricted calculations) keep their global number k + 1."""
    letter = spin_letter(state_symmetries[k])
    if not letter:
        return k + 1
    return sum(1 for sym in state_symmetries[:k + 1] if spin_letter(sym) == letter)


def state_label(state_symmetries, k):
    """'S1', 'T2'... of the state k (see state_number); '3' for a state without a multiplicity."""
    return f"{spin_letter(state_symmetries[k])}{state_number(state_symmetries, k)}"


def state_labels(state_symmetries):
    return [state_label(state_symmetries, k) for k in range(len(state_symmetries))]
