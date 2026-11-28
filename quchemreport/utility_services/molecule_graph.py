"""Molecular graph from a geometry: bonds, equivalent atoms, formula (RDKit).

The logs of Gaussian and ORCA give atoms and coordinates, not bonds. The bonds are perceived
from the interatomic distances; they are used for the sticks of the pictures, the rings of the
camera choice and the classes of equivalent atoms that group the MEP extrema.

Why RDKit and not Open Babel (decision of 2026-11-28, maintainer): Open Babel is licensed under
the GPL version 2 only, which does not combine with the licence planned for QuChemReport, and
has had no release since 2020; RDKit (BSD) was already in the environment. Measured on the 146
molecules of the MEP studies (H to I, Cu, Zn, Pt) before the switch: the same bonds for 146/146,
the same classes of equivalent atoms for 144/146 and the same MEP extrema tables for all
(validation/revue_complete_20261128/rdkit_vs_openbabel/README.md).
"""

from collections import Counter
from typing import List, Optional, Sequence, Tuple

import numpy as np

BOHR_TO_ANGSTROM = 0.52917721092


def connectivity_molecule(atoms_Z: Sequence[int], coords: np.ndarray):
    """RDKit molecule with the atoms, one conformer and single bonds perceived from distances.

    Parameters
    ----------
    atoms_Z : sequence of int
        Atomic numbers.
    coords : array (n, 3)
        Coordinates in Angstrom.

    Returns
    -------
    rdkit.Chem.Mol
        Bonds from the covalent radii (RDKit ``DetermineConnectivity``, two atoms are bonded
        when their distance is below 1.3 times the sum of their radii). Every bond is single:
        bond orders need the charge and often fail (ions, radicals, metals), and the pictures,
        the camera and the equivalent atoms only need the connections.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdDetermineBonds
    from rdkit.Geometry import Point3D

    RDLogger.DisableLog("rdApp.*")
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    # Built atom by atom, never from an XYZ text block: once Mayavi has started Qt, the locale
    # of the user applies (fr_FR on the lab machines) and RDKit's XYZ reader then reads
    # "0.7572" as 0, so that no bond is found (measured 28/11/2026). Open Babel protected
    # itself from the locale when it read the XYZ block.
    rw = Chem.RWMol()
    for z in atoms_Z:
        rw.AddAtom(Chem.Atom(int(z)))
    conformer = Chem.Conformer(len(atoms_Z))
    for i, (x, y, w) in enumerate(coords):
        conformer.SetAtomPosition(i, Point3D(float(x), float(y), float(w)))
    rw.AddConformer(conformer, assignId=True)
    mol = rw.GetMol()
    rdDetermineBonds.DetermineConnectivity(mol)
    return mol


def bond_pairs(atoms_Z: Sequence[int], coords: np.ndarray) -> List[Tuple[int, int]]:
    """Bonds as pairs of atom indices (from 0), in both directions, sorted.

    Parameters
    ----------
    atoms_Z : sequence of int
        Atomic numbers.
    coords : array (n, 3)
        Coordinates in Angstrom.

    Returns
    -------
    list of (int, int)
        (i, j) and (j, i) for each bond, sorted by i then j: the order of the former Open Babel
        loop over all atom pairs, so that the pictures are drawn in the same order.
    """
    mol = connectivity_molecule(atoms_Z, coords)
    pairs = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        pairs += [(i, j), (j, i)]
    return sorted(pairs)


def symmetry_classes(atoms_Z: Optional[Sequence[int]], coords: np.ndarray) -> List[int]:
    """Class of each atom in the molecular graph: atoms in the same class are equivalent.

    Used to group MEP extrema near equivalent atoms (the three H of a methyl group). The
    classes come from the graph only (elements and connections, RDKit ``CanonicalRankAtoms``
    without breaking ties and without chirality), not from the geometry: two CF3 groups in
    different conformations are in the same class, and it is the tolerance on the potential
    (mep_surface.EQUIVALENT_TOL) that keeps their extrema apart when they differ.

    Parameters
    ----------
    atoms_Z : sequence of int or None
        Atomic numbers.
    coords : array (n, 3)
        Nuclei in bohr.

    Returns
    -------
    list of int
        One class per atom; one class per atom (no grouping) when atoms_Z is None, does not
        match the coordinates, or RDKit fails.
    """
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    if atoms_Z is None or len(atoms_Z) != len(coords):
        return list(range(len(coords)))
    try:
        from rdkit import Chem
        mol = connectivity_molecule(atoms_Z, coords * BOHR_TO_ANGSTROM)
        return [int(c) for c in Chem.CanonicalRankAtoms(mol, breakTies=False, includeChirality=False)]
    except Exception:
        return list(range(len(coords)))


def formula(atoms_Z: Sequence[int], charge: int = 0) -> str:
    """Molecular formula in the Hill order, with the charge as repeated signs.

    Hill order: C first, then H, then the other elements alphabetically; without carbon, all
    the elements alphabetically (H2O, Cl4Pt). The charge is written as Open Babel did when it
    gave the formula through cclib: one sign per unit (C3H6O+, H2O--, C4H7NO+++), so that the
    formulas of the reports do not change.

    Parameters
    ----------
    atoms_Z : sequence of int
        Atomic numbers.
    charge : int
        Total charge of the molecule.

    Returns
    -------
    str
    """
    from rdkit import Chem

    table = Chem.GetPeriodicTable()
    counts = Counter(table.GetElementSymbol(int(z)) for z in atoms_Z)
    if "C" in counts:
        order = ["C"] + (["H"] if "H" in counts else []) + sorted(e for e in counts if e not in ("C", "H"))
    else:
        order = sorted(counts)
    text = "".join(e + (str(counts[e]) if counts[e] > 1 else "") for e in order)
    charge = int(charge or 0)
    return text + ("+" * charge if charge > 0 else "-" * -charge)
