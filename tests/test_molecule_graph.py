"""Bonds, equivalent atoms and formula from a geometry (utility_services.molecule_graph).

Expected values: the Hill order (C, H, then alphabetical; alphabetical without carbon) is the
convention of Chemical Abstracts; the charge signs are those Open Babel wrote through cclib
until 28/11/2026 (checked on 2026-11-28 for charges -2 to +3)."""
import numpy as np
import pytest

from quchemreport.utility_services.molecule_graph import bond_pairs, formula, symmetry_classes

WATER_Z = [8, 1, 1]
WATER = np.array([[0, 0, 0.1173], [0, 0.7572, -0.4692], [0, -0.7572, -0.4692]])  # Angstrom
METHANOL_Z = [6, 8, 1, 1, 1, 1]
METHANOL = np.array([[-0.0465, 0.6608, 0.0], [-0.0465, -0.7563, 0.0], [-1.0868, 0.9812, 0.0],
                     [0.4427, 1.0723, 0.8901], [0.4427, 1.0723, -0.8901], [0.8696, -1.0612, 0.0]])


@pytest.mark.parametrize("Z, charge, expected", [
    ([6, 6, 6, 8] + [1] * 6, 0, "C3H6O"),
    ([8, 1, 1], 0, "H2O"),
    ([7, 1, 1, 1, 1], 1, "H4N+"),
    ([6, 1, 9, 17, 35], 0, "CHBrClF"),
    ([78, 17, 17, 17, 17], -2, "Cl4Pt--"),
    ([8, 8], 3, "O2+++"),
    ([6, 53, 1, 1, 1], -1, "CH3I-"),
    ([6, 6], 0, "C2"),
])
def test_formula_hill_order_and_charge(Z, charge, expected):
    assert formula(Z, charge) == expected


def test_bond_pairs_both_directions_sorted():
    assert bond_pairs(WATER_Z, WATER) == [(0, 1), (0, 2), (1, 0), (2, 0)]
    pairs = bond_pairs(METHANOL_Z, METHANOL)
    assert len(pairs) == 2 * 5 and pairs == sorted(pairs)


def test_symmetry_classes_from_the_graph():
    """Methanol: the three methyl H in one class, the hydroxyl H apart; geometry in bohr."""
    classes = symmetry_classes(METHANOL_Z, METHANOL / 0.52917721092)
    assert classes[2] == classes[3] == classes[4]
    assert len({classes[0], classes[1], classes[2], classes[5]}) == 4
    assert symmetry_classes(None, METHANOL) == list(range(6))
    assert symmetry_classes([6, 8], METHANOL) == list(range(6))


def test_bonds_do_not_depend_on_the_locale():
    """Under a locale with a decimal comma (Qt sets the user's one, fr_FR on the lab machines),
    an XYZ text block read by RDKit loses its decimals and no bond is found (28/11/2026)."""
    import locale
    previous = locale.setlocale(locale.LC_NUMERIC)
    try:
        locale.setlocale(locale.LC_NUMERIC, "fr_FR.UTF-8")
    except locale.Error:
        pytest.skip("fr_FR.UTF-8 locale not installed")
    try:
        assert bond_pairs(WATER_Z, WATER) == [(0, 1), (0, 2), (1, 0), (2, 0)]
    finally:
        locale.setlocale(locale.LC_NUMERIC, previous)
