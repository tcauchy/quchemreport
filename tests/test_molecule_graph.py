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


FORMALDEHYDE_Z = [6, 8, 1, 1]
FORMALDEHYDE = np.array([[0, 0, -0.5297], [0, 0, 0.6738], [0, 0.9385, -1.1121], [0, -0.9385, -1.1121]])


def test_identifiers_of_a_closed_shell_molecule():
    """Formaldehyde: standard InChI and monoisotopic mass as given by PubChem (CID 712:
    InChI=1S/CH2O/c1-2/h1H2, 30.010564683 Da)."""
    from quchemreport.utility_services.molecule_graph import identifiers
    ids = identifiers(FORMALDEHYDE_Z, FORMALDEHYDE, 0)
    assert ids.inchi == "InChI=1S/CH2O/c1-2/h1H2"
    assert ids.smiles == "C=O"
    assert ids.monoisotopic_mass == pytest.approx(30.010564683, abs=1e-6)
    assert ids.note is None


def test_identifiers_of_an_ion_and_of_a_radical():
    """Cation: the InChI has its charge layer and the mass loses one electron (CODATA 2018,
    5.48579909065e-4 u). Neutral radical (CH3): no bond orders agree with the valences, so no
    InChI nor SMILES, but the reason and the mass."""
    from quchemreport.utility_services.molecule_graph import identifiers
    hydronium = identifiers([8, 1, 1, 1], np.array([[0, 0, 0.1], [0, 0.94, -0.25], [0.81, -0.47, -0.25],
                                                    [-0.81, -0.47, -0.25]]), 1)
    assert hydronium.inchi == "InChI=1S/H2O/h1H2/p+1"
    assert hydronium.monoisotopic_mass == pytest.approx(19.018389 - 0.000549, abs=1e-5)
    methyl = identifiers([6, 1, 1, 1], np.array([[0, 0, 0], [0, 1.08, 0], [0.935, -0.54, 0], [-0.935, -0.54, 0]]), 0)
    assert methyl.inchi is None and methyl.smiles is None and methyl.note
    assert methyl.monoisotopic_mass == pytest.approx(12 + 3 * 1.00782503223, abs=1e-6)


def test_report_says_why_the_inchi_is_missing():
    """Identification table: when RDKit found no bond orders, a row gives the reason."""
    from pylatex import Section
    from quchemreport.report_generator import generate_report as gr
    from quchemreport.utility_services.computed_data import init_computed_data
    cd = init_computed_data(1)
    cd.molecule.formula = "CH3"
    cd.molecule.monoisotopic_mass = 15.0234751
    cd.molecule.identifiers_note = "no bond orders agree with the charge and the usual valences (radical or unusual ion)"
    section = Section("Molecule")
    assert gr.render_molecule_identification(section, cd, "methyl", "full")
    tex = section.dumps()
    assert "InChI, SMILES" in tex and "not determined from the geometry: no bond orders agree" in tex
    assert "15.02348 Da" in tex


def test_identifiers_of_a_single_atom():
    """Atoms (He, Ne in tests/data): no bond orders to search; ions keep their charge."""
    from quchemreport.utility_services.molecule_graph import identifiers
    he = identifiers([2], np.zeros((1, 3)), 0)
    assert he.inchi == "InChI=1S/He" and he.smiles == "[He]" and he.note is None
    assert he.monoisotopic_mass == pytest.approx(4.00260325413, abs=1e-6)
    chloride = identifiers([17], np.zeros((1, 3)), -1)
    assert chloride.inchi == "InChI=1S/ClH/h1H/p-1" and chloride.smiles == "[Cl-]"
