"""Selection of the relevant excited states (data_enrichment.state_selection, selection.mode
relevant) and its options in the YAML file."""
import pytest

from quchemreport.data_enrichment.state_selection import relevant_states, dropped_by_cap


def singlets(n):
    return ["Singlet-A"] * n


def test_lowest_states_of_each_multiplicity_are_kept():
    """S1-S3 and T1-T3 even when dark; nothing else without bands."""
    energies = [20000, 25000, 27000, 30000, 32000, 34000, 36000, 38000]
    syms = ["Triplet-A", "Singlet-A", "Triplet-A", "Singlet-A", "Triplet-A", "Singlet-A", "Triplet-A", "Singlet-A"]
    kept = relevant_states(energies, syms, oscs=[0.0] * 8)
    assert sorted(kept) == [0, 1, 2, 3, 4, 5]
    assert all(r == ["lowest"] for r in kept.values())


def test_bands_above_the_share_of_the_largest():
    """Two separated bands of 1.0 and 0.3 (30 %): both kept at 20 %, only the strong one at
    40 %; a band at the last computed state is cut, hence left out."""
    energies = [20000, 21000, 22000, 30000, 40000, 50000, 60000]
    oscs = [0.0, 0.0, 0.0, 1.0, 0.3, 0.0, 0.5]
    kept = relevant_states(energies, singlets(7), oscs=oscs, band_share=0.20)
    assert 3 in kept and 4 in kept and 6 not in kept
    assert kept[3] == ["absorption band 333 nm"]
    assert 4 not in relevant_states(energies, singlets(7), oscs=oscs, band_share=0.40)


def test_negative_cd_bands_are_compared_with_negative_ones():
    """A weak negative CD band (5 % of the positive one) is kept: it is the largest negative band."""
    energies = [20000, 21000, 22000, 30000, 40000, 55000]
    rots = [0.0, 0.0, 0.0, 100.0, -5.0, 0.0]
    kept = relevant_states(energies, singlets(6), oscs=[0.0] * 6, rots=rots)
    assert kept[4] == ["CD - band 250 nm"] and kept[3] == ["CD + band 333 nm"]


def test_cap_keeps_user_and_lowest_states_first():
    """Cap of 4: the state asked for, then S1-S3; the band state is left out and reported."""
    energies = [20000, 25000, 30000, 35000, 40000, 60000]
    oscs = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    options = dict(oscs=oscs, forced=["S4"])
    kept = relevant_states(energies, singlets(6), max_selected=4, **options)
    assert sorted(kept) == [0, 1, 2, 3] and kept[3] == ["asked for"]
    assert dropped_by_cap(energies, singlets(6), kept, **options) == [4]


def test_triplets_coupled_to_s1():
    """With SOC: T1, and the triplets within 0.3 eV of S1 coupled to it by at least 1 cm-1."""
    energies = [15000, 22000, 24000, 24500, 26000, 30000, 31000, 33000]
    syms = ["Triplet-A", "Triplet-A", "Singlet-A", "Triplet-A", "Triplet-A", "Triplet-A", "Singlet-A", "Singlet-A"]
    soc = {"socme": {(3, 1): 40.0, (4, 1): 0.5, (5, 1): 30.0}}
    kept = relevant_states(energies, syms, oscs=[0.0] * 8, soc=soc)
    # T3 at 0.06 eV of S1 (40 cm-1) kept for SOC as well as lowest; T4: 0.24 eV but 0.5 cm-1
    assert "couples to S1 (40 cm-1)" in kept[3]
    assert "T1 (spin-orbit)" in kept[0]
    assert 4 not in kept and 5 not in kept


def test_selection_options_in_the_yaml():
    """Defaults: mode relevant, 20 %, 15 states, FWHM 3000 cm-1; state names checked."""
    from quchemreport.config.config import ExcitedStatesSelection, Output
    sel = ExcitedStatesSelection()
    assert (sel.mode, sel.band_share, sel.max_selected_states, sel.states) == ("relevant", 0.20, 15, [])
    assert ExcitedStatesSelection(states=["s4", 7]).states == ["s4", "7"]
    with pytest.raises(ValueError):
        ExcitedStatesSelection(states=["lumo"])
    with pytest.raises(ValueError):
        ExcitedStatesSelection(band_share=1.5)
    assert Output.model_fields["spectrum_fwhm"].default == 3000.0
