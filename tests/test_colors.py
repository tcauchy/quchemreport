"""Colours of the iso-surfaces (output.colors): YAML values, defaults, caption names."""

import pytest
import yaml
from pydantic import ValidationError

from quchemreport.config.config import Colors, Config
from quchemreport.utility_services import parameters
from quchemreport.utility_services.parameters import (DEFAULT_SURFACE_COLORS, color_name,
                                                      hex_to_rgb, set_surface_colors,
                                                      surface_color_names, surface_colors)


@pytest.fixture(autouse=True)
def default_colors():
    """Every test starts and ends with the default colours (module state)."""
    set_surface_colors()
    yield
    set_surface_colors()


def test_defaults():
    """Without output.colors: the documented defaults, with the names of the captions; the
    charge-transfer dipole arrow takes the EDD gain colour."""
    set_surface_colors(Colors().as_dict())
    for kind, signs in DEFAULT_SURFACE_COLORS.items():
        for sign, (hexa, name) in signs.items():
            assert surface_colors[kind][sign] == hex_to_rgb(hexa)
            assert surface_color_names[kind][sign] == name
    assert surface_color_names["NTO"] == {"positive": "blue-grey", "negative": "cream"}
    # cream, not white: the hydrogens are white spheres
    assert all(surface_color_names[k]["negative"] == "cream" for k in ("MO", "EDD", "fukui", "NTO"))
    assert parameters.colors["CTDIP"] == surface_colors["EDD"]["positive"]


def test_example_yaml_gives_the_defaults():
    """The colors block of example.yaml repeats the defaults."""
    with open("example.yaml") as f:
        given = yaml.safe_load(f)["output"]["colors"]
    for kind, signs in DEFAULT_SURFACE_COLORS.items():
        for sign, (hexa, _) in signs.items():
            assert given[kind][sign].upper() == hexa


def test_user_colors_and_names():
    """Hexadecimal, word and list colours; a word names itself, a value gets color_name, an
    explicit name wins, an omitted colour keeps its default."""
    colors = Colors(**{"NTO": {"positive": "#2F4B7C", "negative": "lightpink"},
                       "EDD": {"positive": [0.0, 0.4, 0.7], "positive_name": "dark blue"},
                       "fukui": {"negative": "tab:orange"}})
    set_surface_colors(colors.as_dict())
    assert surface_colors["NTO"]["positive"] == hex_to_rgb("#2F4B7C")
    assert surface_color_names["NTO"]["positive"] == color_name(hex_to_rgb("#2F4B7C"))
    assert surface_color_names["NTO"]["negative"] == "lightpink"
    assert surface_colors["EDD"]["positive"] == (0.0, 0.4, 0.7)
    assert surface_color_names["EDD"]["positive"] == "dark blue"
    assert surface_colors["EDD"]["negative"] == hex_to_rgb("#FFEBDC")
    assert surface_color_names["fukui"] == {"positive": "blue", "negative": "orange"}
    assert surface_colors["MO"]["positive"] == hex_to_rgb("#66003C")
    assert parameters.colors["CTDIP"] == (0.0, 0.4, 0.7)


def test_default_value_keeps_its_name():
    """A default colour written in the YAML (as example.yaml does) keeps the caption name."""
    set_surface_colors(Colors(EDD={"positive": "#008080"}).as_dict())
    assert surface_color_names["EDD"]["positive"] == "blue"


@pytest.mark.parametrize("wrong", [
    {"MO": {"positive": "#12345"}},           # not a colour
    {"MO": {"positive": "notacolour"}},
    {"MO": {"positive": [0, 128, 255]}},      # 0-255 instead of 0-1
    {"MO": {"positive": [0.1, 0.2]}},
    {"MO": {"postive": "#000000"}},           # misspelt key
    {"OM": {"positive": "#000000"}},          # unknown kind
])
def test_wrong_colors_are_refused(wrong):
    with pytest.raises(ValidationError):
        Colors(**wrong)


@pytest.mark.parametrize("hexa, name", [
    ("#66003C", "dark purple"), ("#F2E6ED", "white"), ("#5B6A8F", "greyish blue"),
    ("#F4D3CB", "light pink"), ("#808080", "grey"), ("#000080", "dark blue"),
    ("#FF0000", "red"), ("#0D872E", "dark green"), ("#FFEBDC", "cream"), ("#FCE6D4", "cream"),
])
def test_color_name(hexa, name):
    assert color_name(hex_to_rgb(hexa)) == name


def test_config_reads_output_colors(tmp_path):
    """The colors block reaches Config.output (example.yaml with an NTO colour changed)."""
    with open("example.yaml") as f:
        data = yaml.safe_load(f)
    data["output"]["colors"] = {"NTO": {"negative": "#FFE4E1"}}
    data["logfiles"] = data["logfiles"][:1]
    log = tmp_path / "mol.log"
    log.write_text("")
    data["logfiles"][0]["path"] = str(log)
    config = Config(**data)
    assert config.output.colors.NTO.negative == hex_to_rgb("#FFE4E1")
    assert config.output.colors.MO.positive is None


def test_captions_follow_the_colors():
    """The caption names come from output.colors (generate_report.color_of)."""
    from quchemreport.report_generator.generate_report import color_of
    assert color_of("NTO", "positive", True) == "Blue-grey"
    set_surface_colors(Colors(NTO={"positive": "teal"}).as_dict())
    assert color_of("NTO", "positive", True) == "Teal"
    assert color_of("MO", "negative") == "cream"
