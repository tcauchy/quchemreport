## -*- encoding: utf-8 -*-

"""Default parameters of the computations and pictures.

obk_step is the step of the grids (bohr) and obk_extand their margin around the atoms; FWHM the
width of the Gaussian bands of the spectra (cm-1); img_width / img_height the default size of
the 3D pictures (output.image_size overrides it); surface_colors the colours of the iso-surfaces
(output.colors overrides them, set_surface_colors), colors those of the other surfaces and of the
dipole arrows, scale the scale factors of the arrows. The camera views are chosen from the
geometry (visualization_engine/camera.py).
"""

from quchemreport.utility_services import units

## Default parameters

## Orbkit parameters
# default grid step size
# Spacing of cubic voxels in the grid 
obk_step = 0.075 # Depending on the molecule size, it should be adapted. 
# Oversizing the grid aroung the molecule in Bohr radii
# ORBKIT uses 5 by default, tune this as required
obk_extand = 5

## Gaussian broadening default parameters
FWHM = 3000 # 3000 cm-1 for full width at half maximum of gaussian band. 


## Mayavi parameters
# Set image size in pixels
# Default side of the 3D images; overridden by output.image_size in the YAML
img_width = 1200
img_height = 1200
# Camera views: chosen from the geometry, see visualization_engine/camera.py

# isosurface opacity
surf_opacity = 1

# MO, EDD, Fukui and NTO surfaces: surface_colors below
colors = {"Oif" : (0.95, 0.5, 0.0),  #For transitions wavefunctions overlap positive surface an orange
          "NEG" : (0.95, 0.90, 0.93),  #For the negative Oif surface, almost white
          "CTDIP" : (0.0, 0.5, 0.5), #Charge transfer dipole: the EDD gain colour (set_surface_colors)
          "GSDIP" : (0.1, 0.1, 0.1),
          "ELDIP" : (0, 0.6, 1),
          "VELDIP" : (0.6, 1, 0.2),
          "MAGDIP" : (1, 0.2, 0.6),
          "OVDIP" : (0.95, 0.5, 0.) #For transitions wavefunctions overlap dipole same as Oif surface 
          }
scale = {"CTDIP" : 1., #For transitions charge transfer dipole same as EDD surface
          "GSDIP" : 1.,
          "ELDIP" : 1./units.a0_to_A,
          "VELDIP" : 1./units.a0_to_A,
          "MAGDIP" : 1./units.a0_to_A,
          "OVDIP" : 1. #For transitions wavefunctions overlap dipole same as Oif surface 
          }

# Colours of the iso-surfaces of the 3D pictures, positive and negative parts, as "#RRGGBB";
# output.colors in the YAML overrides them (set_surface_colors). The names are those of the
# captions of the report and of docs/report.md.
#  - MO: positive and negative lobes (the former colors["OM"] and colors["NEG"]).
#  - EDD: electron gain (excited electron) and loss (hole); the charge-transfer dipole arrow
#    (colors["CTDIP"]) takes the gain colour.
#  - fukui: F+, F- and the dual descriptor (positive: electrophilic for F+ and the dual,
#    nucleophilic for F-; negative: nucleophilic for the dual).
#  - NTO: the two phases of the hole and electron orbitals. Green #0D872E and gold #F2A60D until
#    26/11/2026, replaced because they merge for red-green colour-blind readers: CIEDE2000
#    difference between them 45 in normal vision but 18 for protanopia and 27 for deuteranopia
#    (Machado 2009 simulation, severity 1). The blue-grey of the oleron map (Crameri) and the
#    cream differ mainly in lightness and stay at 44-47 in the four visions.
# Negative surfaces of MO, EDD, fukui and NTO: cream #FFEBDC instead of almost white (#F2E6ED,
# fukui #F2F2F2), so
# that a small lobe is not taken for a hydrogen (pure white spheres, Atoms_properties.csv):
# CIEDE2000 to white 9.9 instead of 8.2, 8.4-10.4 for colour-blind readers instead of 4.6-7.8.
# #FFF3F0 (asked first) was closer to white (5.6) and looked like the hydrogens. Pairs with
# the cream: MO 64-77, EDD 34-48, NTO 44-47 in the four visions (26/11/2026).
# Hexadecimal values: the former tuples (0.4, 0, 0.235)... moved by less than 0.002.
# MO positive: Tyrian purple (#66023C in the usual tables).
DEFAULT_SURFACE_COLORS = {
    "MO":    {"positive": ("#66003C", "purple"), "negative": ("#FFEBDC", "cream")},
    "EDD":   {"positive": ("#008080", "blue"),   "negative": ("#FFEBDC", "cream")},
    "fukui": {"positive": ("#008080", "blue"),   "negative": ("#FFEBDC", "cream")},
    "NTO":   {"positive": ("#5B6A8F", "blue-grey"), "negative": ("#FFEBDC", "cream")},
}


def hex_to_rgb(value):
    """(r, g, b) in 0-1 of "#RRGGBB"."""
    return tuple(int(value[i:i + 2], 16) / 255.0 for i in (1, 3, 5))


def color_name(rgb):
    """Plain English name of a colour (r, g, b in 0-1) for the captions, when the YAML gives
    a colour by its value only: grey levels by lightness, otherwise the hue (HLS) with "light",
    "dark" or "greyish". Coarse on purpose: a caption needs "pink" or "dark blue", not a
    shade; output.colors.<kind>.<sign>_name gives an exact name."""
    import colorsys
    hue, light, sat = colorsys.rgb_to_hls(*rgb)
    if max(rgb) - min(rgb) < 0.08:
        return next(name for limit, name in ((0.15, "black"), (0.3, "dark grey"), (0.6, "grey"),
                                             (0.85, "light grey"), (1.01, "white")) if light < limit)
    base = next(name for limit, name in _HUES if hue * 360 < limit)
    if base == "red" and light > 0.7:
        base = "pink"
    elif base == "pink" and light < 0.45:
        base = "purple"
    if base in ("orange", "yellow") and light > 0.9:
        return "cream"
    shade = "light " if light > 0.75 else "dark " if light < 0.3 else ""
    return shade + ("greyish " if sat < 0.3 else "") + base


# Upper limits of the hues (degrees) of color_name
_HUES = ((15, "red"), (40, "orange"), (65, "yellow"), (160, "green"), (200, "teal"),
         (250, "blue"), (290, "purple"), (345, "pink"), (361, "red"))


# Current colours (r, g, b in 0-1) and caption names: kind -> sign -> value
surface_colors = {}
surface_color_names = {}


def set_surface_colors(user=None):
    """Colours of the iso-surfaces: the defaults, replaced by the ones of the YAML.

    user : dict, optional
        kind -> sign -> ((r, g, b), name or None), as given by config Output.colors (None for
        a colour left to its default). Without a name, a colour given by a word ("teal") is
        named by it and a value by color_name.
    """
    user = user or {}
    for kind, signs in DEFAULT_SURFACE_COLORS.items():
        surface_colors[kind], surface_color_names[kind] = {}, {}
        for sign, (hexa, name) in signs.items():
            rgb, user_name = (user.get(kind) or {}).get(sign) or (None, None)
            if rgb is None:
                rgb = hex_to_rgb(hexa)
            elif user_name is None:
                name = name if tuple(rgb) == hex_to_rgb(hexa) else color_name(rgb)
            surface_colors[kind][sign] = tuple(rgb)
            surface_color_names[kind][sign] = user_name or name
    colors["CTDIP"] = surface_colors["EDD"]["positive"]


set_surface_colors()

# TODO : 
THRESHOLD_IR_INTENSITIES = 50.0
