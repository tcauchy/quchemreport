## -*- encoding: utf-8 -*-

"""Default parameters of the computations and pictures.

obk_step is the step of the grids (bohr) and obk_extand their margin around the atoms; FWHM the
width of the Gaussian bands of the spectra (cm-1); img_width / img_height the default size of
the 3D pictures (output.image_size overrides it); colors and scale the colours of the
iso-surfaces and the scale factors of the dipole arrows. The camera views are chosen from the
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

colors = {"OM" : (0.4, 0, 0.235),  #For OM positive surface Tyrian purple
          "EDD" : (0.0, 0.5, 0.5),  #For Electron density difference positive surface metallic blue
          "Oif" : (0.95, 0.5, 0.0),  #For transitions wavefunctions overlap positive surface an orange
          "NEG" : (0.95, 0.90, 0.93),  #For isosurfaces negative surfaces almost white
          "CTDIP" : (0.0, 0.5, 0.5), #For transitions charge transfer dipole same as EDD surface
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

# TODO : 
THRESHOLD_IR_INTENSITIES = 50.0
