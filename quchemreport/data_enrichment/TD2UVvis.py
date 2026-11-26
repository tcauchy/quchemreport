# Transform the Time dependent calculations into spectra with a gaussian broadening.

# This is a modified version of the GaussSum code to generate the UV Spectra
# N. M. O'Boyle, A. L. Tenderholt and K. M. Langner. J. Comp. Chem., 2008, 29, 839-845.

# This program is free software; you can redistribute and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY, without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.

"""Spectra of the excited states with a Gaussian broadening.

Adapted from GaussSum (N. M. O'Boyle, A. L. Tenderholt and K. M. Langner, J. Comput. Chem.
2008, 29, 839-845), GNU General Public License version 2 or later (see the header comments).
"""

import math 
import numpy as np

# Range of the spectra chosen in the YAML file (output.spectrum_range), in cm-1, or None
USER_RANGE_CM = None
# Grid of the broadened spectra (cm-1)
GRID_STEP = 20.0
# The spectrum extends this many FWHM beyond the first and last band (3 FWHM: the Gaussian
# is at 1e-11 of its height). Why (26/11/2026): the former rule capped the absorption grid at
# 100000 cm-1 (emission and phosphorescence at 150000), so the curve of O2 stopped at 100 nm
# while its sticks at 125000 cm-1 (80 nm, f = 0.08) were still drawn.
MARGIN_FWHM = 3.0
LOWEST = 1000.0   # cm-1 (10 micrometres): no spectrum below


def set_user_range(lo_hi_cm):
    """Range of the spectra in cm-1 (output.spectrum_range converted), or None (automatic)."""
    global USER_RANGE_CM
    USER_RANGE_CM = None if lo_hi_cm is None else (min(lo_hi_cm), max(lo_hi_cm))


def spectrum_grid(energies, FWHM):
    """Start, end (cm-1) and number of points of a spectrum of the given band energies: every
    band with MARGIN_FWHM widths on both sides, and the user range when it is wider."""
    energies = [e for e in energies if e is not None]
    start = min(energies) - MARGIN_FWHM * FWHM
    end = max(energies) + MARGIN_FWHM * FWHM
    if USER_RANGE_CM is not None:
        start, end = min(start, USER_RANGE_CM[0]), max(end, USER_RANGE_CM[1])
    start = max(LOWEST, math.floor(start / 1000.0) * 1000.0)
    end = math.ceil(end / 1000.0) * 1000.0
    return start, end, int(round((end - start) / GRID_STEP)) + 1

def GaussianSpectrum(start,end,numpts,et_energies,heights,FWHM):
# Calculate the absorption spectra with a gaussian broadening. Adapted from the GaussSum program
# et_energies is a list of excited states energies in cm-1
# et_oscs is a list of oscillator strength

    """Sum of Gaussian bands centred on the excitation energies.

    Parameters
    ----------
    start, end : float
        Range of the spectrum (cm-1).
    numpts : int
        Number of points.
    et_energies : list of float
        Energies of the bands (cm-1).
    heights : list of list of float
        One list of band heights per spectrum to compute (e.g. absorption, CD).
    FWHM : float
        Full width at half maximum of the bands (cm-1).

    Returns
    -------
    xvalues : numpy.ndarray
        Wavenumbers (cm-1).
    spectrum : numpy.ndarray
        One row per list of heights.
    """
    peaks = et_energies
    xvalues = np.arange(numpts)*float(end-start)/(numpts-1) + start
    data = []
    A = -2.7726/FWHM**2
    for x in xvalues:
        tot = [0]*len(heights) # Sum of each gaussian bands for this x value

        for peakno in range(len(peaks)): # For each peak
            pos = peaks[peakno]
            exponent = math.exp(A*(pos-x)**2)
            for spectrumno in range(len(heights)):
                tot[spectrumno] += heights[spectrumno][peakno]*exponent

        data.append(tot)

    spectrum = np.swapaxes(np.array(data),0,1)
    return xvalues, spectrum

def CDheights(et_energies, et_rotats, FWHM):
    # Equation 8 in Stephens, Harada, Chirality, 2010, 22, 229.
    # This uses Delta, the half width at 1/e height.
    # We use Sigma, the full width at 1/e height.
    #   Delta = Sigma / 2
    """Heights of the CD bands (Delta epsilon, L mol-1 cm-1) from the rotatory strengths.

    Equation 8 of Stephens and Harada, Chirality 2010, 22, 229, with the half width at 1/e
    of the Gaussian bands derived from FWHM.

    Parameters
    ----------
    et_energies : list of float
        Energies (cm-1).
    et_rotats : list of float
        Rotatory strengths (10^-40 erg.esu.cm, as printed by the programs).
    FWHM : float
        Full width at half maximum (cm-1).

    Returns
    -------
    list of float
    """
    sigma = FWHM / math.sqrt(math.log(2))
    Delta = sigma / 2.
    prefactor = 1.0 / (2.296e-39 * math.sqrt(math.pi) * Delta)
    heights = [] 
    for x_cd in range(len(et_rotats)):
        heights.append(prefactor * et_rotats[x_cd] * et_energies[x_cd] * 1e-40)
    return heights
