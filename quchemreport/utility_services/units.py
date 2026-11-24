## -*- encoding: utf-8 -*-

## Conversion factors 

"""Conversion factors and functions between units.

Names read ``<unit>_to_<unit>``: multiply by the factor to convert from the first unit to the
second (e.g. Eh_to_eV). ``a0_to_A`` and ``a0_to_pm`` are the Bohr radius in Angstrom and in
picometres.
"""

import numpy as np


def nm2wnb(x):
    # x in nanometers, returns the equivalent in wavenumber (cm-1)
    """Wavelength (nm) to wavenumber (cm-1)."""
    return 10000000.0 / x

def wnb2nm(x):
    # TODO : TO CHECK (CHECK IF CORRECT, RAISE AN ERROR IF 0 STOP THE EXEC)
    # Remplace zeros by NaN to avoid division by zero
    """Wavenumber (cm-1) to wavelength (nm); zeros give NaN instead of a division by zero
    (the secondary wavelength axis of the spectra may evaluate 0).
    """
    x = np.array(x, dtype=float)
    X = np.where(x == 0, np.nan, x)
    
    # x in wavenumber (cm-1), returns the equivalent in nanometers
    return 10000000.0 / X

## Hartree to eV conversion factor
Eh_to_eV = 27.21138505

## Hartree to cm-1 conversion factor
Eh_to_cm_1 = 219474.6313702

## Hartree to Kcal.mol^-1 conversion factor
Eh_to_Kcal_mol = 627.510

## A.U. to Debye conversion factor
ea0_to_D = 0.393430307

## eA (electron*Angstrom) to Debye conversion factor
eA_to_D = 0.2081943

## Bohr radius in Angstrom: multiply a length in bohr by a0_to_A to get Angstrom, divide a length
## in Angstrom by it to get bohr. (Named A_to_a0 until 24/11/2026, a misleading name: all its
## uses already converted in this direction.)
a0_to_A = 0.52917721092

## Bohr radius in picometres: multiply a length in bohr by a0_to_pm to get pm (formerly pm_to_a0)
a0_to_pm = a0_to_A*100

## Nanometers to wavenumber conversion
nm_to_wnb = 10000000.0


