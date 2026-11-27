"""Selection of the relevant excited states of a TD calculation (``selection.mode: relevant``).

The report shows density differences, NTO, transition densities and dipoles for the selected
states only; drawing them all would make reports of hundreds of pages for 40 or more states.
The selection is the union of:

- the LOWEST_PER_MULTIPLICITY lowest states of each multiplicity (S1-S3, T1-T3): the
  photochemistry starts there (Kasha), even when they are dark;
- the states that make the bands of the broadened spectra: for each maximum of the absorption
  curve, and separately of the positive and of the negative CD bands, at least ``band_share`` of
  the largest maximum of the same curve (the largest positive one for the positive CD bands,
  the largest negative one for the negative bands, so that a weak negative band is not hidden by
  a strong positive one), the states that give together at least BAND_PART of the curve at the
  maximum, largest contribution first; maxima closer than one FWHM to the last computed state
  are left out, since the spectrum is cut there;
- with spin-orbit couplings (ORCA, parser_engine.orca_soc), T1 and the triplets within
  SOC_WINDOW_EV of S1 that couple to it by at least SOC_MIN_CM (intersystem crossing);
- the states asked for by the user (``selection.states``).

At most ``max_selected_states`` states are kept: user states first, then the lowest ones, the
SOC triplets, and the band states from the strongest band. Each state keeps the reasons of its
selection, written under the table of the report.

Why (decisions of Thomas Cauchy, 27/11/2026; calibration validation/calculs_20261127/selection on
[6]helicene (40 singlets, 20 triplets), a push-pull dye (30 singlets), acetone and CHFClBr):
the former rule (first five states, plus f > 0.1 or |R| > 10) missed the bands made of several
weak states and kept every state above a fixed strength, however many; with the rule above,
7 to 11 states are kept for these molecules (helicene: T1-T3, S1-S3, S6, S15, S16, S22, S30);
a share of 5, 10 or 20 % changes the selection by one state at most (20 % by default, Thomas
Cauchy); a cap of 10 states was too small once absorption, CD, singlets and triplets add up,
hence 15. The FWHM is that of the spectra of the report (output.spectrum_fwhm), so that the
bands selected are those seen.
"""
import re

import numpy as np

from quchemreport.utility_services.naming import spin_letter, state_label

LOWEST_PER_MULTIPLICITY = 3
# Share of the curve at a maximum given by the states kept for that band
BAND_PART = 0.5
# Intersystem crossing: triplets within this energy of S1 (eV) coupled to it by at least this
# spin-orbit matrix element (cm-1). Not calibrated: typical n-pi* / pi-pi* couplings of organic
# molecules are 1 to 60 cm-1, and 0.3 eV is about the usual energy window of El-Sayed's rules.
SOC_WINDOW_EV = 0.3
SOC_MIN_CM = 1.0
EV_TO_CM = 8065.54
# Grid of the curves used to find the bands (cm-1)
GRID_STEP = 10.0
# Rotatory strengths below this (10^-40 cgs) are numerical noise (achiral molecule)
ROT_NOISE = 1.0e-3


def _bands(energies, heights, fwhm, share, last):
    """Bands of the curve sum_k h_k g(E - E_k) (Gaussian, FWHM fwhm), positive and negative
    separately: list of (sign, position cm-1, relative height, [(state index, contribution)])."""
    energies = np.asarray(energies, dtype=float)
    heights = np.asarray(heights, dtype=float)
    x = np.arange(energies.min() - 3 * fwhm, energies.max() + 3 * fwhm, GRID_STEP)
    a = 4 * np.log(2) / fwhm ** 2
    contrib = heights[:, None] * np.exp(-a * (x[None] - energies[:, None]) ** 2)
    curve = contrib.sum(axis=0)
    bands = []
    for sign in (1, -1):
        c = sign * curve
        peaks = [i for i in range(1, len(x) - 1) if c[i] >= c[i - 1] and c[i] >= c[i + 1] and c[i] > 0]
        if not peaks:
            continue
        top = max(c[i] for i in peaks)
        for i in peaks:
            if c[i] < share * top or x[i] > last - fwhm:
                continue
            part = sign * contrib[:, i]
            states, total = [], 0.0
            for k in np.argsort(-part):
                if part[k] <= 0 or total >= BAND_PART * c[i]:
                    break
                total += part[k]
                states.append((int(k), float(part[k])))
            bands.append((sign, float(x[i]), float(c[i] / top), states))
    return bands


def _index_of(label, symmetries):
    """Index of the state named label (S4, T2...; or its number in the log, 4) or None."""
    label = str(label).strip()
    if re.fullmatch(r"\d+", label):
        k = int(label) - 1
        return k if 0 <= k < len(symmetries) else None
    for k in range(len(symmetries)):
        if state_label(symmetries, k).upper() == label.upper():
            return k
    return None


def relevant_states(energies, symmetries, oscs=None, rots=None, soc=None, fwhm=3000.0,
                    band_share=0.20, max_selected=15, forced=()):
    """Relevant excited states of a TD calculation and the reasons of their selection.

    Parameters
    ----------
    energies : list of float
        Excitation energies (cm-1), in the order of the log.
    symmetries : list of str
        Symmetry strings of the states (cclib etsyms), for their multiplicity.
    oscs, rots : list of float, optional
        Oscillator strengths and rotatory strengths (10^-40 cgs); None when not printed.
    soc : dict, optional
        parser_engine.orca_soc.read_orca_soc result of the same calculation.
    fwhm : float
        Full width at half maximum of the bands (cm-1), that of the spectra.
    band_share : float
        Smallest maximum of a band, as a share of the largest maximum of the same curve.
    max_selected : int
        Largest number of states kept.
    forced : iterable of str
        States asked for by the user (S4, T2, or a number in the log).

    Returns
    -------
    dict
        {state index (0-based): [reasons]}, in the order of the log; the states left out by
        the cap are given by dropped_by_cap.
    """
    n = len(energies)
    if n == 0:
        return {}
    candidates = []        # (priority, rank, index, reason), the smallest first

    for label in forced:
        k = _index_of(label, symmetries)
        if k is not None:
            candidates.append((0, 0.0, k, "asked for"))

    letters = [spin_letter(s) for s in symmetries]
    for letter in sorted(set(letters)):
        group = sorted((k for k in range(n) if letters[k] == letter), key=lambda k: energies[k])
        for rank, k in enumerate(group[:LOWEST_PER_MULTIPLICITY]):
            candidates.append((1, float(rank), k, "lowest"))

    if soc:
        t1 = _index_of("T1", symmetries)
        if t1 is not None:
            candidates.append((2, 0.0, t1, "T1 (spin-orbit)"))
        s1 = _index_of("S1", symmetries)
        if s1 is not None:
            for (t, s), value in sorted((soc.get("socme") or {}).items()):
                k = _index_of(f"T{t}", symmetries)
                if s != 1 or k is None or value < SOC_MIN_CM:
                    continue
                if abs(energies[k] - energies[s1]) <= SOC_WINDOW_EV * EV_TO_CM:
                    candidates.append((2, -value, k, f"couples to S1 ({value:.0f} cm-1)"))

    # Bands, up to the last state of the multiplicities that give them
    def spectrum(values, what):
        if values is None or len(values) != n:
            return
        values = [float(v) if isinstance(v, (int, float)) else 0.0 for v in values]
        bright = [k for k in range(n) if abs(values[k]) > (ROT_NOISE if what == "CD" else 0.0)]
        if not bright:
            return
        last = max(energies[k] for k in range(n) if letters[k] in {letters[b] for b in bright})
        for sign, x, height, states in _bands(energies, values, fwhm, band_share, last):
            name = what if what == "absorption" else f"CD {'+' if sign > 0 else '-'}"
            for k, _ in states:
                candidates.append((3, -height, k, f"{name} band {1e7 / x:.0f} nm"))

    spectrum(oscs, "absorption")
    spectrum(rots, "CD")

    reasons, order = {}, []
    for priority, rank, k, reason in sorted(candidates, key=lambda c: (c[0], c[1])):
        if k not in reasons:
            order.append(k)
            reasons[k] = []
        if reason not in reasons[k]:
            reasons[k].append(reason)
    kept = set(order[:max_selected])
    return {k: reasons[k] for k in sorted(kept)}


def dropped_by_cap(energies, symmetries, kept, **kwargs):
    """States that the rule would select but that the cap left out (indices)."""
    every = relevant_states(energies, symmetries, max_selected=len(energies), **kwargs)
    return sorted(set(every) - set(kept))
