"""Pictures of the spectra (matplotlib): broadened spectrum on top, sticks of the transitions
below. The main axis is the wavenumber (cm-1), the energy (eV), both linear in energy and
decreasing to the right, or the wavelength (nm, the habit of experimental chemists), chosen with
``output.spectrum_axis``; a second unit can be shown on the top axis
(``output.spectrum_secondary_axis``).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from quchemreport.data_enrichment import TD2UVvis

# eV per cm-1 (CODATA 2018: 1 eV = 8065.543937 cm-1)
EV_PER_CM = 1.0 / 8065.543937


def _reciprocal(x):
    """nm <-> cm-1 (1e7 / x, the same in both directions); NaN for 0, which matplotlib may
    evaluate for the secondary axis."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(x != 0, 1.0e7 / np.where(x != 0, x, 1.0), np.nan)


# Units of the spectrum axes: axis label, label on the top axis, conversion from and to cm-1,
# whether the values decrease with the energy (then the axis is not inverted), and the
# spacings of the top ticks (roundest first; the finer ones only when too few ticks were found)
UNITS = {
    "wavenumber": {"label": r"Wavenumber / cm$^{-1}$", "top": r"$\tilde{\nu}$ / cm$^{-1}$",
                   "from_cm": lambda x: np.asarray(x, dtype=float), "to_cm": lambda x: np.asarray(x, dtype=float),
                   "reciprocal": False, "steps": [10000, 5000, 2000, 1000], "finer": [500, 200, 100]},
    "energy": {"label": r"Energy / eV", "top": r"$E$ / eV",
               "from_cm": lambda x: np.asarray(x, dtype=float) * EV_PER_CM,
               "to_cm": lambda x: np.asarray(x, dtype=float) / EV_PER_CM,
               "reciprocal": False, "steps": [1.0, 0.5, 0.2, 0.1], "finer": [0.05, 0.02, 0.01]},
    "wavelength": {"label": r"Wavelength / nm", "top": r"$\lambda$ / nm",
                   "from_cm": _reciprocal, "to_cm": _reciprocal,
                   "reciprocal": True, "steps": [100, 50, 20, 10], "finer": [5, 2, 1]},
}
# Names accepted in the YAML file for each unit
UNIT_ALIASES = {"wavenumber": "wavenumber", "cm-1": "wavenumber", "cm^-1": "wavenumber",
                "energy": "energy", "ev": "energy", "wavelength": "wavelength", "nm": "wavelength"}

# Main axis of the spectra and unit of the top axis (None: no top axis); set_spectrum_axis
SPECTRUM_AXIS = "wavenumber"
SECONDARY_AXIS = "wavelength"
# Smallest distance between two labels of the top axis, per character of the longer label, as
# a share of the axis length: 0.09 for the 3-digit nm labels, 0.15 for the 5-digit cm-1
# labels (a fixed 0.09 still let "80000" and "70000" touch on a wavelength axis)
TOP_TICK_GAP_PER_CHAR = 0.03
# Displayed range of the spectra (without output.spectrum_range), computed in energy: on the
# high-energy side, up to the last band that can be seen (curve above VISIBLE_SHARE of its
# largest value, or stick above VISIBLE_SHARE of the largest one); on the low-energy side, down
# to the lowest computed state, even a dark one; widened by VIEW_MARGIN of that span on each
# side, within the computed grid.
# Why (27/11/2026, Thomas Cauchy): the wavelength axis was already cut to what is visible, but
# the wavenumber axis showed the whole grid (3 FWHM around every band) plus the 5 % margins of
# matplotlib: a single band filled 42 % of the axis (80 % in nm), and for a band below about
# 10000 cm-1 the margin put the axis below zero, where the top axis (1e7 / x) asked for 1e14
# tick candidates (MemoryError: no report for a near-infrared dye). A dark state of low energy
# must stay on the axis to show that it was computed; a dark state above the visible bands is
# left out, as the states above the last computed one, which could cover it, are unknown.
VISIBLE_SHARE = 0.005
VIEW_MARGIN = 0.10
MIN_TOP_TICKS = 3
# Largest number of candidates nice_ticks tries for one spacing
MAX_TICK_CANDIDATES = 10000
# Dark states (stick below VISIBLE_SHARE of the largest one): open circles on the base line
DARK_MARKER = dict(ls="none", marker="o", ms=4, mfc="white", mec="black", mew=0.8, zorder=3, clip_on=False)


def unit_name(name):
    """Unit of a spectrum axis from its name in the YAML file (wavenumber or cm-1, energy or
    eV, wavelength or nm); ValueError otherwise."""
    key = str(name).strip().lower()
    if key not in UNIT_ALIASES:
        raise ValueError(f"spectrum axis must be wavenumber (cm-1), energy (eV) or wavelength (nm), not {name!r}")
    return UNIT_ALIASES[key]


def set_spectrum_axis(axis, secondary="auto"):
    """Main axis of the spectra (wavenumber, energy or wavelength, or cm-1, eV, nm) and unit of
    the top axis: "auto" (nm, or cm-1 when the main axis is in nm), a unit, or None / "none"."""
    global SPECTRUM_AXIS, SECONDARY_AXIS
    main = unit_name(axis)
    if secondary is None or str(secondary).lower() == "none":
        top = None
    elif str(secondary).lower() == "auto":
        top = "wavenumber" if main == "wavelength" else "wavelength"
    else:
        top = unit_name(secondary)
        if top == main:
            raise ValueError(f"the secondary axis of the spectra must differ from the main one ({main})")
    SPECTRUM_AXIS, SECONDARY_AXIS = main, top


def to_cm(unit, values):
    """Values in the given unit converted to cm-1 (spectrum_range)."""
    return UNITS[unit_name(unit)]["to_cm"](values)


def nice_ticks(lo, hi, to_main, steps, gap_per_char):
    """Round values of the secondary unit between lo and hi, at least gap apart on the main axis.

    The secondary unit is not linear on the main axis (nm on a cm-1 axis): evenly spaced
    values crowd at one end, where the labels overlapped (300 350 400 ... 650 nm squeezed at
    the right of the absorption spectra). The roundest values are taken first (multiples of
    steps[0], then steps[1]...) and a value is kept only if it is far enough from the kept ones.

    Parameters
    ----------
    lo, hi : float
        Range in the secondary unit.
    to_main : callable
        Secondary unit to main axis unit.
    steps : list of float
        Spacings of the candidates, roundest first.
    gap_per_char : float
        Smallest distance on the main axis, per character of the longer of two labels.
    """
    width = lambda value: len(f"{value:g}")
    kept = []
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return kept
    for step in steps:
        if (hi - lo) / step > MAX_TICK_CANDIDATES:
            continue
        for value in np.arange(np.ceil(lo / step) * step, hi + step / 2, step):
            if lo <= value <= hi and all(abs(to_main(value) - to_main(k)) >= gap_per_char * max(width(value), width(k))
                                         for k in kept):
                kept.append(float(value))
    return sorted(kept)


def view_range_cm(wavenumbers, y, sticks, heights):
    """Range (cm-1) to show: from the lowest computed state (even a dark one) to the highest
    visible band (VISIBLE_SHARE of the largest curve value or stick), see VISIBLE_SHARE."""
    x, y = np.asarray(wavenumbers, dtype=float), np.abs(np.asarray(y, dtype=float))
    sticks, heights = np.asarray(sticks, dtype=float), np.abs(np.asarray(heights, dtype=float))
    finite = np.isfinite(sticks)
    visible = [x[y > VISIBLE_SHARE * y.max()]] if y.size and y.max() > 0 else []
    if heights.size and heights.max() > 0:
        visible.append(sticks[finite & (heights > VISIBLE_SHARE * heights.max())])
    visible = np.concatenate(visible) if visible else np.array([])
    if not visible.size:                     # nothing visible: every state and the whole curve
        visible = np.concatenate([x, sticks[finite]])
    low = np.concatenate([visible, sticks[finite]])
    return low.min(), visible.max()


def main_axis_limits(wavenumbers, lo_cm, hi_cm, unit):
    """Limits (main unit, increasing) of the range lo_cm-hi_cm widened by VIEW_MARGIN of its
    span, within the computed grid (never at or below zero)."""
    conv = UNITS[unit]["from_cm"]
    a, b = sorted(conv(np.array([lo_cm, hi_cm])))
    grid = np.asarray(wavenumbers, dtype=float)
    g0, g1 = sorted(conv(np.array([grid[grid > 0].min(), grid.max()])))
    margin = VIEW_MARGIN * (b - a) or 0.05 * b
    return max(a - margin, g0), min(b + margin, g1)


def _fit_ylim(axis, values, signed):
    """Vertical range of axis fitted to values (5 % margin; from 0 unless signed)."""
    top, bottom = max(values.max(), 0.0), min(values.min(), 0.0)
    pad = 0.05 * (top - bottom) or 1.0
    axis.set_ylim(bottom - pad if (signed and bottom < 0) else 0.0, top + pad)


def spectrum_figure(wavenumbers, curve, stick_positions, stick_heights, ylabel, stick_label,
                    file_name, signed=False, sci_sticks=False):
    """Draw and save a spectrum (curve above, sticks below) on the chosen main axis, with the
    chosen unit on the top axis; dark states are open circles on the base line of the sticks.

    Parameters
    ----------
    wavenumbers : numpy.ndarray
        Abscissa of the broadened curve (cm-1); the broadening is computed in energy.
    curve : numpy.ndarray
    stick_positions : list of float
        Transition energies (cm-1).
    stick_heights : list of float
    ylabel, stick_label : str
        Labels of the curve and of the sticks.
    file_name : str or None
        Full name of the picture.
    signed : bool, optional
        Dashed zero line for signed spectra (CD, CPL).
    sci_sticks : bool, optional
        Scientific notation for the sticks (phosphorescence: f about 1e-6).
    """
    plt.close("all")
    fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    unit = UNITS[SPECTRUM_AXIS]
    wavenumbers = np.asarray(wavenumbers, dtype=float)
    keep = wavenumbers > 0
    wavenumbers = wavenumbers[keep]
    y = np.asarray(curve, dtype=float)[keep]
    sticks = np.asarray(stick_positions, dtype=float)
    heights = np.asarray(stick_heights, dtype=float)
    x, sx = unit["from_cm"](wavenumbers), unit["from_cm"](sticks)
    ax[0].plot(x, y)
    if unit["reciprocal"]:
        # lines, not bars: a bar width in nm would not be the same in energy across the axis
        ax[1].vlines(sx, 0, heights, linewidth=2)
    else:
        ax[1].bar(sx, heights, width=float(unit["from_cm"](250.0)))
    dark = np.abs(heights) <= VISIBLE_SHARE * np.abs(heights).max() if heights.size and np.abs(heights).max() > 0 \
        else np.ones(heights.shape, dtype=bool)
    if signed:
        ax[0].axhline(y=0, color='black', ls='--')
    ax[0].set(ylabel=ylabel)
    ax[1].axhline(y=0, color='black')
    ax[1].set(ylabel=stick_label)
    if sci_sticks:
        ax[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    user = TD2UVvis.USER_RANGE_CM
    if user is not None:
        # output.spectrum_range: the axis shows that range; the vertical scales follow what is
        # inside it
        lo, hi = sorted(unit["from_cm"](np.array(user, dtype=float)))
        ax[1].set_xlim(lo, hi)
        inside = (x >= lo) & (x <= hi)
        if inside.any():
            _fit_ylim(ax[0], y[inside], signed)
        in_sticks = (sx >= lo) & (sx <= hi)
        if in_sticks.any():
            _fit_ylim(ax[1], heights[in_sticks], True)
    elif wavenumbers.size:
        ax[1].set_xlim(*main_axis_limits(wavenumbers, *view_range_cm(wavenumbers, y, sticks, heights),
                                         SPECTRUM_AXIS))
    # Dark states inside the axis only: the markers are not clipped (a circle on the base line
    # would be cut in half), and one far outside made tight_layout squeeze the whole figure
    lo, hi = sorted(ax[1].get_xlim())
    shown = dark & np.isfinite(sx) & (sx >= lo) & (sx <= hi)
    if shown.any():
        ax[1].plot(sx[shown], np.zeros(int(shown.sum())), **DARK_MARKER)
        ax[1].set_xlim(lo, hi)
    if not unit["reciprocal"]:
        ax[1].invert_xaxis()                 # energy decreasing to the right, as the wavelength increases
    ax[1].set_xlabel(unit["label"])
    for a in ax:
        a.minorticks_on()
    if SECONDARY_AXIS is not None:
        top = UNITS[SECONDARY_AXIS]
        to_top = lambda v: top["from_cm"](unit["to_cm"](v))
        to_main = lambda v: unit["from_cm"](top["to_cm"](v))
        left, right = sorted(ax[0].get_xlim())
        secax = ax[0].secondary_xaxis('top', functions=(to_top, to_main))
        secax.set_xlabel(top["top"])
        top_values = to_top(np.array([max(left, 1e-9), right]))
        gap = TOP_TICK_GAP_PER_CHAR * (right - left)
        lo_top, hi_top = np.nanmin(top_values), np.nanmax(top_values)
        ticks = nice_ticks(lo_top, hi_top, to_main, top["steps"], gap)
        if len(ticks) < MIN_TOP_TICKS:
            ticks = nice_ticks(lo_top, hi_top, to_main, top["steps"] + top["finer"], gap)
        secax.xaxis.set_major_locator(FixedLocator(ticks))
        # the labels as nice_ticks measured them ("7", "7.5"; matplotlib wrote "7.0" for the eV
        # ticks, wider than measured, and two of them touched)
        secax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        secax.xaxis.set_minor_locator(NullLocator())
    fig.set_size_inches(6, 6)
    # the label of the sticks was cut at the left edge
    fig.tight_layout()
    if file_name is not None:
        fig.savefig(file_name, dpi=300)
    plt.close(fig)


def absoUV(et_energies, et_oscs, xvalues, spectrum, file_name=None):
    """Absorption spectrum: epsilon and oscillator strengths; saved as
    ``<file_name>-UV-Abso-Spectrum.png``.

    Parameters
    ----------
    et_energies : list of float
        Transition energies (cm-1).
    et_oscs : list of float
        Oscillator strengths.
    xvalues : numpy.ndarray
        Wavenumbers of the spectrum (cm-1).
    spectrum : numpy.ndarray
        Broadened spectrum (TD2UVvis.GaussianSpectrum).
    file_name : str
        Path prefix of the picture.
    """
    spectrum_figure(xvalues, spectrum[0], et_energies, et_oscs,
                    r'$\epsilon$ / L mol$^{-1}$ cm$^{-1}$', 'Oscillator strength',
                    None if file_name is None else "{}-UV-Abso-Spectrum.png".format(file_name))


def emiUV(et_energies, et_oscs, xvalues, spectrum, file_name=None):
    """Emission spectrum of an optimized excited state; saved as ``<file_name>-UV-Emi-Spectrum.png`` (parameters as absoUV)."""
    spectrum_figure(xvalues, spectrum[0], et_energies, et_oscs,
                    r'molar emission / L mol$^{-1}$ cm$^{-1}$', 'Oscillator strength',
                    None if file_name is None else "{}-UV-Emi-Spectrum.png".format(file_name))


def absoCD(et_energies, et_rotats, xvalues, CDspectrum, file_name=None):
    """CD spectrum: Delta epsilon and rotatory strengths; saved as ``<file_name>-UV-CD-Spectrum.png`` (parameters as absoUV, with the rotatory strengths)."""
    spectrum_figure(xvalues, CDspectrum[0], et_energies, et_rotats,
                    r'$\Delta \epsilon$ / L mol$^{-1}$ cm$^{-1}$', 'Rotatory strength',
                    None if file_name is None else "{}-UV-CD-Spectrum.png".format(file_name), signed=True)


def emiCD(et_energies, et_rotats, xvalues, CDspectrum, file_name=None):
    """CPL spectrum of an optimized excited state; saved as ``<file_name>-UV-CD-Emi-Spectrum.png``."""
    # the former label '$\Delta molar emission...' had an unclosed $ (raw mathtext text)
    spectrum_figure(xvalues, CDspectrum[0], et_energies, et_rotats,
                    r'$\Delta$ molar emission / L mol$^{-1}$ cm$^{-1}$', 'Rotatory strength',
                    None if file_name is None else "{}-UV-CD-Emi-Spectrum.png".format(file_name), signed=True)





            # Calculating of IR spectrum
            #GaussSum_IRspectrum.Vibfreq(jf[i],os.path.basename(lf[i]),600,4000,1700,6.0,"Gen",1,0,jf[i]["comp_details"]["freq"]["temperature"])
        # Plotting IR spectrum 
        #extracts the datas from the resulting txt file to generate a png file of the spectrum
#            x,y = np.loadtxt("IRSpectrum.txt",skiprows=2,usecols=(0,1), unpack=True)
#            plt.xlabel('Wavenumber (cm-1)')
#            plt.ylabel('IR activity')
#            plt.title("Calculated Infrared Spectrum")
#            plt.gca().invert_yaxis()
#            plt.gca().invert_xaxis()
#            #plt.plot(x,y, 'o', color='black',linestyle="-",markersize= 2 )
#            plt.plot(x,y, color='blue',linestyle="-" )
#            fig = plt.gcf()
#            fig.set_size_inches(11.5, 6.5)
#            plt.savefig("img-IR_Spectrum.png", dpi=300)
            # plt.show()


# Conversion cm-1 -> eV for the energy levels of the Jablonski-Perrin diagram
CM_TO_EV = 1.0 / 8065.544
# Triplets within this energy (eV) above S1 are shown as reachable by intersystem crossing
ISC_WINDOW = 0.3


# Smallest vertical distance between two labels of the Jablonski diagram, as a fraction of the
# energy axis (a 9-point label is about 3 % of the 6-inch figure)
LABEL_GAP = 0.035


def spread_labels(positions, gap):
    """Positions moved apart by at least gap, in the same order: colliding labels form a group
    spaced by gap and centred on their mean; isolated labels do not move."""
    order = sorted(range(len(positions)), key=lambda i: positions[i])
    groups = []                       # lists of indexes, bottom to top
    for i in order:
        groups.append([i])
        while len(groups) > 1:
            low, high = groups[-2], groups[-1]
            low_top = _group_places(low, positions, gap)[-1]
            if _group_places(high, positions, gap)[0] - low_top >= gap - 1e-12:
                break
            groups[-2:] = [low + high]
    out = [0.0] * len(positions)
    for g in groups:
        for i, y in zip(g, _group_places(g, positions, gap)):
            out[i] = y
    return out


def _group_places(group, positions, gap):
    """Places of a group of labels: spaced by gap, centred on the mean of their positions."""
    centre = sum(positions[i] for i in group) / len(group)
    return [centre + gap * (k - (len(group) - 1) / 2) for k in range(len(group))]


def jablonski(soc, file_name=None, max_singlets=3, max_triplets=6):
    """Jablonski-Perrin diagram of the vertical states with their spin-orbit couplings.

    Singlets (left) and triplets (right) at the geometry of the calculation, in eV; dashed
    arrows give the intersystem crossings S1 -> Tn (triplets below S1 + ISC_WINDOW) and
    T1 -> S0, labelled with |<T|H_SO|S>| (cm-1). Saved as ``<file_name>-Jablonski.png``.

    Parameters
    ----------
    soc : dict
        SOC results (parser_engine.orca_soc.read_orca_soc).
    file_name : str, optional
        Path prefix of the picture.
    max_singlets, max_triplets : int, optional
        Number of excited singlets and triplets drawn (at most).
    """
    singlets = [("S0", 0.0)] + [(f"S{s['number']}", s["energy"] * CM_TO_EV) for s in soc["singlets"][:max_singlets]]
    top = max(e for _, e in singlets) + 0.2
    triplets = [(f"T{t['number']}", t["energy"] * CM_TO_EV) for t in soc["triplets"] if t["energy"] * CM_TO_EV <= top]
    triplets = triplets[:max_triplets]
    socme = soc["socme"]
    plt.close("all")
    fig, ax = plt.subplots(figsize=(6.5, 6))
    xs, xt = (0.0, 1.0), (2.0, 3.0)
    # Labels of close levels (S1/S2 at 6.28/6.38 eV, T1/T2 at 5.41/5.57 eV for HCFClBr) and of
    # the couplings of close triplets were printed on top of each other (26/11/2026): they are
    # spread apart (spread_labels), with a thin line to the level when moved.
    gap = LABEL_GAP * (top + 0.7)
    for levels, x_line, x_text, ha, color in ((singlets, xs, xs[0] - 0.05, "right", "tab:blue"),
                                             (triplets, xt, xt[1] + 0.05, "left", "tab:red")):
        ys = spread_labels([e for _, e in levels], gap)
        for (name, e), y in zip(levels, ys):
            ax.hlines(e, *x_line, color=color, lw=2)
            if abs(y - e) > 1e-6:
                edge = x_line[0] if ha == "right" else x_line[1]
                ax.plot([edge, x_text], [e, y], color=color, lw=0.6)
            ax.text(x_text, y, f"{name}  {e:.2f} eV", ha=ha, va="center", fontsize=9, color=color)

    arrows = []

    def coupling(t_name, s_name, start, end):
        value = socme.get((int(t_name[1:]), int(s_name[1:])))
        if value is not None:
            arrows.append((start, end, value))

    if len(singlets) > 1:
        e_s1 = singlets[1][1]
        for name, e in triplets:
            if e <= e_s1 + ISC_WINDOW:
                coupling(name, "S1", (xs[1], e_s1), (xt[0], e))
    if triplets:
        coupling(triplets[0][0], "S0", (xt[0], triplets[0][1]), (xs[1], 0.0))
    ys = spread_labels([(a[1] + b[1]) / 2 for a, b, _ in arrows], gap)
    for (start, end, value), y in zip(arrows, ys):
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="->", linestyle="--", color="0.3", lw=1.2))
        ax.text((start[0] + end[0]) / 2, y, f"{value:.1f}", fontsize=8, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
    ax.set_xlim(-1.4, 4.4)
    ax.set_ylim(-0.4, top + 0.3)
    ax.set_xticks([0.5, 2.5])
    ax.set_xticklabels(["Singlets", "Triplets"])
    ax.set_ylabel("Energy / eV")
    ax.set_title(r"Spin-orbit couplings $|\langle T|H_{SO}|S\rangle|$ / cm$^{-1}$", fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    if file_name is not None:
        fig.savefig("{}-Jablonski.png".format(file_name), dpi=300)
    plt.close(fig)


def phospho(energies, foscs, xvalues, spectrum, file_name=None):
    """Phosphorescence spectrum of the sublevels of a triplet (``<file_name>-Phospho-Spectrum.png``).

    Parameters as emiUV: the sticks are the sublevels with their oscillator strengths.
    """
    spectrum_figure(xvalues, spectrum[0], energies, foscs,
                    r'molar emission / L mol$^{-1}$ cm$^{-1}$', 'Oscillator strength',
                    None if file_name is None else "{}-Phospho-Spectrum.png".format(file_name), sci_sticks=True)
