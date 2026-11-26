"""Pictures of the spectra (matplotlib): broadened spectrum on top, sticks of the transitions
below. The main axis is the wavenumber (cm-1, decreasing, linear in energy) or the wavelength
(nm, the habit of experimental chemists), chosen with ``output.spectrum_axis``; the other unit
is on the top axis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

from quchemreport.data_enrichment import TD2UVvis

# Main axis of the spectra: "wavenumber" (default) or "wavelength" (set_spectrum_axis)
SPECTRUM_AXIS = "wavenumber"
# Smallest distance between two labels of the top axis, per character of the longer label, as
# a share of the axis length: 0.09 for the 3-digit nm labels, 0.15 for the 5-digit cm-1
# labels (a fixed 0.09 still let "80000" and "70000" touch on a wavelength axis)
TOP_TICK_GAP_PER_CHAR = 0.03


def _reciprocal(x):
    """nm <-> cm-1 (1e7 / x, the same in both directions); NaN for 0, which matplotlib may
    evaluate for the secondary axis."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(x != 0, 1.0e7 / np.where(x != 0, x, 1.0), np.nan)


def set_spectrum_axis(axis):
    """Main axis of the spectra: "wavenumber" (cm-1) or "wavelength" (nm)."""
    global SPECTRUM_AXIS
    if axis not in ("wavenumber", "wavelength"):
        raise ValueError(f"spectrum axis must be wavenumber or wavelength, not {axis!r}")
    SPECTRUM_AXIS = axis


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
    for step in steps:
        for value in np.arange(np.ceil(lo / step) * step, hi + step / 2, step):
            if lo <= value <= hi and all(abs(to_main(value) - to_main(k)) >= gap_per_char * max(width(value), width(k))
                                         for k in kept):
                kept.append(float(value))
    return sorted(kept)


def _fit_ylim(axis, values, signed):
    """Vertical range of axis fitted to values (5 % margin; from 0 unless signed)."""
    top, bottom = max(values.max(), 0.0), min(values.min(), 0.0)
    pad = 0.05 * (top - bottom) or 1.0
    axis.set_ylim(bottom - pad if (signed and bottom < 0) else 0.0, top + pad)


def spectrum_figure(wavenumbers, curve, stick_positions, stick_heights, ylabel, stick_label,
                    file_name, signed=False, sci_sticks=False):
    """Draw and save a spectrum (curve above, sticks below) on the chosen main axis.

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
    wavenumbers = np.asarray(wavenumbers, dtype=float)
    sticks = np.asarray(stick_positions, dtype=float)
    if SPECTRUM_AXIS == "wavelength":
        keep = wavenumbers > 0
        x, sx = _reciprocal(wavenumbers[keep]), _reciprocal(sticks)
        ax[0].plot(x, np.asarray(curve)[keep])
        # lines, not bars: a bar width in nm would not be the same in energy across the axis
        ax[1].vlines(sx, 0, stick_heights, linewidth=2)
        # The broadening grid reaches far in the red (1000 nm for an absorption at 140 nm):
        # the axis is limited to where the curve or a stick is visible, with a margin
        y = np.abs(np.asarray(curve, dtype=float)[keep])
        seen = x[y > 0.005 * y.max()] if y.size and y.max() > 0 else x
        span = np.concatenate([seen, sx[np.isfinite(sx)]])
        if span.size:
            lo, hi = span.min(), span.max()
            margin = 0.1 * (hi - lo) + 5
            ax[1].set_xlim(max(lo - margin, x.min()), min(hi + margin, x.max()))
        main_label, top_label = r'Wavelength / nm', r'$\tilde{\nu}$ / cm$^{-1}$'
        to_top = to_main = _reciprocal
        steps = [10000, 5000, 2000, 1000]
    else:
        x, sx = wavenumbers, sticks
        ax[0].plot(x, curve)
        ax[1].bar(sx, stick_heights, width=250)
        main_label, top_label = r'Wavenumber / cm$^{-1}$', r'$\lambda$ / nm'
        to_top = to_main = _reciprocal
        steps = [100, 50, 20, 10]
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
        lo, hi = (sorted(_reciprocal(np.array(user))) if SPECTRUM_AXIS == "wavelength" else user)
        ax[1].set_xlim(lo, hi)
        inside = (x >= lo) & (x <= hi)
        y = np.asarray(curve, dtype=float)[keep] if SPECTRUM_AXIS == "wavelength" else np.asarray(curve, dtype=float)
        if inside.any():
            _fit_ylim(ax[0], y[inside], signed)
        heights = np.asarray(stick_heights, dtype=float)
        in_sticks = (sx >= lo) & (sx <= hi)
        if in_sticks.any():
            _fit_ylim(ax[1], heights[in_sticks], True)
    if SPECTRUM_AXIS == "wavenumber":
        ax[1].invert_xaxis()
    ax[1].set_xlabel(main_label)
    for a in ax:
        a.minorticks_on()
    left, right = sorted(ax[0].get_xlim())
    secax = ax[0].secondary_xaxis('top', functions=(to_top, to_main))
    secax.set_xlabel(top_label)
    top_values = to_top(np.array([max(left, 1e-9), right]))
    ticks = nice_ticks(np.nanmin(top_values), np.nanmax(top_values), to_main, steps,
                       TOP_TICK_GAP_PER_CHAR * (right - left))
    secax.xaxis.set_major_locator(FixedLocator(ticks))
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
