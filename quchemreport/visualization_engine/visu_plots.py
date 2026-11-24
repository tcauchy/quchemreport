"""Pictures of the spectra (matplotlib): broadened spectrum on top, sticks of the transitions
below, wavenumber axis (cm-1, decreasing) with the wavelength (nm) on the top axis.
"""

import matplotlib.pyplot as plt
from quchemreport.utility_services import units


def absoUV(et_energies, et_oscs, xvalues, spectrum, file_name=None):
    # Plotting UV spectrum from the excited states energies and oscillator strengths, the xvalues in wavenumbers and the correspondng absorption in epsilon
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
    plt.cla()
    fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    # First plot specific lines
    x = xvalues
    y = spectrum[0]
    ax[0].plot(x,y)                                
    ax[0].set(ylabel=r'$\epsilon$ / L mol$^{-1}$ cm$^{-1}$')
    secax = ax[0].secondary_xaxis('top', functions=(units.nm2wnb, units.wnb2nm)) # Needs Matplotlib 3.1 ! experimental
    secax.set_xlabel(r'$\lambda$ / nm')
    # Second plot specific lines
    x2 = et_energies
    y2 = et_oscs
    ax[1].axhline(y=0,color='black')
    ax[1].bar(x2,y2,width=250)
    ax[1].set(ylabel='Oscillator strength')
    # Common lines for the 2 plots
    plt.gca().invert_xaxis()                        
    plt.xlabel('Wavenumber / cm$^{-1}$')
    plt.minorticks_on()
    fig = plt.gcf()
    fig.set_size_inches(6, 6)
    plt.savefig("{}-UV-Abso-Spectrum.png".format(file_name), dpi=300)
    #plt.show()                

def emiUV(et_energies, et_oscs, xvalues, spectrum, file_name=None):
    # Plotting UV spectrum from the excited states energies and oscillator strengths, the xvalues in wavenumbers and the correspondng absorption in epsilon
    """Emission spectrum of an optimized excited state; saved as ``<file_name>-UV-Emi-Spectrum.png`` (parameters as absoUV)."""
    plt.cla()
    fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    # First plot specific lines
    x = xvalues
    y = spectrum[0]
    ax[0].plot(x,y)                                
    ax[0].set(ylabel=r'molar emission / L mol$^{-1}$ cm$^{-1}$')
    secax = ax[0].secondary_xaxis('top', functions=(units.nm2wnb, units.wnb2nm)) # Needs Matplotlib 3.1 ! experimental
    secax.set_xlabel(r'$\lambda$ / nm')
    # Second plot specific lines
    x2 = et_energies
    y2 = et_oscs
    ax[1].axhline(y=0,color='black')
    ax[1].bar(x2,y2,width=250)
    ax[1].set(ylabel='Oscillator strength')
    # Common lines for the 2 plots
    plt.gca().invert_xaxis()                        
    plt.xlabel(r'Wavenumber / cm$^{-1}$')
    plt.minorticks_on()
    fig = plt.gcf()
    fig.set_size_inches(6, 6)
    plt.savefig("{}-UV-Emi-Spectrum.png".format(file_name), dpi=300)
    #plt.show()                


def absoCD(et_energies, et_rotats, xvalues, CDspectrum, file_name=None):
    # Plotting CD spectrum from the excited states energies and rotational strengths, the xvalues in wavenumbers and the correspondng absorption in epsilon
    """CD spectrum: Delta epsilon and rotatory strengths; saved as ``<file_name>-UV-CD-Spectrum.png`` (parameters as absoUV, with the rotatory strengths)."""
    plt.cla()
    fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    # First plot specific lines
    x = xvalues
    y = CDspectrum[0]
    ax[0].plot(x,y)                                
    ax[0].axhline(y=0,color='black',ls='--')
    ax[0].set(ylabel=r'$\Delta \epsilon$ / L mol$^{-1}$ cm$^{-1}$')
    secax = ax[0].secondary_xaxis('top', functions=(units.nm2wnb, units.wnb2nm)) # Needs Matplotlib 3.1 ! experimental
    secax.set_xlabel(r'$\lambda$ / nm')
    # Second plot specific lines
    x2 = et_energies
    y2 = et_rotats
    ax[1].axhline(y=0,color='black')
    ax[1].bar(x2,y2,width=250)
    ax[1].set(ylabel='Rotational strength')
    # Common lines for the 2 plots
    plt.gca().invert_xaxis()                        
    plt.xlabel(r'Wavenumber / cm$^{-1}$')
    plt.minorticks_on()
    fig = plt.gcf()
    fig.set_size_inches(6, 6)
    plt.savefig("{}-UV-CD-Spectrum.png".format(file_name), dpi=300)
    #plt.show()                

def emiCD(et_energies, et_rotats, xvalues, CDspectrum, file_name=None):
    # Plotting CD spectrum from the excited states energies and rotational strengths, the xvalues in wavenumbers and the correspondng absorption in epsilon
    """CPL spectrum of an optimized excited state; saved as ``<file_name>-UV-CD-Emi-Spectrum.png``."""
    plt.cla()
    fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    # First plot specific lines
    x = xvalues
    y = CDspectrum[0]
    ax[0].plot(x,y)                                
    ax[0].axhline(y=0,color='black',ls='--')
    ax[0].set(ylabel=r'$\Delta molar emission / L mol$^{-1}$ cm$^{-1}$')
    secax = ax[0].secondary_xaxis('top', functions=(units.nm2wnb, units.wnb2nm)) # Needs Matplotlib 3.1 ! experimental
    secax.set_xlabel(r'$\lambda$ / nm')
    # Second plot specific lines
    x2 = et_energies
    y2 = et_rotats
    ax[1].axhline(y=0,color='black')
    ax[1].bar(x2,y2,width=250)
    ax[1].set(ylabel='Rotational strength')
    # Common lines for the 2 plots
    plt.gca().invert_xaxis()                        
    plt.xlabel(r'Wavenumber / cm$^{-1}$')
    plt.minorticks_on()
    fig = plt.gcf()
    fig.set_size_inches(6, 6)
    plt.savefig("{}-UV-CD-Emi-Spectrum.png".format(file_name), dpi=300)
    #plt.show()                





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
    for name, e in singlets:
        ax.hlines(e, *xs, color="tab:blue", lw=2)
        ax.text(xs[0] - 0.05, e, f"{name}  {e:.2f} eV", ha="right", va="center", fontsize=9, color="tab:blue")
    for name, e in triplets:
        ax.hlines(e, *xt, color="tab:red", lw=2)
        ax.text(xt[1] + 0.05, e, f"{name}  {e:.2f} eV", ha="left", va="center", fontsize=9, color="tab:red")

    def coupling(t_name, s_name, start, end):
        value = socme.get((int(t_name[1:]), int(s_name[1:])))
        if value is None:
            return
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="->", linestyle="--", color="0.3", lw=1.2))
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2, f"{value:.1f}", fontsize=8,
                ha="center", va="bottom", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))

    if len(singlets) > 1:
        e_s1 = singlets[1][1]
        for name, e in triplets:
            if e <= e_s1 + ISC_WINDOW:
                coupling(name, "S1", (xs[1], e_s1), (xt[0], e))
    if triplets:
        coupling(triplets[0][0], "S0", (xt[0], triplets[0][1]), (xs[1], 0.0))
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
    plt.close("all")
    fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    ax[0].plot(xvalues, spectrum[0])
    ax[0].set(ylabel=r'molar emission / L mol$^{-1}$ cm$^{-1}$')
    secax = ax[0].secondary_xaxis('top', functions=(units.nm2wnb, units.wnb2nm))
    secax.set_xlabel(r'$\lambda$ / nm')
    ax[1].axhline(y=0, color='black')
    ax[1].bar(energies, foscs, width=250)
    ax[1].set(ylabel='Oscillator strength')
    ax[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.gca().invert_xaxis()
    plt.xlabel(r'Wavenumber / cm$^{-1}$')
    plt.minorticks_on()
    fig.set_size_inches(6, 6)
    if file_name is not None:
        fig.savefig("{}-Phospho-Spectrum.png".format(file_name), dpi=300)
    plt.close(fig)
