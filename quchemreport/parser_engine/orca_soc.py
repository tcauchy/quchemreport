"""Spin-orbit coupling (SOC) results of ORCA TD-DFT calculations (``%tddft dosoc true end``).

Why a reader of its own (24/11/2026, decision of Thomas Cauchy): cclib reads only the
spin-free singlet and triplet states of these calculations, on purpose (cclib issue #1172:
the spin-mixed states would not match ``etsyms``), and leaves the SOC data to the "extensions"
of its version 2, not designed yet (cclib issue #1419). This module reads, from the log of one
calculation step, what the report needs for the SOC: the spin-free states, the SOC matrix
elements between triplets and singlets, the spin-orbit coupled states with their composition,
and the SOC-corrected absorption spectrum. It should move to cclib once the attributes exist.

Read in the ORCA 6.1 output (blocks named in the comments). When a log has several SOC
calculations (optimization steps), the last one is kept: the one at the final geometry.

Units: energies in cm-1; SOC matrix elements in cm-1, as the magnitude
|<T|H_SO|S>| = sqrt(sum over M_S of |<T,M_S|H_SO|S>|^2) (table "MS= 0 -1 +1").
"""
import re

_STATE = re.compile(r"^STATE\s+(\d+):\s+E=\s*(-?[\d.]+)\s+au\s+(-?[\d.]+)\s+eV\s+(-?[\d.]+)\s+cm\*\*-1.*Mult\s+(\d)")
_COMPLEX = re.compile(r"\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)")
_SOC_STATE = re.compile(r"^\s*STATE\s+(\d+):\s+(-?[\d.]+)\s*$")
_COMPONENT = re.compile(r"^\s+([\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*:\s*(\d+)\s+(\d)\s+(-?\d)\s*$")
_TRANSITION = re.compile(r"^\s*0-[\d.]+A\s+->\s+(\d+)-([\d.]+)A\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.eE+-]+)")


def read_orca_soc(logfile):
    """SOC results of the last SOC calculation of an ORCA log, or None if it has none.

    Parameters
    ----------
    logfile : str
        ORCA output (one calculation step).

    Returns
    -------
    dict or None
        ``singlets``, ``triplets``: spin-free states, lists of dicts with ``number`` (S1, S2...
        or T1, T2...: 1-based within the multiplicity) and ``energy`` (cm-1);
        ``socme``: dict {(triplet number, singlet number): |<T|H_SO|S>| (cm-1)}, singlet 0
        being the ground state;
        ``states``: spin-orbit coupled states (state 0: ground state), lists of dicts with
        ``energy`` (cm-1), ``fosc`` (oscillator strength from the ground state, 0 for the ground
        state) and ``components`` (list of (weight, "S1" / "T1", M_S), largest first);
        ``ground_state_shift``: SOC stabilization of the ground state (cm-1).
    """
    with open(logfile, errors="replace") as fd:
        lines = fd.read().splitlines()
    starts = [i for i, line in enumerate(lines) if "SPIN-ORBIT COUPLING" in line and "TD-DFT" in line]
    if not starts:
        return None
    # Spin-free states printed before the last SOC block
    first = starts[-1]
    singlets, triplets = _spin_free_states(lines[:first])
    block = lines[first:]
    result = {"singlets": singlets, "triplets": triplets, "socme": {}, "states": [],
              "ground_state_shift": None}

    i = 0
    while i < len(block):
        line = block[i]
        #            Root                          <T|HSO|S>  (Re, Im) cm-1
        #          T      S           MS= 0                  -1                    +1
        #          1      0    (   0.00 ,  -38.07)    ( -18.72 ,  -23.08)    ( -18.72 ,   23.08)
        if "SOCME BETWEEN TRIPLETS AND SINGLETS" in line and "REDUCED" not in line \
                and i + 3 < len(block) and "MS=" in block[i + 3]:
            i += 5
            while i < len(block) and not block[i].strip().startswith("---"):
                parts = block[i].split()
                values = _COMPLEX.findall(block[i])
                if len(parts) >= 2 and len(values) == 3:
                    magnitude = sum(float(re_) ** 2 + float(im) ** 2 for re_, im in values) ** 0.5
                    result["socme"][(int(parts[0]), int(parts[1]))] = magnitude
                i += 1
            continue
        # SOC stabilization of the ground state:  -0.1540 cm-1
        if "SOC stabilization of the ground state" in line:
            result["ground_state_shift"] = float(line.split(":")[1].split()[0])
        #  STATE  1:  30721.54
        #                       0.36035      0.50632      0.32248 : 1     1     0
        if line.startswith("Eigenvectors of the SOC matrix"):
            i += 1
            while i < len(block) and (block[i].strip() == "" or "E(cm-1)" in block[i]):
                i += 1
            while i < len(block):
                state = _SOC_STATE.match(block[i])
                component = _COMPONENT.match(block[i])
                if state:
                    result["states"].append({"energy": float(state.group(2)), "fosc": 0.0, "components": []})
                elif component and result["states"]:
                    weight, root, spin, ms = (float(component.group(1)), int(component.group(4)),
                                              int(component.group(5)), int(component.group(6)))
                    label = "S0" if root == 0 else f"{'T' if spin == 1 else 'S'}{root}"
                    result["states"][-1]["components"].append((weight, label, ms))
                elif block[i].strip():
                    break
                i += 1
            continue
        #       SOC CORRECTED ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS
        #   0-1.0A  ->  1-3.0A    3.808986   30721.5   325.5   0.000000001   0.00000 ...
        if "SOC CORRECTED ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS" in line:
            i += 1
            while i < len(block):
                found = _TRANSITION.match(block[i])
                if found:
                    k = int(found.group(1))
                    if k < len(result["states"]):
                        result["states"][k]["fosc"] = float(found.group(6))
                elif block[i].strip() and not block[i].strip().startswith(("-", "Transition", "(eV)")) \
                        and "->" not in block[i]:
                    break
                i += 1
            continue
        i += 1
    for state in result["states"]:
        state["components"].sort(key=lambda c: -c[0])
    return result


def _spin_free_states(lines):
    """Spin-free singlets and triplets of the last TD-DFT block of lines.

    ORCA numbers all the states together (STATE 1..n singlets, then n+1..2n triplets in the
    triplets block); they are renumbered within their multiplicity.
    """
    singlets, triplets = [], []
    current = None
    for line in lines:
        # TD-DFT/TDA EXCITED STATES (SINGLETS)  (TD-DFT EXCITED STATES without TDA)
        header = re.search(r"EXCITED STATES \((SINGLETS|TRIPLETS)\)", line)
        if header:
            current = singlets if header.group(1) == "SINGLETS" else triplets
            current.clear()
            continue
        found = _STATE.match(line)
        if found and current is not None:
            mult = int(found.group(5))
            target = singlets if mult == 1 else triplets if mult == 3 else None
            if target is not None:
                target.append({"number": len(target) + 1, "energy": float(found.group(4))})
    return singlets, triplets
