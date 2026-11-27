"""The PDF report: a LaTeX document built with pylatex and compiled with pdflatex.

generate_report fills the document section by section, in this order: log files, molecule,
computational details, wavefunction (orbitals, MEP, population analysis), frequencies,
reactivity (conceptual DFT, Fukui functions), excited states, optimized excited states, other
topologies, atomic coordinates. Each ``fill_*_section`` function creates a section and calls
``render_*`` functions, one per table or figure; a section that receives nothing is left out.
Only the parts asked in ``output.include`` are rendered, and the verbosity changes what is shown:
``full`` (all tables and pictures), ``si`` (supporting information, plain layout, fewer
pictures) and ``text`` (no grid pictures).

The pictures are read from the output folder with the names written by
visualization_engine.generate_visualization (``img-log<k>-...png``, see docs/architecture.md);
a missing picture is simply not shown.
"""

import json
import os
import re
from collections import defaultdict
import numpy as np
from pylatex import Document, Section, MultiColumn,LargeText, Tabular, LongTable
from pylatex.utils import italic, bold, NoEscape, escape_latex
from pylatex.package import Package
from pylatex.table import Tabularx
from cclib.parser.utils import PeriodicTable

from quchemreport.config.config import Config, OutputInclude
from quchemreport.utility_services.computed_data import ComputedData, config_logfile
from quchemreport.utility_services.log_data import functional_label
from quchemreport.utility_services.parameters import THRESHOLD_IR_INTENSITIES, surface_color_names
# km/mol: below this for every mode, the molecule has no IR-active vibration (O2, N2...)
IR_INACTIVE = 0.1
from quchemreport.utility_services.units import nm_to_wnb, Eh_to_Kcal_mol, a0_to_pm
from quchemreport.data_enrichment import mep_surface
from quchemreport.utility_services.naming import spin_letter, state_label, state_number

# Paragraph column without justification (needs the array package): the justified lists of
# log files were stretched with large spaces and hyphenated (SP-plus.log)
RAGGED = r">{\raggedright\arraybackslash}p"
# Paragraph column centred. Tables: first column as before, the others centred (Thomas
# Cauchy, 27/11/2026); the two-column "label: value" lists stay left-aligned.
CENTERED = r">{\centering\arraybackslash}p"

# TODO : NEED TO THINK ABOUT THE MAX VALUE

def add_row_filter(tab, l, expr='N/A'):
    """Add the row l to the table unless its value (l[1], or every value of a tuple) is expr ('N/A')."""
    if (
        ((l[1].__class__ != tuple) and (l[1] != expr)) # case 1
        or ((l[1].__class__ == tuple) and np.all([elt != expr for elt in l[1]])) # case 2
    ): 
        ## case 1 : value is not a tuple AND is not expr (eg. N/A)
        ## case 2 : value is a tuple AND ALL values are different from expr (eg. N/A)
        tab.add_row(l)


# Folder of the pictures of the report being written (generate_report), read by trim_option.
IMAGE_DIR = None
# Margin kept around the drawn part of a picture by trim_option, fraction of the picture side.
TRIM_MARGIN = 0.03


def content_box(names):
    """Union (left, top, right, bottom, width, height, dpi) of the non-white parts of the
    pictures (same size), or None (missing picture, different sizes, nothing drawn)."""
    from PIL import Image
    box, size, dpi = None, None, 72.0
    for name in names:
        path = os.path.join(IMAGE_DIR or "", name)
        if not os.path.isfile(path):
            return None
        with Image.open(path) as im:
            if size is not None and im.size != size:
                return None
            size = im.size
            dpi = float((im.info.get("dpi") or (72.0, 72.0))[0]) or 72.0
            drawn = im.convert("L").point(lambda v: 255 if v < 250 else 0).getbbox()
        if drawn is None:
            continue
        box = drawn if box is None else (min(box[0], drawn[0]), min(box[1], drawn[1]),
                                         max(box[2], drawn[2]), max(box[3], drawn[3]))
    if box is None:
        return None
    return (*box, *size, dpi)


def trim_option(names):
    """Options ``,trim=...,clip`` of \\includegraphics that cut the same white border off
    all the pictures given (the cells of one grid keep the same proportions), with a margin of
    TRIM_MARGIN; "" when the pictures cannot be read.

    Why (27/11/2026, Thomas Cauchy): the 3D pictures are square and framed on the largest
    dimension of the molecule, so an elongated molecule leaves a white band above and below
    (acetone topology: 57 % of the height drawn). The picture files are kept whole (used for
    talks and papers); only their place in the report is cropped."""
    box = content_box(names)
    if box is None:
        return ""
    left, top, right, bottom, width, height, dpi = box
    m = TRIM_MARGIN * max(width, height)
    bp = 72.0 / dpi
    cut = [max(0.0, left - m), max(0.0, height - bottom - m), max(0.0, width - right - m), max(0.0, top - m)]
    return ",trim=" + " ".join(f"{c * bp:.1f}bp" for c in cut) + ",clip"


def trimmed_ratio(names):
    """Height / width of the pictures once cropped by trim_option (1 if not cropped)."""
    box = content_box(names)
    if box is None:
        return 1.0
    left, top, right, bottom, width, height, _ = box
    m = TRIM_MARGIN * max(width, height)
    w = min(width, right + m) - max(0.0, left - m)
    h = min(height, bottom + m) - max(0.0, top - m)
    return h / w


class Xltabular(Tabularx):
    """xltabular environment: a longtable (page breaks) of fixed total width with X columns."""
    _latex_name = "xltabular"
    packages = [Package("xltabular")]


def figure_one_col(doc, nomPng, taillePng="7cm", caption="Figure", escapeCaption=True):
    """Add a centred figure with one picture.

    Parameters
    ----------
    doc : pylatex container
    nomPng : str
        Picture file, relative to the output folder.
    taillePng : str, optional
        LaTeX width.
    caption : str, optional
    escapeCaption : bool, optional
        Escape the LaTeX special characters of the caption.
    """
    if escapeCaption:
        caption = escape_latex(caption)
    
    doc.append(NoEscape(r'\begin{figure}[H]'))
    doc.append(NoEscape(r'\begin{center}'))
    doc.append(NoEscape(r'\includegraphics[width='+taillePng+trim_option([nomPng])+']{'+nomPng+'}'))
    doc.append(NoEscape(r'\end{center}'))
    doc.append(NoEscape(r'\vspace{-5mm}'))
    doc.append(NoEscape(r'\caption{' + caption + '}'))
    doc.append(NoEscape(r'\end{figure}'))


def figure_two_col(doc, nomPng, nomPng2, taillePng="7cm", caption="Figure", escapeCaption=True):
    """Add a centred figure with two pictures side by side (parameters as figure_one_col)."""
    if escapeCaption:
        caption = escape_latex(caption)
    
    doc.append(NoEscape(r'\begin{figure}[H]'))
    doc.append(NoEscape(r'\begin{center}'))
    trim = trim_option([nomPng, nomPng2])
    doc.append(NoEscape(r'\includegraphics[width='+taillePng+trim+']{'+nomPng+'}'))
    doc.append(NoEscape(r'\includegraphics[width='+taillePng+trim+']{'+nomPng2+'}'))
    doc.append(NoEscape(r'\end{center}'))
    doc.append(NoEscape(r'\vspace{-5mm}'))
    doc.append(NoEscape(r'\caption{' + caption + '}'))
    doc.append(NoEscape(r'\end{figure}'))


# Height kept for the pictures of one grid figure (A4, 15 mm margins: the text block is about
# 250 mm high; the rest is left for the caption so that a figure never exceeds a page).
GRID_MAX_HEIGHT_MM = 190
TEXT_WIDTH_MM = 180


def color_of(kind, sign, capital=False):
    """Name of the positive or negative surface colour of kind (MO, EDD, fukui, NTO) for the
    captions: output.colors of the YAML or the default (parameters.set_surface_colors), so
    that the captions follow the colours chosen by the user. Not escaped: the captions are."""
    name = surface_color_names[kind][sign]
    return name[:1].upper() + name[1:] if capital else name


def figure_grid(doc, cells, ncol, caption, escapeCaption=True, max_width=None):
    """Add the pictures of several states or orbitals as a grid, with one label under each cell.

    Replaces one figure per picture (a third of a page each): the 10 density differences and
    the 11 NTO pairs of the acetone test series took 9 pages, against 3 with the grid. Each
    cell holds one picture, or two side by side (two views, alpha/beta, hole/electron). A
    grid taller than GRID_MAX_HEIGHT_MM is split into several figures, the next ones with
    "(continued)" in the caption, so that the cells are not cut by a page break. Any list of
    cells is accepted: the states shown are the ones whose pictures exist.

    The pictures all have the same size: a cell is as wide as its number of pictures, and the
    cells are put on a row while they fit (a row holds ncol cells of the largest one). Why
    (27/11/2026): the transition densities of a state seen from two points of view (4 pictures)
    put the grid at one state per row, and the states with one view (2 pictures) then took the
    whole width, at twice the size; the caption overflowed onto the page footer.

    Parameters
    ----------
    doc : pylatex container
    cells : list of (list of str, str)
        Picture files (relative to the output folder) and label of each cell; the label is
        LaTeX code (escape it before).
    ncol : int
        Cells per row.
    caption : str
    escapeCaption : bool, optional
        Escape the LaTeX special characters of the caption.
    max_width : float, optional
        Largest width of a cell, as a fraction of the text width (the orbitals are kept at
        about the 7 cm of the former single figures instead of half the page).
    """
    if not cells:
        return
    if escapeCaption:
        caption = escape_latex(caption)
    gap = 0.01
    # rounded down: 3 x 0.327 + 2 x 0.01 exceeded the line and left 2 cells per row
    width = int(1000 * (1.0 - gap * (ncol - 1)) / ncol) / 1000
    if max_width is not None:
        width = min(width, max_width)
    n_pictures = max(len(pngs) for pngs, _ in cells)
    # the same crop for all the pictures of the grid (trim_option): aligned rows
    all_pngs = [p for pngs, _ in cells for p in pngs]
    trim = trim_option(all_pngs)
    ratio = trimmed_ratio(all_pngs) if trim else 1.0
    # height of a row = height of a picture + label (3.5 mm per extra line of label: \newline)
    extra_lines = max(str(label).count(r"\newline") for _, label in cells)
    row_mm = TEXT_WIDTH_MM * width / n_pictures * ratio + 6 + 3.5 * extra_lines
    rows_per_figure = max(1, int(GRID_MAX_HEIGHT_MM // row_mm))
    # rows: cells of len(pngs) pictures, ncol * n_pictures pictures per row
    rows, current, used = [], [], 0
    for cell in cells:
        n = len(cell[0])
        if current and used + n > ncol * n_pictures:
            rows.append(current)
            current, used = [], 0
        current.append(cell)
        used += n
    if current:
        rows.append(current)
    chunks = [rows[i:i + rows_per_figure] for i in range(0, len(rows), rows_per_figure)]
    for c, chunk in enumerate(chunks):
        doc.append(NoEscape(r'\begin{figure}[H]'))
        doc.append(NoEscape(r'\centering'))
        for r, row in enumerate(chunk):
            if r:
                doc.append(NoEscape(r'\\[2mm]'))
            for k, (pngs, label) in enumerate(row):
                if k:
                    doc.append(NoEscape(r'\hspace{' + f'{gap}' + r'\textwidth}%'))
                n = len(pngs)
                # width of n pictures of the grid, with the gaps between cells it replaces
                cell_width = width * n / n_pictures + gap * (n / n_pictures - 1) if n < n_pictures else width
                cell_width = int(1000 * cell_width) / 1000
                sub = 1.0 / n
                doc.append(NoEscape(r'\begin{minipage}[t]{' + f'{cell_width:.3f}' + r'\textwidth}\centering'))
                doc.append(NoEscape(''.join(r'\includegraphics[width=' + f'{sub:.3f}' + r'\linewidth' + trim + ']{' + p + '}'
                                            for p in pngs) + '%'))
                doc.append(NoEscape(r'\\[-1mm]{\footnotesize ' + label + '}'))
                doc.append(NoEscape(r'\end{minipage}%'))  # % : no space between the cells
        doc.append(NoEscape(r'\caption{' + (continued_caption(caption) if c else caption) + '}'))
        doc.append(NoEscape(r'\end{figure}'))


def continued_caption(caption):
    """Caption of the next figures of a grid split by figure_grid: the first sentence of the
    caption and its source, with "(continued)". Why (27/11/2026): the whole caption was repeated,
    up to 13 lines for the transition densities of a chiral molecule, often on the same page."""
    first = caption.split(". ", 1)[0].rstrip(".")
    source = caption[caption.rfind("(Source:"):] if "(Source:" in caption else ""
    return first + ". " + (source + " " if source else "") + "(continued)"


def sanitize_filename(name: str) -> str:
    """Name usable in a file name: characters other than letters, digits, _ and - become _."""
    return re.sub(r'[^A-Za-z0-9_-]+', '_', name)


def generate_empty_document(report_type):
    #####################################################################
    #                                                                   #
    ## production du rapport en .tex                                 #
    #                                                                   #
    #####################################################################
    
    ### create document and import needed packages
    
    """Empty LaTeX document with the packages, page style and title of the report.

    Parameters
    ----------
    report_type : str
        full, si or text (the full report has the coloured layout).

    Returns
    -------
    pylatex Document
    """
    doc = Document(documentclass="article",font_size='small',geometry_options='a4paper, left=15mm, right=15mm, top=15mm, bottom=15mm')
    # packages
    doc.packages.append(NoEscape(r'\usepackage{xcolor}'))
    doc.packages.append(NoEscape(r'\usepackage{longtable}'))
    # array: ragged-right paragraph columns (>{\raggedright\arraybackslash}p{}), the justified
    # lists of log files were stretched with large spaces
    doc.packages.append(NoEscape(r'\usepackage{array}'))
    doc.packages.append(NoEscape(r'\usepackage{graphicx}'))
    
    # Add 'float' to force figures at exact positions with [H].
    # If space is lacking, LaTeX moves them to the next page, preserving order.
    doc.packages.append(NoEscape(r'\usepackage{float}'))
    doc.packages.append(NoEscape(r'\usepackage{placeins}'))
    
    # For full report, definition of an original layout. 
    if report_type == 'full': 
        doc.packages.append(NoEscape(r'\usepackage{titlesec}'))
        doc.packages.append(Package('datetime'))
        doc.packages.append(NoEscape(r'\usepackage{extramarks}'))
        doc.packages.append(NoEscape(r'\usepackage{fancyhdr}'))
        doc.packages.append(NoEscape(r'\pagestyle{fancy}'))
        doc.packages.append(NoEscape(r'\fancyhf{}'))
        # define header / footer texts
        doc.packages.append(NoEscape(r'\lhead{MOLECULAR CALCULATION REPORT}'))
        doc.packages.append(NoEscape(r'\rhead{Generated by quchemreport}'))
        doc.packages.append(NoEscape(r'\lfoot{\today ~  \currenttime}'))
        doc.packages.append(NoEscape(r'\rfoot{Page \thepage}'))
        doc.packages.append(NoEscape(r'\cfoot{}'))
        # redefine rules for header / footer
        doc.packages.append(NoEscape(r'\renewcommand\headrulewidth{0.4pt}'))
        doc.packages.append(NoEscape(r'\renewcommand\footrulewidth{0.4pt}'))
        # redefine section
        doc.packages.append(NoEscape(r'\definecolor{ufrblue}{RGB}{0,161,140}'))
        doc.packages.append(NoEscape(r'\definecolor{bordeau}{RGB}{125,31,31}'))
        doc.packages.append(NoEscape(r'\titleformat{name=\section}[block]{\sc\large}{}{0pt}{\colorsection}'))
        doc.packages.append(NoEscape(r'\titlespacing{\section}{0pt}{\baselineskip}{2pt}'))
        doc.packages.append(NoEscape(r'\titlespacing{\subsection}{4pt}{\baselineskip}{0pt}'))
        doc.packages.append(NoEscape(r'\newcommand{\colorsection}[1]{'
                                        +r'\colorbox{ufrblue}{\parbox{\dimexpr\textwidth-2\fboxsep}{\textcolor{white}{'
                                        +r'\textbf{{\thesection.\ #1}}}}}}'))
        
    # For non full report, definition of a simple Title. 
    # SI mode is more suitable to conversion to word for supp Info generation. 
    if report_type != 'full': 
        doc.append(NoEscape(r'\begin{center}'))
        doc.append(LargeText(bold("Molecular Calculation Report generated by quchemreport \n")))
        doc.append(NoEscape(r'\today '))
        doc.append(NoEscape(r'\end{center}'))
        
    return doc


def render_ref_molecule_representation(section, output_dir, computed_data: ComputedData):
    """Topology picture of the reference, at the top of the molecule section.

    Returns
    -------
    bool
        Whether something was added to the section.
    """
    ref_id = computed_data.metadata.ref_log_file_idx
    log_file = computed_data.metadata.log_files[ref_id]
    
    nomPng = f"img-log{ref_id + 1}-TOPOLOGY.png"
    
    if ( not os.path.isfile(output_dir + nomPng) ):
        print(f"{nomPng} not found. Molecule representation for the reference can not be added to the report.\n")
        return False
    
    topology_figure(section, output_dir, nomPng, f"(Source: {log_file})")
    return True


def topology_figure(doc, output_dir, nomPng, source):
    """Molecule picture with the atom numbers, and its second view when there is one."""
    nomPng2 = nomPng.replace("-TOPOLOGY.png", "-TOPOLOGY_cam2.png")
    caption = "Chemical structure diagram with atomic numbering"
    if os.path.isfile(output_dir + nomPng2):
        figure_grid(doc, [([nomPng, nomPng2], "")], 1, max_width=0.8,
                    caption=f"{caption} (two points of view). {source}")
    else:
        figure_one_col(doc, nomPng, caption=f"{caption}. {source}")


def render_grouped_values_with_logs(table, label, values, log_files, ref_id, table_col):
    # Regrouper les logs par valeurs équivalentes
    """Table row of a computational detail, grouping the log files by value.

    A single value is written alone; several values are written in bold with their log
    files, the value of the reference first.

    Parameters
    ----------
    table : pylatex table
    label : str
        First column.
    values : list
        One value per model (None values are skipped).
    log_files : list of str
    ref_id : int
        Index of the reference model.
    table_col : int
        Number of columns of the table.
    """
    grouped = []  # liste de tuples (valeur, [logs])

    for val, log in zip(values, log_files):
        
        if val is None:
            continue

        # Chercher si une valeur équivalente existe déjà
        for entry in grouped:
            if val == entry[0]:  # comparaison par égalité
                entry[1].append(log)
                break
        else:
            grouped.append([val, [log]])  # nouvelle entrée

    if not grouped:
        return
    
    # Display only a single value if it’s the same across all log files
    if len(grouped) == 1:
        val = grouped[0][0]
        row = [label, f"{val}"]
        row += [''] * (table_col - len(row))
        table.add_row(row)
        return

    ref_value = values[ref_id]
    grouped_entries = []

    # Display the reference value first if present
    if ref_value is not None:
        for val, logs in grouped:
            if val == ref_value:
                log_str = ', '.join(escape_latex(str(log)) for log in logs)
                grouped_entries.append(f"\\textbf{{{val}}} ({log_str})")
                break

    # Add other values
    grouped_entries += [
        f"\\textbf{{{val}}} ({', '.join(escape_latex(str(log)) for log in logs)})"
        for val, logs in grouped
        if val != ref_value
    ]

    # Add row to the table
    row = [label, NoEscape('; '.join(grouped_entries))]
    row += [''] * (table_col - len(row))
    table.add_row(row)
    table.add_row([''] * table_col)


def imaginary_frequencies(freqs):
    """Imaginary frequencies (negative values in cclib), in cm-1."""
    return [f for f in (freqs or []) if f is not None and f < 0]


def format_imaginary(freqs):
    """Imaginary frequencies as text: ``1157.2i, 830.2i``."""
    return ", ".join("%.1fi" % abs(f) for f in freqs)


def quality_warnings(computed_data):
    """Warnings on the nature of the structures: optimizations that did not converge and
    imaginary frequencies (the structure is not a minimum).

    Returns
    -------
    list of str
        One sentence per problem, with its log file.
    """
    warnings = []
    log_files = computed_data.metadata.log_files
    for i, log in enumerate(log_files):
        if computed_data.results.geometry.OPT_DONE[i] is False:
            warnings.append(f"{log}: the geometry optimization did not converge "
                            "(the geometry is not a stationary point).")
        imag = imaginary_frequencies(computed_data.results.freq.vibrational_freq[i])
        if imag:
            warnings.append(f"{log}: {len(imag)} imaginary frequenc{'y' if len(imag) == 1 else 'ies'} "
                            f"({format_imaginary(imag)} cm-1): the structure is not a minimum.")
    return warnings


# Order of the calculations in the first table of the report: the usual sequence of a study
# (optimization, then the calculations at that geometry, then the excited-state optimizations).
JOB_SEQUENCE = {"opt": (0, "Ground-state optimization"), "freq": (1, "Frequencies"),
                "td": (2, "Vertical excited states"), "sp": (3, "Single points"),
                "nmr": (4, "NMR"), "opt_es": (5, "Excited-state optimizations"),
                "freq_es": (6, "Excited-state frequencies")}


def optimized_state_name(computed_data, config, i):
    """S1, T1... of the state optimized in log i (opt_es), or None."""
    es_number = config_logfile(config, computed_data, i).excited_state_number
    if es_number is None:
        return None
    # S1/T1 as in the tables ("State 1" did not say which multiplicity)
    et_sym_i = computed_data.results.excited_states.et_sym[i]
    if et_sym_i and es_number <= len(et_sym_i) and spin_letter(et_sym_i[es_number - 1]):
        name = state_label(et_sym_i, es_number - 1)
        if state_number(et_sym_i, es_number - 1) != es_number:
            name += f" (state {es_number} of the log)"
        return name
    return f"State {es_number}"


def excited_states_count(et_sym):
    """'10 singlets, 10 triplets' from the symmetries of the excited states of a log."""
    if not et_sym:
        return ""
    names = {"S": "singlet", "D": "doublet", "T": "triplet", "Q": "quartet", "P": "quintet"}
    counts = {}
    for sym in et_sym:
        letter = spin_letter(sym)
        counts[letter] = counts.get(letter, 0) + 1
    parts = []
    for letter, n in counts.items():
        word = names.get(letter, "state")
        parts.append(f"{n} {word}{'s' if n > 1 else ''}")
    return ", ".join(parts)


def logfile_rows(computed_data, config):
    """Rows (calculation, log file, charge and multiplicity, geometry, details) of the table
    of the log files, in the order of JOB_SEQUENCE; the calculation is only written on the
    first row of each kind."""
    ref_id = computed_data.metadata.ref_log_file_idx
    log_files = computed_data.metadata.log_files
    job_types = computed_data.comp_details.general.job_type
    charges = computed_data.molecule.charges or [None] * len(log_files)
    mults = computed_data.molecule.multiplicity or [None] * len(log_files)
    geometry_of = {}
    for _, members in geometry_groups(computed_data):
        for i in members:
            geometry_of[i] = members[0]

    def kind(i):
        jobs = [j for j in (job_types[i] or []) if j in JOB_SEQUENCE] or ["sp"]
        jobs.sort(key=lambda j: JOB_SEQUENCE[j][0])
        label = JOB_SEQUENCE[jobs[0]][1]
        if len(jobs) > 1:   # ORCA Opt Freq in one job
            label += " and " + " and ".join(JOB_SEQUENCE[j][1].lower() for j in jobs[1:])
        return JOB_SEQUENCE[jobs[0]][0], label

    rows, previous = [], None
    for i in sorted(range(len(log_files)), key=lambda i: (kind(i)[0], i)):
        order, label = kind(i)
        jobs = job_types[i] or []
        name = log_files[i] + (" (reference)" if i == ref_id else "")
        charge = "" if charges[i] is None else (f"{charges[i]:+d}" if charges[i] else "0")
        mult = "" if mults[i] is None else str(mults[i])
        first = geometry_of.get(i)
        if first is None:
            geometry = ""
        elif first != i:
            geometry = f"of {log_files[first]}"
        else:
            geometry = "optimized" if any(j.startswith("opt") for j in jobs) else "own"
        details = []
        if "td" in jobs:
            details.append(excited_states_count(computed_data.results.excited_states.et_sym[i]))
        if "opt_es" in jobs:
            state = optimized_state_name(computed_data, config, i)
            if state:
                details.append(state)
        rows.append(("" if label == previous else label, name, f"{charge}, {mult}".strip(", "),
                     geometry, "; ".join(d for d in details if d)))
        previous = label
    return rows


def fill_logfiles_details_section(doc, computed_data, config: Config):
    """First section: table of the log files grouped by kind of calculation, in the order of
    a study (JOB_SEQUENCE), with charge, multiplicity, geometry and states; then the quality
    warnings.

    Why (27/11/2026, Thomas Cauchy): the former list (one log per line with its type in
    brackets, then the reference and the optimized states on separate rows, and the charges
    and multiplicities in the molecule section) did not show how the calculations follow
    each other. The geometry column says which logs are at the geometry of which
    optimization."""
    with doc.create(Section('LOGFILES DETAILS')) :

        rows = logfile_rows(computed_data, config)
        # xltabular: the columns X share the text width (a longtable with fixed widths was
        # wider than the text with its column separations, 27/11/2026); \hsize factors sum to
        # the number of X columns. Log file names may break after "_".
        spec = (r">{\hsize=1.15\hsize\raggedright\arraybackslash}X "
                r">{\hsize=1.15\hsize\centering\arraybackslash}X c "
                r">{\hsize=0.95\hsize\centering\arraybackslash}X "
                r">{\hsize=0.75\hsize\centering\arraybackslash}X")
        breakable = lambda text: NoEscape(escape_latex(text).replace(r"\_", r"\_\allowbreak{}"))
        with doc.create(Xltabular(spec)) as log_table:
            log_table.add_hline()
            log_table.add_row(["Calculation", "Log file", "Charge, mult.", "Geometry", "States"])
            log_table.add_hline()
            for k, row in enumerate(rows):
                if k and row[0]:
                    log_table.append(NoEscape(r'\noalign{\vskip 1mm}'))   # space between kinds
                log_table.add_row([row[0], breakable(row[1]), row[2], breakable(row[3]), row[4]])
            log_table.add_hline()

        # Non-converged optimizations and imaginary frequencies
        warnings = quality_warnings(computed_data)
        for warning in warnings:
            print("WARNING: " + warning)
        if warnings:
            with doc.create(LongTable("p{0.2\\textwidth} " + RAGGED + "{0.76\\textwidth}")) as log_table:
                log_table.add_row([bold('Warnings'), bold("\n".join(warnings))])


def render_molecule_identification(section, computed_data: ComputedData, designated_name, report_type):
    
    """Table identifying the molecule: name, formula, charge, multiplicity, mass, InChI...
    (``molecule_identification`` tags).


    Returns
    -------
    bool
        Whether something was added to the section.
    """
    with section.create(LongTable("p{0.2\\textwidth} " + RAGGED + "{0.76\\textwidth}")) as mol_table:

        ref_id = computed_data.metadata.ref_log_file_idx
        log_files = computed_data.metadata.log_files
        charges = computed_data.molecule.charges
        multiplicities = computed_data.molecule.multiplicity
        monoisotopic_mass = computed_data.molecule.monoisotopic_mass
        inchi = (computed_data.molecule.inchi).rstrip().split("=")[-1] if computed_data.molecule.inchi else None
        smi = computed_data.molecule.smi
        
        # No data to display
        if (
            log_files is None
            and charges is None
            and multiplicities is None
            and monoisotopic_mass is None
            and inchi is None
            and smi is None
        ):
            return False
        
        # Render the corresponding Designated Name 
        if designated_name:
            mol_table.add_row(['Designated Name', designated_name ])
        
        # Render the corresponding formula (unique value due to uniformity check)
        if computed_data.molecule.formula != None: 
            mol_table.add_row(['Formula', computed_data.molecule.formula])
        
        # No charge and multiplicity here: those of every log are in the table of the log files
        # (logfile_rows), since 27/11/2026
        
        # For full report, more data will be printed 
        if report_type == 'full': 
            if monoisotopic_mass is not None:
                mol_table.add_row(['Monoisotopic mass', "%.5f Da" % monoisotopic_mass])

            if inchi is not None:
                mol_table.add_row(['InChI'  , inchi ])

            # TODO : IS IT BETTER TO JUST TRUNCATE THE SMILES IF TOO LONG ?
            if (
                smi is not None
                and (len(smi) < 80)
            ):
                mol_table.add_row(['SMILES' , smi ])
                
    return True


def render_atomic_coordinates_table(doc, atoms, atoms_Z, log_file: str, table_title: str):
    """Table of Cartesian coordinates, printed in pm (given in Angstrom).

    Parameters
    ----------
    doc : pylatex container
    atoms : list of float
        Flat coordinates.
    atoms_Z : list of int
    log_file : str
        Source written under the title.
    table_title : str
    """
    doc.append(NoEscape(r'\begin{footnotesize}'))
    doc.append(NoEscape(r'\begin{center}'))

    # tabular for the small molecules: a longtable left its title alone at the bottom of a page
    table_class = Tabular if len(atoms) <= 40 else LongTable
    with doc.create(table_class('p{0.5cm}rcccp{0.5cm}')) as table:
        table.add_row(
            [MultiColumn(6, align='c', data=table_title)]
        )
        table.add_row(
            # paragraph column: the list of the models at the same geometry would widen the table
            [MultiColumn(6, align=NoEscape('p{6.5cm}'), data=f"Source: {log_file}")]
        )
        table.add_hline()
        table.add_row([
            "",
            MultiColumn(1, align='c', data='Atom'),
            MultiColumn(1, align='c', data='X'),
            MultiColumn(1, align='c', data='Y'),
            MultiColumn(1, align='c', data='Z'),
            ""
        ])
        table.add_hline()

        # pm with 2 decimals = the 4 decimals in Angstrom of the former tables
        for j, coord in enumerate(atoms):
            table.add_row([
                "",
                PeriodicTable().element[atoms_Z[j]],
                f"{100 * coord[0]:.2f}",
                f"{100 * coord[1]:.2f}",
                f"{100 * coord[2]:.2f}",
                ""
            ])

        table.add_hline()

    doc.append(NoEscape(r'\end{center}'))
    doc.append(NoEscape(r'\end{footnotesize}'))

def coordinates_title(computed_data, log_idx):
    """Title of a coordinates table, which says when the optimization did not converge."""
    if computed_data.results.geometry.OPT_DONE[log_idx] is False:
        return "Table. Last cartesian atomic coordinates in pm (optimization NOT converged)"
    return "Table. Converged cartesian atomic coordinates in pm"


# Two geometries are the same when no coordinate differs by more than this (Angstrom); the
# tables print 4 decimals.
SAME_GEOMETRY_TOL = 5e-4


def geometry_groups(computed_data):
    """Models grouped by geometry, the group of the reference first.

    Only the reference geometry and the geometries that differ from it (optimizations of
    ions or excited states) are worth a table: the frequency, TD and single-point models
    computed at the reference geometry repeated the same table (nine tables for the acetone
    test series instead of three). Models with the same geometry share one table.

    Returns
    -------
    list of (numpy.ndarray, list of int)
        Coordinates (n_atoms x 3) and the indexes of the models with these coordinates.
    """
    ref_id = computed_data.metadata.ref_log_file_idx
    all_atoms = computed_data.results.geometry.elements_3D_coords
    order = [ref_id] + [i for i in range(len(all_atoms)) if i != ref_id]
    groups = []
    for i in order:
        if all_atoms[i] is None or computed_data.molecule.atoms_Z[i] is None:
            continue
        atoms = np.array(all_atoms[i], dtype=float).reshape((-1, 3))
        for coords, members in groups:
            if coords.shape == atoms.shape and np.max(np.abs(coords - atoms)) <= SAME_GEOMETRY_TOL:
                members.append(i)
                break
        else:
            groups.append((atoms, [i]))
    return groups


def render_coordinates_group(doc, computed_data, coords, members):
    """Coordinate table of one geometry group (see geometry_groups), with all its sources."""
    log_files = computed_data.metadata.log_files
    first = members[0]
    sources = log_files[first]
    if len(members) > 1:
        sources += " (same geometry: " + ", ".join(log_files[i] for i in members[1:]) + ")"
    render_atomic_coordinates_table(
        doc=doc,
        atoms=coords,
        atoms_Z=computed_data.molecule.atoms_Z[first],
        log_file=sources,
        table_title=coordinates_title(computed_data, first),
    )


def fill_molecule_section(doc, output_dir, computed_data: ComputedData, designated_name, report_type, config_output: OutputInclude):
    """Molecule section: topology picture, identification and coordinates of the reference."""
    section = Section('MOLECULE')
    
    added_ref_molecule = False
    added_mol_identification = False
    added_atomic_oordinates = False
    
    if config_output.molecule.molecule_representation:
        added_ref_molecule = render_ref_molecule_representation(section, output_dir, computed_data)
    
    # TODO : TO IMPLEMENT : The selection of which informations to display is not done. For now all data available are displayed.
    if len(config_output.molecule.molecule_identification) != 0:
        added_mol_identification = render_molecule_identification(section, computed_data, designated_name, report_type)
        
    # All the coordinates are in the last section (fill_atomic_coordinates_section), the
    # reference first. Why (27/11/2026, Thomas Cauchy): the reference was in this section for
    # fewer than 15 atoms and the other geometries at the end, which split them, and a large
    # table at the top of the report pushed the results back.

    if (
        added_ref_molecule
        or added_mol_identification
        or added_atomic_oordinates
    ):
        doc.append(section)


def program_label(package, version):
    """Program and version as chemists write them: 'Gaussian 09, revision D.01' (cclib
    package_version '2009+D.01'), 'ORCA 6.1.0' ('6.1.0.post0')."""
    names = {"gaussian": "Gaussian", "orca": "ORCA", "gamess": "GAMESS"}
    name = names.get(str(package).lower(), str(package)) if package else ""
    if version is None:
        return name or None
    v = str(version)
    if name == "Gaussian":
        m = re.match(r"(\d{4})\+(\S+)", v)
        if m:
            return f"Gaussian {m.group(1)[2:]}, revision {m.group(2)}"
    if name == "ORCA":
        v = v.split(".post")[0]
    return f"{name} {v}".strip()


def states_summary(et_sym):
    """'10 singlets, 10 triplets' from the symmetries of the states read in the log."""
    if not et_sym:
        return ""
    counts = {}
    for sym in et_sym:
        letter = spin_letter(sym)
        kind = {"S": "singlet", "T": "triplet"}.get(letter, "state")
        counts[kind] = counts.get(kind, 0) + 1
    order = ["singlet", "triplet", "state"]
    return ", ".join(f"{counts[kind]} {kind}{'s' if counts[kind] > 1 else ''}"
                     for kind in order if kind in counts)


# Columns of the geometry convergence criteria (cclib geovalues / geotargets) by program
GEOMETRY_CRITERIA = {
    "gaussian": ["Max force", "RMS force", "Max displ.", "RMS displ."],
    "gamess": ["Max force", "RMS force"],
}


def fill_computational_details_section(doc, computed_data: ComputedData):
    """Computational details section: a table of the settings, each value with its log files
    when they differ (render_grouped_values_with_logs), then one table with a row per log
    file for what is specific to each one (convergence of the optimization, nuclear
    repulsion energy, temperature, excited states).

    The specific values were blocks of rows under 'Logfile : <name>' (two pages for the
    acetone test series); the table holds them in a few lines and shows the thresholds once
    when they are the same for every optimization.
    """
    with doc.create(Section('COMPUTATIONAL DETAILS')) :

        ref_id = computed_data.metadata.ref_log_file_idx
        log_files = computed_data.metadata.log_files
        general = computed_data.comp_details.general
        package = general.package
        software = str(package).lower() if package else None

        programs = [program_label(package, v) if v is not None else None for v in general.package_version]
        functional = [functional_label(f, d) if f is not None else None
                      for f, d in zip(general.functional, general.dispersion)]

        doc.append(NoEscape(r'\begin{footnotesize}'))
        with doc.create(LongTable("p{0.3\\textwidth} " + RAGGED + "{0.66\\textwidth}")) as param_table:
            render_grouped_values_with_logs(param_table, 'Program', programs, log_files, ref_id, table_col=2)
            render_grouped_values_with_logs(param_table, 'Computational method', general.last_theory, log_files, ref_id, table_col=2)
            render_grouped_values_with_logs(param_table, 'Functional', functional, log_files, ref_id, table_col=2)
            render_grouped_values_with_logs(param_table, 'Basis set', general.basis_set_name, log_files, ref_id, table_col=2)
            render_grouped_values_with_logs(param_table, 'Number of basis functions', general.basis_set_size, log_files, ref_id, table_col=2)
            # No "Electronic structure" row (closed / open shell): the table of the energies
            # gives restricted / unrestricted for each log (27/11/2026, redundant)
            render_grouped_values_with_logs(param_table, 'Integration grid', general.integration_grid, log_files, ref_id, table_col=2)
            render_grouped_values_with_logs(param_table, 'Solvent', general.solvent, log_files, ref_id, table_col=2)

            # Targets of the last SCF of the reference log
            scf_targets = general.scf_targets
            scfTargets = scf_targets[ref_id][-1] if scf_targets[ref_id] else None
            if software == "gaussian" and scfTargets is not None and len(scfTargets) >= 3:
                param_table.add_row(["SCF convergence targets",
                                     f"RMS density {scfTargets[0]:g}, max density {scfTargets[1]:g}, "
                                     f"energy {scfTargets[2]:g} (reference log)"])
            if software == "gamess" and scfTargets is not None:
                param_table.add_row(["SCF convergence target", f"density {scfTargets[0]:g} (reference log)"])
        doc.append(NoEscape(r'\end{footnotesize}'))

        render_log_specific_table(doc, computed_data, software)


def render_log_specific_table(doc, computed_data: ComputedData, software):
    """Table with one row per log file: convergence of the optimization (value, and the
    threshold in a first row when it is the same for all), nuclear repulsion energy,
    temperature and anharmonicity of the frequencies, excited states read."""
    ref_id = computed_data.metadata.ref_log_file_idx
    log_files = computed_data.metadata.log_files
    geometric_targets = computed_data.comp_details.geometry.geometric_targets
    geometric_values = computed_data.results.geometry.geometric_values
    temperature = computed_data.comp_details.freq.temperature
    anharmonicity = computed_data.comp_details.freq.anharmonicity
    nre = computed_data.results.geometry.nuclear_repulsion_energy_from_xyz
    et_sym = computed_data.results.excited_states.et_sym

    criteria = GEOMETRY_CRITERIA.get(software, [])
    n_crit = len(criteria)
    order = [ref_id] + [i for i in range(len(log_files)) if i != ref_id]
    geo = {}
    for i in order:
        if n_crit and geometric_targets[i] is not None and geometric_values[i] is not None \
                and len(geometric_targets[i]) >= n_crit and len(geometric_values[i][-1]) >= n_crit:
            geo[i] = (list(geometric_values[i][-1][:n_crit]), list(geometric_targets[i][:n_crit]))
    same_targets = len({tuple(t) for _, t in geo.values()}) == 1
    show_geo = bool(geo)

    rows = []
    for i in order:
        other = []
        if temperature[i] is not None:
            other.append(f"T = {temperature[i]:.2f} K")
        if anharmonicity[i] is not None:
            other.append("anharmonic")
        states = states_summary(et_sym[i])
        if states:
            other.append(states)
        if i not in geo and nre[i] is None and not other:
            continue
        cells = [log_files[i]]
        if show_geo:
            if i in geo:
                values, targets = geo[i]
                cells += [f"{v:.6f}" if same_targets else f"{v:.6f} ({t:.6f})" for v, t in zip(values, targets)]
            else:
                cells += [""] * n_crit
        cells.append(f"{nre[i]:.5f}" if nre[i] is not None else "")
        cells.append("; ".join(other))
        rows.append(cells)
    if not rows:
        return

    ncols = 1 + (n_crit if show_geo else 0) + 2
    doc.append(NoEscape(r'\begin{footnotesize}'))
    doc.append(NoEscape(r'\begin{center}'))
    with doc.create(Tabular(RAGGED + "{0.17\\textwidth}" + " c" * (ncols - 3) + " c " + CENTERED + "{0.22\\textwidth}")) as table:
        title = "Table. Specific results of each log file"
        if show_geo:
            title += (": convergence of the geometry optimization (last step, atomic units)"
                      + ("" if same_targets else ", value (threshold)"))
        table.add_row([MultiColumn(ncols, align='c', data=title)])
        table.add_hline()
        table.add_row(["Log file"] + (criteria if show_geo else [])
                      + [NoEscape(r"$E_{\mathrm{nuc}}$ (hartrees)"), "Other"])
        table.add_hline()
        if show_geo and same_targets:
            targets = next(iter(geo.values()))[1]
            table.add_row([italic("Threshold")] + [f"{t:.6f}" for t in targets] + ["", ""])
        for cells in rows:
            table.add_row(cells)
        table.add_hline()
    doc.append(NoEscape(r'\end{center}'))
    doc.append(NoEscape(r'\end{footnotesize}'))


def serialize_energies_data(total_energy, homo_ind, MO_energies, MO_labels, MO_indexes):
    """Canonical JSON of the energy data of a model, to group models with the same table."""
    return json.dumps({
        "total_energy": total_energy,
        "homo_ind": homo_ind,
        "MO_energies": MO_energies,
        "MO_labels": MO_labels,
        "MO_indexes": MO_indexes,
    }, sort_keys=True)


def energies_rows(computed_data: ComputedData, log_indexes):
    """Rows of the energy table for a group of models with the same values (see render_energies).

    Returns
    -------
    list of list
        [models, calculation, total energy, HOMO number, energies of the requested orbitals]
        for each set of orbitals (one row, or alpha and beta rows for unrestricted models).
    """
    log_files = computed_data.metadata.log_files
    i_log = log_indexes[0]
    total_energy = computed_data.results.wavefunction.total_molecular_energy[i_log]
    homo_ind = computed_data.results.wavefunction.homo_indexes[i_log]
    MO_energies = computed_data.results.wavefunction.MO_energies[i_log]
    MO_labels = computed_data.derived.wavefunction.MO_analysis.MO_labels[i_log]
    MO_indexes = computed_data.derived.wavefunction.MO_analysis.MO_indexes[i_log]
    models = ", ".join(log_files[idx] for idx in log_indexes)
    energy = "%.5f" % total_energy if total_energy is not None else ""
    if len(homo_ind) == 2:
        kinds = ["unrestricted, $\\alpha$", "unrestricted, $\\beta$"]
    else:
        multiplicity = computed_data.molecule.multiplicity[i_log]
        # ROHF/ROKS: one set of orbitals, the HOMO is the highest singly occupied MO
        kinds = ["restricted open-shell" if multiplicity is not None and multiplicity > 1 else "restricted"]
    rows = []
    for spin, kind in enumerate(kinds):
        values = ["%.2f" % MO_energies[spin][MO_indexes[spin][k]] for k in range(len(MO_labels[spin]))]
        rows.append([models if spin == 0 else "", NoEscape(kind), energy if spin == 0 else "",
                     homo_ind[spin] + 1] + values)
    return rows


def render_energies(section, computed_data: ComputedData):
    """Energy tables of all the models (one table per group of identical values).

    Returns
    -------
    bool
        Whether something was added to the section.
    """
    n_logs = len(computed_data.results.wavefunction.MO_energies)
    ref_id = computed_data.metadata.ref_log_file_idx
    
    isContentAdded = False

    seen = defaultdict(list)

    for i in range(n_logs):
        total_energy = computed_data.results.wavefunction.total_molecular_energy[i]
        homo_ind = computed_data.results.wavefunction.homo_indexes[i]
        MO_energies = computed_data.results.wavefunction.MO_energies[i]
        MO_labels = computed_data.derived.wavefunction.MO_analysis.MO_labels[i]
        MO_indexes = computed_data.derived.wavefunction.MO_analysis.MO_indexes[i]

        if (
            total_energy is None 
            or homo_ind is None 
            or MO_energies is None
            or MO_labels is None
            or MO_indexes is None
        ):
            continue

        key = serialize_energies_data(total_energy, homo_ind, MO_energies, MO_labels, MO_indexes)
        seen[key].append(i)

    # Find the group of data tha contains the reference
    groups = list(seen.values())
    ref_group = None
    other_groups = []

    # Put the reference first in its own data group
    for group in groups:
        if ref_id in group:
            group.remove(ref_id)
            group.insert(0, ref_id)
            ref_group = group
        else:
            other_groups.append(group)

    # One table for all the models: the six tables of the acetone test series (one per group
    # of models) repeated the same rows; the group of the reference comes first.
    ordered = ([ref_group] if ref_group is not None else []) + other_groups
    if not ordered:
        return False
    labels = [label.replace("_alpha", "").upper()
              for label in computed_data.derived.wavefunction.MO_analysis.MO_labels[ordered[0][0]][0]]
    n = len(labels)
    open_shell = any(len(computed_data.results.wavefunction.homo_indexes[g[0]]) == 2 for g in ordered)
    # tabular, not longtable: the header was left alone at the bottom of a page
    section.append(NoEscape(r'\begin{footnotesize}'))
    with section.create(Tabular(RAGGED + "{0.2\\textwidth} " + CENTERED + "{0.13\\textwidth} c c" + " c" * n)) as table:
        table.add_row([MultiColumn(4 + n, align='l',
                                   data="Table. Total energy (hartrees) and orbital energies (eV)"
                                        + (" (one row per spin for the unrestricted calculations)"
                                           if open_shell else ""))])
        table.add_hline()
        table.add_row(["Models", "Calculation", "Energy", "HOMO no."] + labels)
        table.add_hline()
        for group in ordered:
            rows = energies_rows(computed_data, group)
            for row in rows:
                # groups computed with other orbital lists (should not happen) are padded
                table.add_row((row + [""] * n)[:4 + n])
        table.add_hline()
    section.append(NoEscape(r'\end{footnotesize}'))
    return True


def render_ref_MO_diagram(section, output_dir, computed_data: ComputedData, report_type):
    
    # figure with MO not available in text report type
    """Pictures of the requested orbitals of the reference (two views when the molecule needs them).

    Parameters
    ----------
    section : pylatex Section
    output_dir : str
        Folder of the pictures (with the final separator).
    computed_data : ComputedData
    report_type : str
        full, si or text: most pictures are only shown in full reports.


    Returns
    -------
    bool
        Whether something was added to the section.
    """
    if report_type == 'text':
        return False
    
    ref_id = computed_data.metadata.ref_log_file_idx
    homo_ind = computed_data.results.wavefunction.homo_indexes[ref_id]
    MO_labels = computed_data.derived.wavefunction.MO_analysis.MO_labels[ref_id]
    
    # Skip if no data to display
    if (
        homo_ind == None
        or MO_labels == None
        or len(MO_labels[0]) == 0
    ):
        return False
    
    # All the requested orbitals in one grid of two pictures per row (figure_grid), instead of
    # one figure per orbital
    cells = []
    # Unrestricted calculation: alpha and beta orbitals of each label side by side
    if len(homo_ind) == 2:
        for label_alpha, label_beta in zip(MO_labels[0], MO_labels[1]):
            nomPng_alpha = f"img-log{ref_id + 1}-MO-{label_alpha}.png"
            nomPng_beta  = f"img-log{ref_id + 1}-MO-{label_beta}.png"
            base_label = label_alpha.replace("_alpha", "").upper()
            pngs = [nomPng_alpha, nomPng_beta]
            present = [os.path.isfile(os.path.join(output_dir, png)) for png in pngs]
            if all(present):
                cells.append((pngs, escape_latex(base_label)))
            elif present[0]:
                cells.append(([nomPng_alpha], escape_latex(base_label) + r" ($\alpha$)"))
            elif present[1]:
                cells.append(([nomPng_beta], escape_latex(base_label) + r" ($\beta$)"))
    # Restricted calculation: one picture per orbital, or two views
    else:
        for label in MO_labels[0]:
            nomPng = f"img-log{ref_id + 1}-MO-{label}.png"
            nomPng2 = f"img-log{ref_id + 1}-MO-{label}_cam2.png"
            if os.path.isfile(os.path.join(output_dir, nomPng)):
                pngs = [nomPng] + ([nomPng2] if os.path.isfile(os.path.join(output_dir, nomPng2)) else [])
                cells.append((pngs, escape_latex(label.upper())))
    if not cells:
        return False
    two = any(len(pngs) == 2 for pngs, _ in cells)
    if len(homo_ind) == 2:
        views = r" ($\alpha$ left, $\beta$ right)" if two else ""
    else:
        views = " (two points of view)" if two else ""
    # Two orbitals per row also with two pictures per orbital (alpha/beta or two views): one
    # orbital per row put 4 orbitals on 2-3 pages (O2, HCFClBr with camera_views: 2, 26/11/2026)
    figure_grid(section, cells, 2, max_width=None if two else 0.4, escapeCaption=False, caption=
                f"Molecular orbitals of the reference{views}. "
                f"{escape_latex(color_of('MO', 'positive', True))} and {escape_latex(color_of('MO', 'negative'))} "
                f"show opposite orbital phases.{degenerate_note(computed_data, ref_id)} "
                f"(Source: {escape_latex(computed_data.metadata.log_files[ref_id])})")
    return True


def degenerate_note(computed_data: ComputedData, ref_id):
    """Sentence naming the shown orbitals that are degenerate (compute.degenerate_partners) and
    how their pairs are drawn (compute.show_degenerate_pairs); empty when there is none."""
    from quchemreport.data_enrichment.compute import degenerate_partners
    energies = computed_data.results.wavefunction.MO_energies[ref_id]
    homo_ind = computed_data.results.wavefunction.homo_indexes[ref_id]
    indexes = computed_data.derived.wavefunction.MO_analysis.MO_indexes[ref_id]
    if not energies or not indexes or not homo_ind:
        return ""
    groups = []
    for spin, idx_list in enumerate(indexes):
        for i in idx_list:
            partners = degenerate_partners(energies[spin], i)
            if not partners:
                continue
            group = sorted([i] + partners)
            names = "/".join(_orbital_name(k, homo_ind[spin]) for k in group)
            names += {0: r" ($\alpha$)", 1: r" ($\beta$)"}[spin] if len(indexes) == 2 else ""
            if names not in groups:
                groups.append(names)
    if not groups:
        return ""
    return (" Degenerate orbitals: " + ", ".join(groups) + ". Any combination of degenerate orbitals "
            "is as valid: the pictures of a degenerate pair show the two combinations seen alike from "
            "this point of view.")


def _orbital_name(index, homo):
    """HOMO, HOMO-1, LUMO, LUMO+2... of the 0-based orbital index."""
    if index <= homo:
        return "HOMO" if index == homo else f"HOMO-{homo - index}"
    return "LUMO" if index == homo + 1 else f"LUMO+{index - homo - 1}"


def render_ref_MEP_maps(section, output_dir, computed_data: ComputedData, report_type):
    # MEP map figures is only printed in full type report 
    """MEP figure of the reference (picture, colour bar, caption giving the surface and the
    colour scale) followed by the table of its extrema (full reports only).

    Parameters
    ----------
    section : pylatex Section
    output_dir : str
        Folder of the pictures (with the final separator).
    computed_data : ComputedData
    report_type : str
        full, si or text: most pictures are only shown in full reports.


    Returns
    -------
    bool
        Whether something was added to the section.
    """
    if report_type != "full":
        return False
    
    ref_id = computed_data.metadata.ref_log_file_idx
    maps = computed_data.derived.wavefunction.MEP_maps
    isovalue = maps.surface_isovalue or 0.002
    extrema = maps.surface_extrema or []
    
    # MEP figure (visu_mayavi.viz_Potential): fixed, symmetric colour scale, with its colour bar
    nomPng = f"img-log{ref_id + 1}-MEP.png"
    nomBar = f"img-log{ref_id + 1}-MEP_colorbar.png"
    color_range = maps.color_range
    caption = ("Molecular Electrostatic Potential (MEP) mapped on the electron density isosurface "
               f"({isovalue:g} e-/bohr3). Red: negative MEP (electron-rich regions), blue: positive MEP "
               "(electron-poor regions), green: close to zero")
    if color_range:
        caption += (f"; fixed colour scale from {-color_range * Eh_to_Kcal_mol:.0f} to "
                    f"+{color_range * Eh_to_Kcal_mol:.0f} kcal/mol ({-color_range:g} to +{color_range:g} a.u.), "
                    "the same for the compounds of the same charge")
    caption += ". "
    weakest = f"{mep_surface.PROMINENCE_MIN * Eh_to_Kcal_mol:.1f} kcal/mol"
    if extrema:
        caption += ("Numbered spheres: local minima (white) and maxima (black) of the MEP on this surface "
                    f"(see the table below; extrema standing out by less than {weakest} from their "
                    "surroundings are not shown; the numbers on the hidden side are not written).")
    elif maps.surface_extrema is not None:
        caption += f"No local extremum of the MEP standing out by more than {weakest} on this surface."
    nomPng2 = f"img-log{ref_id + 1}-MEP_cam2.png"
    two = os.path.isfile(output_dir + nomPng2)
    if two:
        caption += " Two points of view."
    if os.path.isfile(output_dir + nomPng):
        section.append(NoEscape(r'\begin{figure}[H]'))
        section.append(NoEscape(r'\begin{center}'))
        if two:
            trim = trim_option([nomPng, nomPng2])
            section.append(NoEscape(r'\includegraphics[width=8.5cm' + trim + ']{' + nomPng + r'}\hspace{2mm}'
                                    r'\includegraphics[width=8.5cm' + trim + ']{' + nomPng2 + '}'))
        else:
            section.append(NoEscape(r'\includegraphics[width=10cm' + trim_option([nomPng]) + ']{' + nomPng + '}'))
        if os.path.isfile(output_dir + nomBar):
            section.append(NoEscape(r'\\'))
            section.append(NoEscape(r'\includegraphics[width=9cm]{' + nomBar + '}'))
        section.append(NoEscape(r'\end{center}'))
        section.append(NoEscape(r'\vspace{-5mm}'))
        section.append(NoEscape(r'\caption{' + escape_latex(caption) + '}'))
        section.append(NoEscape(r'\end{figure}'))
        render_MEP_extrema_table(section, computed_data, isovalue, extrema)
        return True

    return False


def render_MEP_extrema_table(section, computed_data: ComputedData, isovalue, extrema):
    """Table of the local minima and maxima of the MEP on the surface, numbered as the markers
    of the picture: type, value in kcal/mol and a.u., nearest atom and its distance;
    V_S,min and V_S,max in bold.

    Parameters
    ----------
    section : pylatex Section
    computed_data : ComputedData
    isovalue : float
        Density of the surface (e-/bohr3).
    extrema : list of dict
        mep_surface.mep_surface_extrema.
    """
    if not extrema:
        return
    ref_id = computed_data.metadata.ref_log_file_idx
    atoms_Z = computed_data.molecule.atoms_Z[ref_id]
    log_file = computed_data.metadata.log_files[ref_id]
    table_spec = "p{0.06\\textwidth}" + "".join(CENTERED + f"{{{w}\\textwidth}}" for w in (0.12, 0.16, 0.16, 0.2, 0.16))
    with section.create(LongTable(table_spec)) as table:
        table.add_row([MultiColumn(6, align=NoEscape('p{0.95\\textwidth}'), data=NoEscape(
            f"Table. Local minima and maxima of the MEP on the {isovalue:g} e-/bohr3 isosurface "
            r"($V_\mathrm{S,min}$ and $V_\mathrm{S,max}$, the most negative and most positive values, in bold). "
            f"Source: {escape_latex(log_file)}"))])
        table.add_hline()
        table.add_row(["N°", "Type", NoEscape(r"$V_\mathrm{S}$ (kcal/mol)"), NoEscape(r"$V_\mathrm{S}$ (a.u.)"),
                       "Nearest atom (N°)", "Distance (pm)"])
        table.add_hline()
        first = {"min": extrema[0] if extrema[0]["kind"] == "min" else None,
                 "max": next((e for e in extrema if e["kind"] == "max"), None)}
        for number, ext in enumerate(extrema, start=1):
            atom = f"{PeriodicTable().element[atoms_Z[ext['atom']]]} (N°{ext['atom'] + 1})"
            kcal = f"{ext['value'] * Eh_to_Kcal_mol:+.1f}"
            au = f"{ext['value']:+.4f}"
            if ext is first[ext["kind"]]:
                kcal, au = bold(kcal), bold(au)
            kind = ("minimum" if ext["kind"] == "min" else "maximum") + (" (ring)" if ext.get("ring") else "")
            table.add_row([str(number), kind, kcal, au,
                           atom, f"{ext['distance'] * a0_to_pm:.0f}"])
        table.add_hline()
        if any(ext.get("ring") for ext in extrema):
            table.add_row([MultiColumn(6, align='l', data=NoEscape(
                r"\footnotesize Ring: the same value all around the axis of the linear molecule "
                r"(marker at the top of the picture)."))])


def render_ref_population_analysis(section, computed_data: ComputedData, report_type):
    # Only display when report type is full
    """Statistics of the atomic charges of the reference and table of the atoms whose charge
    deviates by more than one standard deviation (full reports only).

    Returns
    -------
    bool
        Whether something was added to the section.
    """
    if report_type != "full":
        return False

    ref_id = computed_data.metadata.ref_log_file_idx
    log_file = computed_data.metadata.log_files[ref_id]
    atoms_Z = computed_data.molecule.atoms_Z[ref_id]
    mulliken = np.array(computed_data.results.wavefunction.Mulliken_partial_charges or [])
    hirshfeld = np.array(computed_data.results.wavefunction.Hirshfeld_partial_charges or [])
    cm5 = np.array(computed_data.results.wavefunction.CM5_partial_charges or [])
    derived = computed_data.derived.wavefunction.pop_analysis
    indices = computed_data.derived.wavefunction.pop_analysis.indices
    
    stats = {
        "Mulliken": 
            (derived.Mulliken_mean, derived.Mulliken_std) if derived.Mulliken_mean is not None and derived.Mulliken_std is not None else None,
        "Hirshfeld": 
            (derived.Hirshfeld_mean, derived.Hirshfeld_std) if derived.Hirshfeld_mean is not None and derived.Hirshfeld_std is not None else None,
        "CM5": 
            (derived.CM5_partial_mean, derived.CM5_partial_std) if derived.CM5_partial_mean is not None and derived.CM5_partial_std is not None else None,
    }

    # Skip if no data to display
    stats = {k: v for k, v in stats.items() if v is not None}
    if not stats:
        return False

    # Skip if no data to display
    if not indices:
        return False

    # Display mean and std for each method available
    with section.create(LongTable("p{0.6\\textwidth}" + CENTERED + "{0.15\\textwidth}" + CENTERED + "{0.15\\textwidth}")) as stats_table:
        stats_table.add_row(
            [MultiColumn(3, align='l', data=f"Population analysis: atomic charge statistics. (Source: {log_file})")])
        for method, (mean, std) in stats.items():
            stats_table.add_row([
                f"Mean {method} atomic charge and standard deviation",
                f"{fmt_signed(mean, 4)} e",
                f"{std:.4f} e"
            ])

    # Display atoms with significant charges (exceeding ±1σ in at least one method)
    # Show all methods' charges per atom, bolding values exceeding their method-specific threshold
    n_cols = 1 + len(stats)
    col_format = "p{0.15\\textwidth}" + (CENTERED + "{0.15\\textwidth}") * (n_cols - 1)
    with section.create(LongTable(col_format)) as pop_analysis_table:
        pop_analysis_table.add_row(
            [MultiColumn(n_cols, align='l', data="Table. Population analysis: atoms with significant charge deviations.")]
        )
        # Legend for values in bold if multipe methods displayed
        if len(stats) != 1:
            pop_analysis_table.add_row(
                [MultiColumn(n_cols, align='l', data="Values in bold lie outside the ±1 standard deviation range for their respective method.")]
            )
        pop_analysis_table.add_row(
            [MultiColumn(n_cols, align='l', data=f"Source: {log_file}")]
        )
        pop_analysis_table.add_hline()
        pop_analysis_table.add_row(["Atom (N°)"] + [f"{m} charge" for m in stats])
        pop_analysis_table.add_hline()

        for idx in indices:
            atom_label = f"{PeriodicTable().element[atoms_Z[idx]]} (N°{idx+1})"
            row = [atom_label]

            for method in stats:
                charges = {
                    "Mulliken": mulliken,
                    "Hirshfeld": hirshfeld,
                    "CM5": cm5
                }[method]

                val = charges[idx]
                mean, std = stats[method]
                
                # If multiple methods displayed, draw in bold value outisde of the range
                if (
                    len(stats) != 1
                    and 
                    (
                        val < mean - std
                        or val > mean + std
                    )
                ):
                    val_str = NoEscape(r'\textbf{' + f"{val:+.3f}" + r'}')
                else:
                    val_str = f"{val:+.3f}"

                row.append(val_str)

            pop_analysis_table.add_row(row)
        pop_analysis_table.add_hline()
        
    return True


def fill_wavefunction_section(doc, output_dir, computed_data: ComputedData, report_type, config_output: OutputInclude):
    
    """Wavefunction section: energies, orbitals, MEP map, population analysis."""
    section = Section('WAVEFUNCTION')
    
    added_energies = False
    added_ref_MO_diagram = False
    added_ref_MEP_maps = False
    added_ref_population_analysis = False
    
    # Render energies tables for each log file when available (reference first)
    if config_output.wavefunction.MO_analysis.energies:
        added_energies = render_energies(section, computed_data)
    
    # Render the MO_diagram of the reference data
    if config_output.wavefunction.MO_analysis.MO_diagrams:
        added_ref_MO_diagram = render_ref_MO_diagram(section, output_dir, computed_data, report_type)

    # Render the MEP_maps of the reference data
    if config_output.wavefunction.MEP_maps :
        added_ref_MEP_maps = render_ref_MEP_maps(section, output_dir, computed_data, report_type)
    
    # Render population population analysis on the reference
    if config_output.wavefunction.population_analysis :
        added_ref_population_analysis = render_ref_population_analysis(section, computed_data, report_type)
    
    if (
        added_energies
        or added_ref_MO_diagram
        or added_ref_MEP_maps
        or added_ref_population_analysis
    ):
        doc.append(section)


def render_thermo_data(section, computed_data: ComputedData):
    
    """Table of the thermochemistry of each frequency calculation: imaginary frequencies, zero-point correction, sums of the electronic energy with the zero-point and thermal energies, enthalpy, Gibbs free energy and entropy at the temperature of the calculation.

    Returns
    -------
    bool
        Whether something was added to the section.
    """
    ref_id = computed_data.metadata.ref_log_file_idx
    log_files = computed_data.metadata.log_files
    all_temperatures = computed_data.comp_details.freq.temperature
    all_vibrational_int = computed_data.results.freq.vibrational_int
    all_zero_point_energy = computed_data.results.freq.zero_point_energy
    all_electronic_thermal_energy = computed_data.results.freq.electronic_thermal_energy
    all_enthalpy = computed_data.results.freq.enthalpy
    all_free_energy = computed_data.results.freq.free_energy
    all_entropy = computed_data.results.freq.entropy
    
    isContentAdded = False
    
    # Display reference first if available
    indices = list(range( len(log_files) ))
    if ref_id in indices:
        indices.remove(ref_id)
        indices.insert(0, ref_id)
    
    # Display thermo data for each logfile when available
    for i in indices:
        
        if all_temperatures[i] != None:
            rtemper = all_temperatures[i]
        else:
            rtemper = []
            
        if all_vibrational_int[i] != None:
            vibrational_int = np.array(all_vibrational_int[i])
        else:
            vibrational_int = []
        
        # TODO : TO CHECK : Unused nb_negatives value
        # if len(vibrational_int) == 0 :
        #     vibrational_int = []
        # else :  
        #     # Print number of negative frequencies   
        #     nb_negatives = np.sum(vibrational_freq < 0, axis=0)
        
        log_file = log_files[i]
        zero_point_energy = all_zero_point_energy[i]
        electronic_thermal_energy = all_electronic_thermal_energy[i]
        enthalpy = all_enthalpy[i]
        free_energy = all_free_energy[i]
        entropy = all_entropy[i]
        
        # Skip if no data to display
        if (
            zero_point_energy == None
            and electronic_thermal_energy == None
            and enthalpy == None
            and free_energy == None
            and entropy == None
        ):
            continue
        
        if (isContentAdded == False):
            isContentAdded = True
        
        with section.create(LongTable("p{0.6\\textwidth} " + CENTERED + "{0.15\\textwidth} " + CENTERED + "{0.15\\textwidth}")) as thermo_data_table :
            thermo_data_table.add_row(["Frequency and Thermochemistry specific results", " " , " "])
            thermo_data_table.add_row([f"Source: {log_file}", " ", " "])
            
            imag = imaginary_frequencies(computed_data.results.freq.vibrational_freq[i])
            if imag:
                thermo_data_table.add_row([bold("Imaginary frequencies (not a minimum)"),
                                           bold(format_imaginary(imag) + " cm-1"), " "])
            
            if (len(vibrational_int) != 0) and (rtemper != "N/A") :
                if zero_point_energy != None:
                    # cclib gives the zero-point correction; the sum uses the SCF energy of the same log
                    thermo_data_table.add_row(['Zero-point correction',
                                        "%.5f Hartrees" % zero_point_energy, " "])
                    scf_energy = computed_data.results.wavefunction.total_molecular_energy[i]
                    if scf_energy is not None:
                        thermo_data_table.add_row(['Sum of electronic and zero-point energies',
                                        "%.5f Hartrees" % (scf_energy + zero_point_energy), " "])
                if electronic_thermal_energy != None:
                    thermo_data_table.add_row(["Sum of electronic and thermal energies at  %.2f K"  % rtemper, 
                                        "%.5f Hartrees" % electronic_thermal_energy, " "])
                if enthalpy != None:
                    thermo_data_table.add_row(["Enthalpy at %.2f K" % rtemper,
                                        "%.5f Hartrees" % enthalpy, " "])
                if free_energy != None :
                    thermo_data_table.add_row(["Gibbs free energy at %.2f K" % rtemper,
                                        "%.5f Hartrees" % free_energy, " "])
                if entropy != None :
                    # cclib entropy in hartree/K -> cal/(mol K)
                    thermo_data_table.add_row(["Entropy at %.2f K" % rtemper,
                                        NoEscape(r"%.3f cal mol$^{-1}$ K$^{-1}$" % (entropy * Eh_to_Kcal_mol * 1000.0)), " "])
                thermo_data_table.add_row([" ", " " , " "])
    
    return isContentAdded


def render_IR_spectrum(section, computed_data: ComputedData):
    
    """Table of the IR vibrations more intense than THRESHOLD_IR_INTENSITIES (km/mol): frequency, intensity, symmetry.

    Returns
    -------
    bool
        Whether something was added to the section.
    """
    ref_id = computed_data.metadata.ref_log_file_idx
    log_files = computed_data.metadata.log_files
    all_vibrational_int = computed_data.results.freq.vibrational_int
    all_vibrational_freq = computed_data.results.freq.vibrational_freq
    all_vibrational_sym = computed_data.results.freq.vibrational_sym
    
    isContentAdded = False
    
    # Display reference first if available
    indices = list(range( len(log_files) ))
    if ref_id in indices:
        indices.remove(ref_id)
        indices.insert(0, ref_id)
    
    # Display thermo data for each logfile when available
    for i in indices:
        
        if all_vibrational_int[i] != None:
            vibrational_int = np.array(all_vibrational_int[i])
        else:
            vibrational_int = []
            
        if all_vibrational_freq[i] != None:
            vibrational_freq = np.array(all_vibrational_freq[i])
        else:
            vibrational_freq = []   
            
        if all_vibrational_sym[i] != None:
            vibrational_sym = np.array(all_vibrational_sym[i])
        else:
            vibrational_sym = np.array(["N/A" for _ in range(len(vibrational_int))])
            
        log_file = log_files[i]
        if len(vibrational_freq) == 0 or len(vibrational_int) != len(vibrational_freq):
            continue
        isContentAdded = True
        all_freq, all_int = vibrational_freq, vibrational_int
        vib_filter = vibrational_int > THRESHOLD_IR_INTENSITIES
        vib_order = np.argsort(vibrational_freq[vib_filter])[::-1]
        vibrational_int = vibrational_int[vib_filter][vib_order]
        vibrational_freq = vibrational_freq[vib_filter][vib_order]
        vibrational_sym = vibrational_sym[vib_filter][vib_order] if len(vibrational_sym) == len(all_freq) else []

        if len(vibrational_freq) == 0:
            # No intense vibration: say so instead of leaving the section silent (O2, 26/11/2026:
            # IR-inactive, the report showed nothing at all)
            freqs = ", ".join("%d" % f for f in sorted(all_freq, reverse=True)[:6])
            if np.max(all_int) < IR_INACTIVE:
                text = (f"No IR-active vibration (all intensities below {IR_INACTIVE} km/mol). "
                        f"Vibrations: {freqs}{', ...' if len(all_freq) > 6 else ''} cm$^{{-1}}$.")
            else:
                k = int(np.argmax(all_int))
                text = (f"No vibration more intense than {THRESHOLD_IR_INTENSITIES:.0f} km/mol; the most intense: "
                        f"{all_freq[k]:.0f} cm$^{{-1}}$ ({all_int[k]:.1f} km/mol).")
            section.append(NoEscape(text + f" (Source: {escape_latex(log_file)})"))
            continue

        # The symmetry column only when the program gives it (ORCA: none)
        has_sym = any(str(x) not in ("N/A", "", "None") for x in vibrational_sym)
        section.append(NoEscape(r'\begin{center}'))
        with section.create(Tabular('rc' + ('c' if has_sym else ''))) as ir_spectrum_table:
            ncol = 3 if has_sym else 2
            ir_spectrum_table.add_row([MultiColumn(ncol, align='c', data=NoEscape(
                f"Table. Most intense (> {THRESHOLD_IR_INTENSITIES:.0f} km/mol) vibrations."))])
            ir_spectrum_table.add_row([MultiColumn(ncol, align='c', data=f"Source: {log_file}")])
            ir_spectrum_table.add_hline()
            ir_spectrum_table.add_row([NoEscape(r"Wavenumber (cm$^{-1}$)"), "Intensity (km/mol)"]
                                      + (["Symmetry"] if has_sym else []))
            ir_spectrum_table.add_hline()
            for k in range(len(vibrational_freq)):
                ir_spectrum_table.add_row(["%d" % vibrational_freq[k], "%d" % vibrational_int[k]]
                                          + ([vibrational_sym[k]] if has_sym else []))
            ir_spectrum_table.add_hline()
        section.append(NoEscape(r'\end{center}'))

    return isContentAdded


def fill_frequencies_section(doc, computed_data: ComputedData, config_output: OutputInclude):
    
    """Frequencies section: thermochemistry and IR data."""
    section = Section('FREQUENCIES')
    
    added_thermo_data = False
    added_IR_spectrum = False
        
    # Render thermo data for each log file when available (reference first)
    if config_output.frequencies.thermo_data:
        added_thermo_data = render_thermo_data(section, computed_data)
    
    # Render IR spectrum for each log file when available (reference first)
    if config_output.frequencies.IR_spectrum:
        added_IR_spectrum = render_IR_spectrum(section, computed_data)

    if (
        added_thermo_data
        or added_IR_spectrum
    ):
        doc.append(section)

def render_CDFT_global_indices(section, computed_data: ComputedData, report_type):
    # CDFT Indices table only in full report
    """Table of the conceptual DFT global indices (A, I, electronegativity, hardness, electrophilicity, DeltaN).

    Returns
    -------
    bool
        Whether something was added to the section.
    """
    if report_type != 'full': 
        return False
    
    A = computed_data.results.wavefunction.A
    I = computed_data.results.wavefunction.I
    Khi = computed_data.results.wavefunction.Khi
    Eta = computed_data.results.wavefunction.Eta
    Omega = computed_data.results.wavefunction.Omega
    DeltaN = computed_data.results.wavefunction.DeltaN
    
    # Skip if no data to display
    if (
        A == None
        and I == None
        and Khi == None
        and Eta == None
        and Omega == None
        and DeltaN == None
    ):
        return False
    
    # One titled table: "CDFT indices:" was repeated in front of every row
    rows = [("Electron affinity (A)", A, "%.3f eV"), ("Ionization potential (I)", I, "%.3f eV"),
            ("Electronegativity ($\\chi$)", Khi, "%.3f eV"), ("Hardness ($\\eta$)", Eta, "%.3f eV"),
            ("Electrophilicity ($\\omega$)", Omega, "%.3f eV"),
            ("Electron flow ($\\Delta N$)", DeltaN, "%.4f e")]
    section.append(NoEscape(r'\begin{center}'))
    with section.create(Tabular("l c")) as CDFT_indicies_table:
        CDFT_indicies_table.add_row([MultiColumn(2, align='c', data="Table. Conceptual DFT global indices.")])
        CDFT_indicies_table.add_hline()
        for label, value, form in rows:
            if value is not None:
                CDFT_indicies_table.add_row([NoEscape(label), form % value])
        CDFT_indicies_table.add_hline()
    section.append(NoEscape(r'\end{center}'))
    return True


# Molecules up to this number of atoms list every atom in the condensed Fukui table; larger
# ones only the atoms whose dual descriptor is more than one standard deviation from the mean.
# Why (26/11/2026): the outlier rule alone gave an empty table for symmetric molecules (O2: both
# atoms equal) and was not explained in the caption.
FUKUI_ALL_ATOMS = 20


def fukui_table_atoms(n_atoms, fplus, fminus, fdual):
    """Atom indexes of the condensed Fukui table and whether all the atoms are listed.

    Sorted by dual descriptor when there is one. Every atom up to FUKUI_ALL_ATOMS atoms (or when
    no atom stands out), otherwise the atoms whose dual descriptor departs from the mean by more
    than one standard deviation."""
    has_value = [i for i in range(n_atoms)
                 if (i < len(fplus) and fplus[i] is not None) or (i < len(fminus) and fminus[i] is not None)
                 or (i < len(fdual) and fdual[i] is not None)]
    if len(fdual) == 0:
        return has_value, True
    fd = np.array([np.nan if v is None else v for v in fdual], dtype=float)
    order = [i for i in np.argsort(fd) if i in has_value]
    if n_atoms <= FUKUI_ALL_ATOMS:
        return order, True
    mean, dev = np.nanmean(fd), np.nanstd(fd)
    picked = [i for i in order if abs(fd[i] - mean) > dev]
    return (picked, False) if picked else (order, True)


def render_fukui_table(doc, atoms_Z, fplus, fminus, fdual, title):
    """Table of condensed Fukui functions (f+, f-, dual descriptor) per atom, for one charge
    model (Mulliken or Hirshfeld); atoms chosen by fukui_table_atoms.
    """
    if len(fplus) == 0 and len(fminus) == 0:
        return
    ind, all_atoms = fukui_table_atoms(len(atoms_Z), fplus, fminus, fdual)
    if not ind:
        return
    which = "" if all_atoms else (" Atoms whose dual descriptor departs from the mean by more than"
                                  " one standard deviation.")

    doc.append(NoEscape(r'\begin{center}'))
    # No empty p{0cm} side columns: their width made the header cells overlap
    with doc.create(Tabular('lccc')) as fukui_table:
        row_cells = [MultiColumn(4, align='c', data=f"Table. Condensed Fukui functions based on {title} charges.")]
        fukui_table.add_row(row_cells)
        fukui_table.add_hline()
        fukui_table.add_row(["Atom", NoEscape(r"Dual descriptor $f^+ - f^-$"),
                             NoEscape(r"Electrophilicity $f^+$"), NoEscape(r"Nucleophilicity $f^-$")])
        fukui_table.add_hline()
        for i in ind:
            values = [v[i] if i < len(v) else None for v in (fdual, fplus, fminus)]
            atom_label = "%s%d" % (PeriodicTable().element[atoms_Z[i]], i + 1)
            fukui_table.add_row([atom_label] + [fmt_signed(v, 2) if v is not None else "" for v in values])
        fukui_table.add_hline()
        if which:
            fukui_table.add_row([MultiColumn(4, align='c', data=NoEscape(r"\footnotesize" + which))])
    doc.append(NoEscape(r'\end{center}'))


def render_condensed_fukui_functions(section, computed_data: ComputedData):

    """Tables of condensed Fukui functions for the available charge models.

    Returns
    -------
    bool
        Whether something was added to the section.
    """
    def safe(val): return val if val is not None else [] 
    
    ref_id = computed_data.metadata.ref_log_file_idx
    atoms_Z = computed_data.molecule.atoms_Z[ref_id]

    fplus_lambda_hirshfeld = safe(computed_data.results.wavefunction.fplus_lambda_hirshfeld)
    fminus_lambda_hirshfeld = safe(computed_data.results.wavefunction.fminus_lambda_hirshfeld)
    fdual_lambda_hirshfeld = safe(computed_data.results.wavefunction.fdual_lambda_hirshfeld)

    fplus_lambda_mulliken = safe(computed_data.results.wavefunction.fplus_lambda_mulliken)
    fminus_lambda_mulliken = safe(computed_data.results.wavefunction.fminus_lambda_mulliken)
    fdual_lambda_mulliken = safe(computed_data.results.wavefunction.fdual_lambda_mulliken)
    
    # Skip if no data
    if (
        len(fplus_lambda_hirshfeld) == 0
        and len(fminus_lambda_hirshfeld) == 0
        and len(fplus_lambda_mulliken) == 0
        and len(fminus_lambda_mulliken) == 0
    ):
        return False

    render_fukui_table(
        section,
        atoms_Z,
        fplus_lambda_hirshfeld,
        fminus_lambda_hirshfeld,
        fdual_lambda_hirshfeld,
        "Hirshfeld"
    )
    
    render_fukui_table(
        section,
        atoms_Z,
        fplus_lambda_mulliken,
        fminus_lambda_mulliken,
        fdual_lambda_mulliken,
        "Mulliken"
    )
    
    return True


def render_fukui_functions(section, output_dir, computed_data: ComputedData, report_type):
    # Figures of that are only printed in full type report 
    """Pictures of the Fukui functions and of the dual descriptor in one grid: F+, F- and
    the dual descriptor side by side when the molecule needs only one view, otherwise one row
    each with its two views.

    Parameters
    ----------
    section : pylatex Section
    output_dir : str
        Folder of the pictures (with the final separator).
    computed_data : ComputedData
    report_type : str
        full, si or text: most pictures are only shown in full reports.


    Returns
    -------
    bool
        Whether something was added to the section.
    """
    if report_type != 'full':
        return False
    
    isContentAdded = False
        
    # Figures of the Fukui functions and of the dual descriptor, side by side in one grid.
    # Why (27/11/2026, Thomas Cauchy): F+ and F- were side by side and the dual descriptor in a
    # second figure; the three are read together. Three cells per row are 59 mm wide, cropped
    # to the molecule (trim_option), which stays readable for large molecules since these
    # pictures carry no labels. The second view (_cam2), when the molecule needs one
    # (visualization_engine.camera), puts each function on its own row with its two views.
    has = lambda name: os.path.isfile(output_dir + name)
    cells = []
    for name, label in (("img-fukui-SP_plus.png", "F+ (electrophilic sites)"),
                        ("img-fukui-SP_minus.png", "F- (nucleophilic sites)"),
                        ("img-Fdual.png", "Dual descriptor")):
        if has(name):
            name2 = name.replace(".png", "_cam2.png")
            cells.append(([name] + ([name2] if has(name2) else []), escape_latex(label)))
    if cells:
        two_views = any(len(pngs) == 2 for pngs, _ in cells)
        positive, negative = color_of('fukui', 'positive'), color_of('fukui', 'negative')
        figure_grid(section, cells, 1 if two_views else 3, max_width=0.7 if two_views else None,
                    caption=("Fukui functions and dual descriptor"
                             + (", from two points of view" if two_views else "") + ". "
                             f"F+ and F-: the {positive} regions are the most electrophilic (F+) and the "
                             f"most nucleophilic (F-) sites. Dual descriptor: electrophilic and nucleophilic "
                             f"regions correspond to {positive} and {negative} surfaces."))
        isContentAdded = True
    
    return isContentAdded


def fill_reactivity_section(doc, output_dir, computed_data: ComputedData, report_type, config_output: OutputInclude):
    
    """Reactivity section: conceptual DFT indices, condensed and 3D Fukui functions."""
    section = Section('REACTIVITY')
    
    added_CDFT_global_indices = False
    added_condensed_fukui_functions = False
    added_fukui_functions = False
    
    # Render CDFT global indicies
    if config_output.reactivity.CDFT_indices:
        added_CDFT_global_indices = render_CDFT_global_indices(section, computed_data, report_type)
    
    if config_output.reactivity.fukui_functions:
        # Render condensed fukui functions
        added_condensed_fukui_functions = render_condensed_fukui_functions(section, computed_data)
    
        # Render fukui functions
        added_fukui_functions = render_fukui_functions(section, output_dir, computed_data, report_type)
    
    if (
        added_CDFT_global_indices
        or added_condensed_fukui_functions
        or added_fukui_functions
    ):
        doc.append(section)


def fmt_signed(x, digits):
    """x with the given decimals, without the "-0.0000" of tiny negative values."""
    text = f"{x:.{digits}f}"
    return text.lstrip("-") if float(text) == 0 else text


def frontier_label(mo, homo):
    """'H', 'H-1', 'L', 'L+2'... of the 0-based orbital mo, homo being the 0-based HOMO."""
    if mo <= homo:
        return "H" if mo == homo else f"H-{homo - mo}"
    return "L" if mo == homo + 1 else f"L+{mo - homo - 1}"


def build_CIS_string(transitions, homo_ind, filter_threshold=5):
    """Main excitations of a state as LaTeX, e.g. ``H$\to$L (95)``: orbitals named from the
    HOMO (H) and LUMO (L) of their spin, alpha/beta for unrestricted calculations, and weight
    in % (C^2 x 100), for the weights of at least filter_threshold %.

    The orbital numbers (16->17) of the former tables had to be compared with the HOMO
    number of the wavefunction section; the orbital number is kept in the table caption.
    """
    parts = []
    unrestricted = len(homo_ind) == 2
    for ST in transitions:
        coeff = int(ST[2] ** 2 * 100)
        if coeff >= filter_threshold:
            mo_i, spin_i = ST[0]
            mo_f, spin_f = ST[1]
            spin = {0: r"$\alpha$", 1: r"$\beta$"} if unrestricted else {0: "", 1: ""}
            homo_i = homo_ind[spin_i] if unrestricted else homo_ind[0]
            homo_f = homo_ind[spin_f] if unrestricted else homo_ind[0]
            parts.append(f"{frontier_label(mo_i, homo_i)}{spin.get(spin_i, '')}$\\to$"
                         f"{frontier_label(mo_f, homo_f)}{spin.get(spin_f, '')} ({coeff})")
    return NoEscape(", ".join(parts))


# Spin contamination: <S^2> of an excited state of an open-shell reference farther than this
# from S(S+1) is shown in bold (0.1, the usual warning level for UKS ground states).
S2_CONTAMINATION = 0.1
MULTIPLICITY_WORDS = {2: "doublet", 3: "triplet", 4: "quartet", 5: "quintet"}
_GREEK = {"sigma": r"\Sigma", "pi": r"\Pi", "delta": r"\Delta", "phi": r"\Phi"}


def fmt_spatial_symmetry(sym):
    """Spatial symmetry of a state for LaTeX: 'sigma.u' (cclib, linear molecules) -> Sigma_u,
    other labels (A1, B2...) unchanged."""
    if "." in sym:
        name, rest = sym.split(".", 1)
        if name.lower() in _GREEK:
            return NoEscape(f"${_GREEK[name.lower()]}_{{{escape_latex(rest)}}}$")
    return escape_latex(sym)


def velocity_rotatory_strengths(computed_data: ComputedData, log_idx):
    """Rotatory strengths in the velocity gauge of a log (compute.rotatory_strengths_velocity), or
    None (no velocity or magnetic transition dipoles, e.g. ORCA)."""
    from quchemreport.data_enrichment.compute import rotatory_strengths_velocity
    es = computed_data.results.excited_states
    return rotatory_strengths_velocity(es.et_energies[log_idx], es.et_veldips[log_idx], es.et_magdips[log_idx])


def achiral_note(rotatory_strengths, spectrum, source):
    """Sentence replacing a CD or CPL spectrum asked for an achiral molecule (compute.is_chiral),
    or "" when the molecule is chiral or has no rotatory strengths."""
    from quchemreport.data_enrichment.compute import is_chiral, CHIRAL_ROTATORY_STRENGTH
    values = [abs(r) for r in (rotatory_strengths or []) if r is not None]
    if not values or is_chiral(values):
        return ""
    return (f"No {spectrum} spectrum: every rotatory strength of {source} is below {CHIRAL_ROTATORY_STRENGTH:g} "
            f"(1e-40 erg.esu.cm/G; largest {max(values):.1e}), so the molecule is not chiral and these values are "
            "numerical noise.\n")


def dark_states_note(heights):
    """Caption sentence for the open circles of the spectra (visu_plots.DARK_MARKER), or "" when
    no stick is below visu_plots.VISIBLE_SHARE of the largest one."""
    from quchemreport.visualization_engine.visu_plots import VISIBLE_SHARE
    values = [abs(h) for h in (heights or []) if h is not None]
    if not values or not any(v <= VISIBLE_SHARE * max(values) for v in values):
        return ""
    return (f" Open circles: states too weak to be seen (below {100 * VISIBLE_SHARE:g} % of the largest stick); "
            "the axis reaches the lowest computed state, even a dark one.")


def render_excited_state_transitions_table(doc, computed_data: ComputedData, log_idx, is_optimized=False, indices=None, report_type="full", title=""):

    """Table of the selected excited states of a model: symmetry, wavelength (nm) and
    wavenumber (cm-1), oscillator and rotatory strengths, main excitations in %
    (build_CIS_string), and charge-transfer indices in full reports.


    Returns
    -------
    bool
        Whether something was added to the section.
    """
    log_files = computed_data.metadata.log_files[log_idx]
    homo_ind = computed_data.results.wavefunction.homo_indexes[log_idx]
    et_energies = computed_data.results.excited_states.et_energies[log_idx]
    et_oscs = computed_data.results.excited_states.et_oscs[log_idx]
    et_rot = computed_data.results.excited_states.et_rot[log_idx]
    et_transitions = computed_data.results.excited_states.et_transitions[log_idx]
    et_sym = computed_data.results.excited_states.et_sym[log_idx]

    # Skip is data is missing for excited state
    if (
        homo_ind == None
        or et_energies == None
        or et_oscs == None
        or et_transitions == None
        or et_sym == None
    ):
        return False
    # Rotatory strengths are optional (not printed by every program/version): no R column then.
    has_rot = et_rot is not None and len(et_rot) == len(et_energies)
    # R(velocity), computed from the velocity and magnetic transition dipoles (Gaussian)
    r_vel = velocity_rotatory_strengths(computed_data, log_idx) if has_rot else None
    has_vel = r_vel is not None
    
    if is_optimized:
        Tozer_lambda = computed_data.derived.optimized_excited_states.Tozer_lambda[log_idx]
        d_ct = computed_data.derived.optimized_excited_states.d_ct[log_idx]
        q_ct = computed_data.derived.optimized_excited_states.q_ct[log_idx]
    else:
        Tozer_lambda = computed_data.derived.excited_states.Tozer_lambda[log_idx]
        d_ct = computed_data.derived.excited_states.d_ct[log_idx]
        q_ct = computed_data.derived.excited_states.q_ct[log_idx]
    
    use_full = (report_type != 'text')

    # Skip is data is missing for optimized excited state
    if (
        use_full
        and
        (
            Tozer_lambda == None
            or d_ct == None
            or q_ct == None
        )
    ):
        return False
    
    if is_optimized:
        title = "Table. Excited state at its optimized geometry (vertical emission)."
    else:
        title = "Table. Vertical excitations from the ground state."

    et_nm = [nm_to_wnb / e if e else 0 for e in et_energies]

    if indices is None:
        indices = range(len(et_transitions))
    indices = list(indices)
    # Spatial symmetry only (A, B2...): the spin is in the S/T label; ORCA gives none, and
    # the column is then left out
    spatial = [str(sym).split("-", 1)[1] if "-" in str(sym) else "" for sym in et_sym]
    has_sym = any(spatial[idx] for idx in indices)
    # Rank of the state in the log (the excited_state_number of the YAML file) when it differs
    # from the number within the multiplicity (TD with singlets and triplets): own column
    # instead of the former "S1 (2)" next to a plain "T1"
    mixed = any(state_number(et_sym, idx) != idx + 1 for idx in indices)

    doc.append(NoEscape(r'\begin{center}'))
    
    # <S^2> of the states of an open-shell reference (scanlog._unrestricted_states)
    et_s2 = computed_data.results.excited_states.et_s2[log_idx]
    has_s2 = et_s2 is not None and len(et_s2) == len(et_energies)
    if has_s2:
        mult = computed_data.molecule.multiplicity[log_idx] or 1
        s2_pure = (mult - 1) / 2 * ((mult - 1) / 2 + 1)
    header = (["No."] if mixed else []) + ["State"] + (["Sym."] if has_sym else []) \
        + ["nm", NoEscape(r"cm$^{-1}$"), italic("f")] + (["R (len.)", "R (vel.)"] if has_vel else ["R"] if has_rot else []) \
        + ([NoEscape(r"$\langle S^2\rangle$")] if has_s2 else [])
    if use_full:
        header += [NoEscape(r"$\Lambda$"), NoEscape(r"d$_{CT}$ (pm)"), NoEscape(r"q$_{CT}$ (e)")]
    header += [NoEscape(r"Main excitations (\%)")]
    ncols = len(header)

    table_format = 'r' + 'c' * (ncols - 2) + CENTERED + '{5.5cm}'
    with doc.create(Tabular(table_format)) as es_transitions_table:
        es_transitions_table.add_row([MultiColumn(ncols, align='c', data=title)])
        es_transitions_table.add_row([MultiColumn(ncols, align='c', data=f"Source: {log_files}")])
        es_transitions_table.add_hline()
        es_transitions_table.add_row(header)
        es_transitions_table.add_hline()

        for idx in indices:
            
            energy = et_energies[idx] if et_energies[idx] is not None else 0.0
            nm_val = et_nm[idx] if et_nm[idx] is not None else 0
            osc = et_oscs[idx]
            if has_rot:
                # 2 decimals (27/11/2026, Thomas Cauchy), as R(velocity), computed from the printed
                # vectors to about 0.03 (4 decimals before: Gaussian printed 0.0 and -0.0001, ORCA 0.02572)
                rot_val = fmt_signed(et_rot[idx], 2) if et_rot[idx] is not None else "N/A"

            if use_full:
                lambda_val = Tozer_lambda[idx]
                d_ct_val = d_ct[idx]
                q_ct_val = q_ct[idx]
            
                # Skip if no data computed
                # Only print the visualization of the user-selected excited transition
                if (
                    lambda_val is None
                    or d_ct_val is None
                    or q_ct_val is None
                ):
                    continue

            trans = et_transitions[idx]
            CIS = build_CIS_string(trans, homo_ind, filter_threshold=0 if is_optimized else 5)

            # S<n>/T<n> within the multiplicity (as the images)
            row = ([f"{idx + 1}"] if mixed else []) + [state_label(et_sym, idx)] \
                + ([fmt_spatial_symmetry(spatial[idx])] if has_sym else []) + [
                f"{int(nm_val)}",
                f"{int(energy)}",
                f"{osc:.3f}",
            ]
            if has_rot:
                row.append(f"{rot_val}")
            if has_vel:
                # 2 decimals, as R(length): computed from the printed vectors (4 decimals), up to 0.034 off (HCFClBr)
                row.append(fmt_signed(r_vel[idx], 2) if r_vel[idx] is not None else "N/A")
            if has_s2:
                s2 = f"{et_s2[idx]:.3f}"
                row.append(bold(s2) if abs(et_s2[idx] - s2_pure) > S2_CONTAMINATION else s2)

            if use_full:
                row += [
                    f"{lambda_val:.2f}",
                    f"{d_ct_val:.2f}",
                    f"{q_ct_val:.2f}",
                ]

            row.append(CIS)
            es_transitions_table.add_row(row)

        es_transitions_table.add_hline()
        homo = homo_ind[0] + 1 if len(homo_ind) == 1 else f"{homo_ind[0] + 1}/{homo_ind[1] + 1}"
        note = (f"H: HOMO (orbital {homo}"
                + (r" for $\alpha$/$\beta$" if len(homo_ind) == 2 else "") + "), L: LUMO. ")
        if mixed:
            note += "No.: rank of the state in the log file. "
        if has_vel:
            note += (r"\textit{f}: oscillator strength; R: rotatory strength ($10^{-40}$ erg.esu.cm/G) in the length "
                     r"and velocity gauges (velocity: computed from the printed velocity and magnetic transition "
                     r"dipoles, within 0.04); R(length) depends on the origin of the program, R(velocity) does not. ")
        elif has_rot:
            note += r"\textit{f}: oscillator strength; R: rotatory strength. "
        if has_s2:
            note += (r"$\langle S^2\rangle$: " + f"{s2_pure:.2f} for a pure {MULTIPLICITY_WORDS.get(mult, 'spin')} state; "
                     f"in bold, more than {S2_CONTAMINATION} away (spin contamination). ")
        if use_full:
            note += (r"$\Lambda$: hole-electron overlap (0 to 1, small for charge transfer); "
                     r"d$_{CT}$, q$_{CT}$: charge-transfer distance and charge.")
        es_transitions_table.add_row([MultiColumn(ncols, align=NoEscape(RAGGED + '{0.95\\textwidth}'),
                                                  data=NoEscape(r"\footnotesize " + note))])

    doc.append(NoEscape(r'\end{center}'))
    
    return True


def render_es_transitions(section, computed_data: ComputedData, report_type):

    """Transition tables of every TD model.

    Returns
    -------
    bool
        Whether something was added to the section.
    """
    log_files = computed_data.metadata.log_files
    job_types = computed_data.comp_details.general.job_type
    isContentAdded = False
    
    for i, _ in enumerate(log_files):
        if job_types[i] == None or  "td" not in job_types[i]:
            continue
        
        content_added = render_excited_state_transitions_table(
            section,
            computed_data,
            log_idx=i,
            report_type=report_type
        )
        
        if (
            isContentAdded == False 
            and content_added
        ):
            isContentAdded = True
            
    return isContentAdded


def render_es_absorption_and_CD_spectrum(section, output_dir, computed_data: ComputedData, report_type):
    
    """Absorption and CD spectrum pictures of every TD model.

    Parameters
    ----------
    section : pylatex Section
    output_dir : str
        Folder of the pictures (with the final separator).
    computed_data : ComputedData
    report_type : str
        full, si or text: most pictures are only shown in full reports.


    Returns
    -------
    bool
        Whether something was added to the section.
    """
    log_files = computed_data.metadata.log_files
    job_types = computed_data.comp_details.general.job_type
    isContentAdded = False
    
    for log_idx, log_file in enumerate(log_files):
    
        jobs = job_types[log_idx]
        
        if (
            "td" not in jobs
        ):
            continue
        
        content_added = False
            
        # UV visible Absorption plots
        imgNameAbso = f"img-log{log_idx + 1}-UV-Abso-Spectrum.png"
        if (os.path.isfile(output_dir + imgNameAbso)) :
            figure_one_col(section, imgNameAbso, taillePng="10cm", 
                caption=("Calculated UV-visible absorption spectrum with a Gaussian broadening (FWHM = 3000 cm-1)."
                         + dark_states_note(computed_data.results.excited_states.et_oscs[log_idx])
                         + f" (Source: {log_file})")
            )
            content_added = True
            
        
        # Circular dischroism plots
        imgNameCD= f"img-log{log_idx + 1}-UV-CD-Spectrum.png"
        if (os.path.isfile(output_dir + imgNameCD)) :
            figure_one_col(section, imgNameCD, taillePng="10cm", 
                caption=("Calculated circular dichroism spectrum with a Gaussian broadening (FWHM = 3000 cm-1)."
                         + dark_states_note(computed_data.results.excited_states.et_rot[log_idx])
                         + f" (Source: {log_file})")
            )
            content_added = True
        elif computed_data.derived.excited_states.CD_spectrum[log_idx] is not None:
            note = achiral_note(computed_data.results.excited_states.et_rot[log_idx], "circular dichroism", log_file)
            if note:
                section.append(note)
                content_added = True
            
        if (
            isContentAdded == False
            and content_added
        ):
            isContentAdded = True
            
    return isContentAdded


def render_es_EDD(section, output_dir, computed_data: ComputedData, report_type):
    # figures not available in text report type
    """Density difference pictures of the selected states of every TD model.

    Parameters
    ----------
    section : pylatex Section
    output_dir : str
        Folder of the pictures (with the final separator).
    computed_data : ComputedData
    report_type : str
        full, si or text: most pictures are only shown in full reports.


    Returns
    -------
    bool
        Whether something was added to the section.
    """
    if report_type == 'text':
        return False

    log_files = computed_data.metadata.log_files
    job_types = computed_data.comp_details.general.job_type

    isContentAdded = False
    
    for log_idx, log_file in enumerate(log_files):
        
        et_transitions = computed_data.results.excited_states.et_transitions[log_idx]
        et_sym = computed_data.results.excited_states.et_sym[log_idx]
        jobs = job_types[log_idx]
        
        if (
            "td" not in jobs
            or et_transitions is None
        ):
            continue
        
        cells = []
        for k in range(len(et_transitions)):
            # viz_EDD names the files EDD-S<n> (singlet), EDD-T<n> (triplet), EDD-<n> (other states,
            # e.g. unrestricted), see utility_services.naming
            known = et_sym is not None and k < len(et_sym)
            letter = spin_letter(et_sym[k]) if known else ""
            number = state_number(et_sym, k) if known else k + 1
            state = f"{letter}{number}" if letter else f"state {number}"
            nomPng = f"img-log{log_idx + 1}-EDD-{letter}{number}.png"
            nomPng_cam2 = f"img-log{log_idx + 1}-EDD-{letter}{number}_cam2.png"
            if os.path.isfile(output_dir + nomPng):
                pngs = [nomPng] + ([nomPng_cam2] if os.path.isfile(output_dir + nomPng_cam2) else [])
                cells.append((pngs, escape_latex(state)))
        if not cells:
            continue
        two_views = any(len(pngs) == 2 for pngs, _ in cells)
        # EDD = sum of c^2 (rho_end - rho_start) (calc_orb.TD): positive part (output.colors.EDD,
        # blue by default) = electron, negative part (cream) = hole. Identical pictures for
        # two states (acetone S1 and T1) are expected when both are the same single excitation
        # (16->17): the surfaces enclose a share of the density, independent of c^2.
        caption = ("Electron density difference (excited state minus ground state) of the excited states"
                   + (", from two points of view" if two_views else "")
                   + f". {color_of('EDD', 'positive', True)}: excited electron; "
                   f"{color_of('EDD', 'negative')}: hole; each surface encloses 30% of the "
                   f"density gained or lost. (Source: {log_file})")
        figure_grid(section, cells, 2 if two_views else 3, caption)
        incomplete_amplitudes_warning(section, computed_data, log_idx, range(len(et_transitions)),
                                      "their density differences, and their d_CT and q_CT in the table, are approximate")
        isContentAdded = True
        # transition densities of the same log, after its density differences
        render_transition_densities(section, output_dir, computed_data, log_idx,
                                    computed_data.derived.excited_states.transition_dipoles[log_idx], "", log_file)

    return isContentAdded


def incomplete_amplitudes_warning(section, computed_data: ComputedData, log_idx, indices, what):
    """Paragraph naming the states of a log whose printed excitation amplitudes are incomplete
    (sum X^2 below nto.X_NORM_COMPLETE_THRESHOLD), followed by what (the pictures and values
    built from them, e.g. "their density differences ... are approximate"); nothing when they
    are all complete.

    Why (27/11/2026, Thomas Cauchy): measured against the density differences from all the
    amplitudes (Gaussian IOp(9/40=4), acetone and HCFClBr), those from the amplitudes printed by
    default differ by up to 21 % of the map and 35 pm of d_CT on the states whose printed
    amplitudes are incomplete (acetone S5: d_CT 100 pm instead of 65), and by at most 12 % on
    the others."""
    from quchemreport.data_enrichment.nto import X_NORM_COMPLETE_THRESHOLD
    et_transitions = computed_data.results.excited_states.et_transitions[log_idx]
    et_sym = computed_data.results.excited_states.et_sym[log_idx]
    if not et_transitions or not et_sym:
        return False
    names = []
    for k in indices:
        if k >= len(et_transitions) or not et_transitions[k]:
            continue
        norm = sum(c * c for _, _, c in et_transitions[k] if np.isfinite(c))
        if norm < X_NORM_COMPLETE_THRESHOLD:
            names.append(f"{state_label(et_sym, k)} ({norm:.3f})")
    if not names:
        return False
    section.append(NoEscape(r'\par\noindent{\small ' + escape_latex(
        f"The printed excitation amplitudes of {', '.join(names)} are incomplete (sum of the squared amplitudes "
        f"in parentheses, below {X_NORM_COMPLETE_THRESHOLD}): {what}. Rerun the "
        "excited-state calculation with all the amplitudes printed (Gaussian: IOp(9/40=4); ORCA: tprint 1e-6 "
        "in the %tddft block, with TDA).") + r'}\par\medskip'))
    return True


def transition_density_caption(two_views, without_log, source, emission=False, chiral=False):
    """Caption (LaTeX) of the transition density pictures (render_transition_densities)."""
    positive, negative = color_of('transition_density', 'positive'), color_of('transition_density', 'negative')
    states = "of the emitting states, at their optimized geometry" if emission else "of the excited states"
    text = escape_latex(
        f"Transition density (charge) {states}" + (", from two points of view" if two_views else "")
        + f", left, and transition dipole, right. Left: {positive} positive, {negative} negative; each surface "
        "encloses 30% of the positive or negative density. Right: electric transition dipole (blue arrow, "
        f"from the {negative} to the {positive} regions, through the centre of the density; 1 e.bohr = 2.54 D "
        "drawn as 4 bohr); its square gives the oscillator strength, ")
    text += r"$f = \frac{2}{3}\,\Delta E\,|\mu|^2$"
    text += escape_latex(". A blue sphere marks a forbidden transition: the dipoles of the positive and "
                         "negative regions cancel. The lengths of the arrows are proportional to the dipoles (thin "
                         "skeleton, arrows of constant thickness); their values are written in the corner, with ")
    text += r"$\odot$ / $\otimes$"
    text += escape_latex(" for an arrow pointing towards / away from the viewer (within 30 degrees of the line "
                         "of sight), the state being then also shown from a second point of view.")
    if chiral:
        text += escape_latex(
            " Magenta arrow: magnetic transition dipole m, 1 Bohr magneton drawn as 4 bohr: in common units "
            "(1 Bohr magneton = alpha/2 e.bohr) it is magnified 2/alpha = 274 times relative to the electric "
            "dipole. It is oriented so that the rotatory strength R is positive when the two arrows make an acute "
            "angle (E-M angle of Gaussian); then g = 4R/D = (4/274) x (projection of the magenta arrow on the blue "
            "one) / (length of the blue one). g is given with R(length) (rotatory strengths in the table of the "
            "excited states); m and R(length) depend on the origin of the program, R(velocity) does not: when "
            "the two gauges give opposite signs, the sign of g is not reliable.")
    if without_log:
        text += escape_latex(" The log gives no transition dipole: the arrows are computed from the printed "
                             f"amplitudes ({', '.join(without_log)}).")
    return text + escape_latex(f" (Source: {source})")


def dissymmetry_label(r_length, r_velocity, g):
    """Part of the label of a chiral state (LaTeX): the dissymmetry factor g, with "(sign
    uncertain)" when the rotatory strengths of the two gauges have opposite signs; "" without g.

    Why only g (27/11/2026, Thomas Cauchy): the rotatory strengths are in the table of the
    excited states; under the pictures, f and g are the relevant numbers (absorption and its
    dissymmetry). A first version gave R in both gauges there too."""
    if g is None:
        return ""
    mantissa, exponent = f"{g:.1e}".split("e")
    text = r"$g = " + mantissa + r"\times10^{" + str(int(exponent)) + r"}$"
    if r_length is not None and r_velocity is not None and r_length * r_velocity < 0:
        text += escape_latex(" (sign uncertain)")
    return text


def transition_density_warning(deviating):
    """Paragraph listing the states whose density, built from the printed amplitudes, does not
    give the transition dipole of the log (compute.TRANSITION_DIPOLE_TOLERANCE)."""
    if not deviating:
        return None
    return ("Transition densities built from the excitation amplitudes printed in the log: their dipoles depart "
            "from the transition dipoles of the log for " + ", ".join(deviating) + " (relative deviation in "
            "parentheses), so their shapes are approximate. The amplitudes below the printing threshold, and the "
            "de-excitation amplitudes Y of TD-DFT, are missing. Rerun the excited-state calculation with all the "
            "amplitudes printed (Gaussian: IOp(9/40=4); ORCA: tprint 1e-6 in the %tddft block, with TDA) for exact "
            "transition densities.")


def render_transition_densities(section, output_dir, computed_data: ComputedData, log_idx, tdms, prefix, source,
                                emission=False):
    """Grid of the transition density (visu_mayavi.viz_TDEN) and transition dipole
    (visu_mayavi.viz_TDIP) pictures of one log, side by side for each state, with its
    oscillator strength, and the warning for the approximate densities.

    Why (27/11/2026, Thomas Cauchy): the transition density shows where the transition dipole,
    hence the intensity, comes from; it replaces the overlap pictures (Oif), which were not
    in the report and lost the signs of the amplitudes (calc_orb.TD)."""
    from quchemreport.data_enrichment.compute import TRANSITION_DIPOLE_TOLERANCE, dissymmetry_factor
    et_sym = computed_data.results.excited_states.et_sym[log_idx]
    et_oscs = computed_data.results.excited_states.et_oscs[log_idx]
    et_rot = computed_data.results.excited_states.et_rot[log_idx]
    r_vel = velocity_rotatory_strengths(computed_data, log_idx)
    if tdms is None or et_sym is None:
        return False
    cells, deviating, without_log, chiral = [], [], [], False
    for k, tdm in enumerate(tdms):
        if tdm is None or k >= len(et_sym):
            continue
        letter, number = spin_letter(et_sym[k]), state_number(et_sym, k)
        state = f"{letter}{number}" if letter else f"state {number}"
        density = f"img-log{log_idx + 1}-{prefix}TDEN-{letter}{number}.png"
        dipoles = f"img-log{log_idx + 1}-{prefix}TDIP-{letter}{number}.png"
        if not (os.path.isfile(output_dir + density) and os.path.isfile(output_dir + dipoles)):
            continue
        pngs = [density, dipoles]
        second = [p.replace(".png", "_cam2.png") for p in pngs]
        if all(os.path.isfile(output_dir + p) for p in second):
            pngs += second
        label = escape_latex(state + (f": f = {et_oscs[k]:.4f}" if et_oscs is not None and k < len(et_oscs) else ""))
        if tdm.get("magnetic") is not None:
            r_length = et_rot[k] if et_rot is not None and k < len(et_rot) else None
            r_velocity = r_vel[k] if r_vel is not None and k < len(r_vel) else None
            g = dissymmetry_label(r_length, r_velocity, dissymmetry_factor(r_length, tdm["dipole"]))
            if g:
                label += ", " + g
        cells.append((pngs, label))
        chiral = chiral or tdm.get("magnetic") is not None
        if tdm["source"] == "grid":
            without_log.append(state)
        elif tdm["deviation"] is not None and tdm["deviation"] > TRANSITION_DIPOLE_TOLERANCE:
            deviating.append(f"{state} ({tdm['deviation']:.1f})")
    if not cells:
        return False
    two_views = any(len(pngs) == 4 for pngs, _ in cells)
    caption = transition_density_caption(two_views, without_log, source, emission, chiral)
    # one state per row with two views, else two states (density | dipole) per row, as the NTO pairs
    figure_grid(section, cells, 1 if two_views else 2, caption, escapeCaption=False)
    warning = transition_density_warning(deviating)
    if warning:
        section.append(NoEscape(r'\par\noindent{\small ' + escape_latex(warning) + r'}\par\medskip'))
    return True


def render_es_nto(section, output_dir, computed_data, report_type):
    """NTO pair pictures of the selected states, with their weights and the completeness of the amplitudes.

    Parameters
    ----------
    section : pylatex Section
    output_dir : str
        Folder of the pictures (with the final separator).
    computed_data : ComputedData
    report_type : str
        full, si or text: most pictures are only shown in full reports.


    Returns
    -------
    bool
        Whether something was added to the section.
    """
    # The method and the completeness check were repeated for every state (one paragraph per
    # state, 11 in the acetone test series); they are now given once per TD model, with the
    # sum of X^2 of every state on one line and one warning naming the incomplete states.
    added = False
    for log_idx, states in enumerate(computed_data.derived.excited_states.nto_pairs):
        et_sym = computed_data.results.excited_states.et_sym[log_idx]
        log_file = computed_data.metadata.log_files[log_idx]
        states = [state for state in states or [] if state["pairs"]]
        if not states:
            continue
        cells, norms, incomplete, text_lines = [], [], [], []
        for state in states:
            # S<n> / T<n> as for the EDD (utility_services.naming); <n> for other states
            label = state_label(et_sym, state["state"] - 1) if et_sym else str(state["state"])
            name = label if label[0].isalpha() else f"state {label}"
            if "x_norm" in state:
                n = state.get("n_amplitudes", "?")
                norms.append(f"{name} {state['x_norm']:.3f} ({n} amplitude{'s' if n != 1 else ''})")
                if not state.get("complete", True):
                    incomplete.append(name)
            single = len(state["pairs"]) == 1
            for pair_idx, pair in enumerate(state["pairs"], 1):
                spin = {"alpha": r", $\alpha$", "beta": r", $\beta$"}.get(pair["spin"], "")
                cell = (escape_latex(f"{name}{'' if single else f', pair {pair_idx}'}") + spin
                        + escape_latex(f": {100 * pair['weight']:.1f}%"))
                prefix = f"img-log{log_idx + 1}-NTO-{label}-pair{pair_idx}"
                hole, electron = prefix + "-MO-hole.png", prefix + "-MO-electron.png"
                if report_type != "text" and os.path.isfile(output_dir + hole) and os.path.isfile(output_dir + electron):
                    cells.append(([hole, electron], NoEscape(cell)))
                else:
                    text_lines.append(NoEscape(f"NTO {cell}."))
        added = True
        section.append(f"Natural transition orbitals (NTO) of {log_file}. The weights are normalized "
                       "over the printed excitation amplitudes X; the de-excitation amplitudes Y are "
                       "excluded, and the canonical orbitals of the reference must match the TD "
                       "calculation. ")
        if norms:
            section.append("Sum of the squared printed amplitudes X (about 1, or slightly above for "
                           "full TD-DFT, when all amplitudes are printed): " + "; ".join(norms) + ". ")
        if incomplete:
            section.append(bold(f"Warning: the printed excitation amplitudes of {', '.join(incomplete)} "
                                "are incomplete; their NTO weights and shapes are approximate. Rerun the "
                                "excited-state calculation with a lower printing threshold "
                                "(Gaussian: IOp(9/40=4); ORCA: TPrint in the %tddft block). "))
        for line in text_lines:
            section.append(NoEscape(r'\par '))
            section.append(line)
        figure_grid(section, cells, 2,
                    "Natural transition orbitals of the excited states: hole (left) and electron "
                    f"(right) of each pair, with its weight. {color_of('NTO', 'positive', True)} and "
                    f"{color_of('NTO', 'negative')} show opposite orbital "
                    f"phases; their absolute signs are arbitrary. (Source: {log_file})")
    return added


def fill_excited_states_section(doc, output_dir, computed_data: ComputedData, report_type, config_output: OutputInclude):
    """Excited states section: transitions, spectra, density differences and NTO."""
    section = Section('EXCITED STATES')

    added_es_transition = False
    added_es_abso_and_cd = False
    added_es_EDD = False
    added_es_nto = False
    
    # Render excited states transition for each available TD logfile
    if config_output.excited_states.transitions:
        added_es_transition = render_es_transitions(section, computed_data, report_type)
    
    # Render excited states Absorption and CD Spectrum for each available TD logfile
    if (
        config_output.excited_states.absorption_spectrum
        or config_output.excited_states.CD_spectrum
    ):
        added_es_abso_and_cd = render_es_absorption_and_CD_spectrum(section, output_dir, computed_data, report_type)
    
    # Render excited states EDD for each available TD logfile
    if config_output.excited_states.electron_density_difference:
        added_es_EDD = render_es_EDD(section, output_dir, computed_data, report_type)

    if config_output.excited_states.natural_transition_orbitals.enabled:
        added_es_nto = render_es_nto(section, output_dir, computed_data, report_type)
    # Spin-orbit couplings (ORCA dosoc), with the transitions
    added_es_soc = False
    if config_output.excited_states.transitions:
        added_es_soc = render_es_soc(section, output_dir, computed_data, report_type)
    
    if (
        added_es_transition
        or added_es_abso_and_cd
        or added_es_EDD
        or added_es_nto
        or added_es_soc
    ):
        doc.append(section)


# Spin-orbit coupling tables: triplets (rows) and singlets (columns) shown at most
SOC_TABLE_TRIPLETS = 6
SOC_TABLE_SINGLETS = 3


def render_soc_couplings(section, output_dir, computed_data: ComputedData, log_idx, report_type):
    """Spin-orbit couplings of one model (ORCA ``dosoc``): table of |<Tn|H_SO|Sm>| (cm-1) for
    the lowest triplets and singlets, with the energies of the states, and the
    Jablonski-Perrin diagram (``img-log<k>-Jablonski.png``, full reports).

    Returns
    -------
    bool
        Whether something was added to the section.
    """
    soc = computed_data.results.excited_states.soc[log_idx]
    if not soc or not soc["socme"]:
        return False
    log_file = computed_data.metadata.log_files[log_idx]
    singlets = [{"number": 0, "energy": 0.0}] + soc["singlets"][:SOC_TABLE_SINGLETS]
    triplets = soc["triplets"][:SOC_TABLE_TRIPLETS]
    ncol = 1 + len(singlets)
    # Fixed column widths: a title cell wider than the columns would widen the last one
    width = 0.16 + 0.13 * len(singlets)
    spec = "p{0.16\\textwidth}" + (CENTERED + "{0.13\\textwidth}") * len(singlets)
    with section.create(LongTable(spec)) as table:
        # ragged title, and the definition of the norm moved under the table: the justified
        # title with the formula was stretched with large spaces
        table.add_row([MultiColumn(ncol, align=NoEscape(RAGGED + f'{{{width:.2f}\\textwidth}}'), data=NoEscape(
            r"Table. Spin-orbit couplings $|\langle T_n|H_{SO}|S_m\rangle|$ (cm$^{-1}$) between the "
            r"triplets (rows) and the singlets (columns), state energies (eV) in parentheses. "
            f"Source: {escape_latex(log_file)}"))])
        table.add_hline()
        table.add_row([""] + [NoEscape(f"S$_{s['number']}$ ({s['energy'] / 8065.544:.2f})") for s in singlets])
        table.add_hline()
        for t in triplets:
            row = [NoEscape(f"T$_{t['number']}$ ({t['energy'] / 8065.544:.2f})")]
            for s_ in singlets:
                value = soc["socme"].get((t["number"], s_["number"]))
                row.append("" if value is None else f"{value:.1f}")
            table.add_row(row)
        table.add_hline()
        table.add_row([MultiColumn(ncol, align=NoEscape(RAGGED + f'{{{width:.2f}\\textwidth}}'), data=NoEscape(
            r"\footnotesize $|\langle T_n|H_{SO}|S_m\rangle| = "
            r"\sqrt{\sum_{M_S}|\langle T_n,M_S|H_{SO}|S_m\rangle|^2}$, over the three sublevels of the triplet."))])
    picture = f"img-log{log_idx + 1}-Jablonski.png"
    if report_type == "full" and os.path.isfile(output_dir + picture):
        figure_one_col(section, picture, taillePng="11cm", escapeCaption=False, caption=(
            "Jablonski-Perrin diagram of the vertical singlet and triplet states at the geometry of "
            f"{escape_latex(log_file)}, with the spin-orbit couplings "
            r"$|\langle T|H_{SO}|S\rangle|$ (cm$^{-1}$) of the intersystem "
            "crossings from S$_1$ (triplets below S$_1$ + 0.3 eV) and from T$_1$ to S$_0$."))
    return True


def render_es_soc(section, output_dir, computed_data: ComputedData, report_type):
    """Spin-orbit couplings of every TD model that has them (render_soc_couplings).

    Returns
    -------
    bool
        Whether something was added to the section.
    """
    added = False
    for log_idx, jobs in enumerate(computed_data.comp_details.general.job_type):
        if jobs and "td" in jobs:
            added = render_soc_couplings(section, output_dir, computed_data, log_idx, report_type) or added
    return added


def render_phosphorescence(section, output_dir, computed_data: ComputedData, report_type):
    """Phosphorescence of the optimized triplets with SOC (compute.compute_phosphorescence):
    table of the sublevels (energy, wavelength, oscillator strength, radiative rate), mean
    radiative rate and radiative lifetime, spectrum, and the spin-orbit couplings at the
    optimized geometry.

    Returns
    -------
    bool
        Whether something was added to the section.
    """
    added = False
    data = computed_data.derived.optimized_excited_states.phosphorescence
    for log_idx, phospho in enumerate(data):
        if not phospho:
            continue
        log_file = computed_data.metadata.log_files[log_idx]
        with section.create(LongTable("p{0.14\\textwidth}" + "".join(CENTERED + f"{{{w}\\textwidth}}" for w in (0.12, 0.1, 0.14, 0.14)))) as table:
            table.add_row([MultiColumn(5, align=NoEscape('p{0.72\\textwidth}'), data=NoEscape(
                f"Table. Phosphorescence from the sublevels of {phospho['state']} at its optimized "
                "geometry (vertical emission, spin-orbit coupled states). Radiative rates "
                r"$k_r = f\,\tilde\nu^2/1.499$ s$^{-1}$ (Strickler-Berg). "
                f"Source: {escape_latex(log_file)}"))])
            table.add_hline()
            table.add_row(["Sublevel", NoEscape(r"cm$^{-1}$"), "nm", "f", NoEscape(r"$k_r$ (s$^{-1}$)")])
            table.add_hline()
            for k, sub in enumerate(phospho["sublevels"], 1):
                table.add_row([f"{phospho['state']} ({k})", f"{sub['energy']:.0f}", f"{1e7 / sub['energy']:.0f}",
                               f"{sub['fosc']:.2e}", f"{sub['k_r']:.3g}"])
            table.add_hline()
            lifetime = phospho["lifetime"]
            table.add_row([MultiColumn(5, align=NoEscape('p{0.72\\textwidth}'), data=NoEscape(
                f"Mean radiative rate (sublevels in thermal equilibrium): {phospho['k_r']:.3g} s$^{{-1}}$; "
                + (f"radiative lifetime {lifetime * 1e3:.3g} ms" if lifetime else "no radiative decay")))])
        picture = f"img-log{log_idx + 1}-Phospho-Spectrum.png"
        if os.path.isfile(output_dir + picture):
            figure_one_col(section, picture, taillePng="10cm", caption=(
                "Calculated phosphorescence spectrum: one Gaussian band (FWHM = 3000 cm-1) per sublevel "
                "of the triplet, weighted by its oscillator strength."))
        render_soc_couplings(section, output_dir, computed_data, log_idx, report_type)
        added = True
    return added


def render_optimized_es_transitions(section, computed_data: ComputedData, report_type, config: Config):
    """Table of the emitting state of every optimized excited state.

    Returns
    -------
    bool
        Whether something was added to the section.
    """
    log_files = computed_data.metadata.log_files
    job_types = computed_data.comp_details.general.job_type

    isContentAdded = False

    for i, _ in enumerate(log_files):
        if job_types[i] == None or "opt_es" not in job_types[i]:
            continue

        emi_state = config_logfile(config, computed_data, i).excited_state_number
        indices = emi_state - 1
        et_energies = computed_data.results.excited_states.et_energies[i]
        if et_energies is None or not 0 <= indices < len(et_energies):
            print(f"Warning: excited_state_number {emi_state} of {log_files[i]} is not among its "
                  f"{0 if et_energies is None else len(et_energies)} excited states: no emission table.")
            continue
        
        content_added = render_excited_state_transitions_table(
            section,
            computed_data,
            log_idx=i,
            indices=[indices],
            is_optimized=True,
            report_type=report_type
        )
        
        if (
            isContentAdded == False
            and content_added
        ):
            isContentAdded = True
            
    return isContentAdded


def render_optimized_es_emission_spectrum(section, output_dir, computed_data: ComputedData, report_type):
    
    """Emission spectrum pictures.

    Parameters
    ----------
    section : pylatex Section
    output_dir : str
        Folder of the pictures (with the final separator).
    computed_data : ComputedData
    report_type : str
        full, si or text: most pictures are only shown in full reports.


    Returns
    -------
    bool
        Whether something was added to the section.
    """
    log_files = computed_data.metadata.log_files
    job_types = computed_data.comp_details.general.job_type
    
    isContentAdded = False
    
    for log_idx,_ in enumerate(log_files):
        
        jobs = job_types[log_idx]
        
        if not any("_es" in jb for jb in jobs):
            continue
        
        # UV visible Emission plots
        imgNamePng = f"img-log{log_idx + 1}-UV-Emi-Spectrum.png"
        if (not os.path.isfile(output_dir + imgNamePng)) :
            continue
        
        figure_one_col(section, imgNamePng, taillePng="10cm", 
            caption=("Calculated UV-visible emission spectrum with a Gaussian broadening (FWHM = 3000 cm-1). "
                     f"(Source: {log_files[log_idx]})")
        )
        
        if (isContentAdded == False):
            isContentAdded = True
    
    return isContentAdded


def render_optimized_es_CPL_spectrum(section, output_dir, computed_data: ComputedData, report_type):
    
    """CPL spectrum pictures.

    Parameters
    ----------
    section : pylatex Section
    output_dir : str
        Folder of the pictures (with the final separator).
    computed_data : ComputedData
    report_type : str
        full, si or text: most pictures are only shown in full reports.


    Returns
    -------
    bool
        Whether something was added to the section.
    """
    log_files = computed_data.metadata.log_files
    job_types = computed_data.comp_details.general.job_type
    
    isContentAdded = False
    
    for log_idx,_ in enumerate(log_files):
        
        jobs = job_types[log_idx]
        
        if not any("_es" in jb for jb in jobs):
            continue
    
        imgNamePng = f"img-log{log_idx + 1}-UV-CD-Emi-Spectrum.png"
        if (not os.path.isfile(output_dir + imgNamePng)) :
            if computed_data.derived.optimized_excited_states.CPL_spectrum[log_idx] is not None:
                note = achiral_note(computed_data.results.excited_states.et_rot[log_idx],
                                    "circularly polarized luminescence", log_files[log_idx])
                if note:
                    section.append(note)
                    isContentAdded = True
            continue
        
        figure_one_col(section, imgNamePng, taillePng="10cm", 
            caption="Calculated circularly polarized luminescence spectrum with a Gaussian broadening (FWHM = 3000 cm-1)."
        )
        
        if (isContentAdded == False):
            isContentAdded = True
    
    return isContentAdded


def render_optimized_es_EDD(section, output_dir, computed_data: ComputedData, config: Config, report_type):
    """Density difference pictures of the emitting states.

    Returns
    -------
    bool
        Whether something was added to the section.
    """
    if report_type == 'text':
        return False
    
    log_files = computed_data.metadata.log_files
    job_types = computed_data.comp_details.general.job_type
    
    isContentAdded = False
    
    cells = []
    for log_idx,log_file in enumerate(log_files):
        
        jobs = job_types[log_idx]
        
        if not any("_es" in jb for jb in jobs):
            continue
        
        emi_state = config_logfile(config, computed_data, log_idx).excited_state_number
        et_sym = computed_data.results.excited_states.et_sym[log_idx]
        if et_sym is None or not 0 < emi_state <= len(et_sym):
            continue
        letter = spin_letter(et_sym[emi_state - 1])
        number = state_number(et_sym, emi_state - 1)
        state = f"{letter}{number}" if letter else f"state {number}"
        
        # figure with EDD for emission (named by output_optimized_es_EDD_DIP)
        imgNamePng = f"img-log{log_idx + 1}-emi-EDD-{letter}{number}.png" 
        imgNamePng2 = f"img-log{log_idx + 1}-emi-EDD-{letter}{number}_cam2.png" 
        if (not os.path.isfile(output_dir + imgNamePng)) :
            continue
        pngs = [imgNamePng] + ([imgNamePng2] if os.path.isfile(output_dir + imgNamePng2) else [])
        cells.append((pngs, escape_latex(f"{state} ({log_file})")))

    # One grid for all the optimized states (one figure per state before)
    if cells:
        two_views = any(len(pngs) == 2 for pngs, _ in cells)
        figure_grid(section, cells, 1 if two_views else 2, max_width=None if two_views else 0.4, caption=
                    "Electron density difference of the excited states at their optimized geometry"
                    + (", from two points of view" if two_views else "")
                    # Drawn for the emission (output_optimized_es_EDD_DIP: ground minus excited
                    # state density, 27/11/2026): the positive surface is where the electron goes
                    # back to, the reverse of the absorption EDD at the same geometry.
                    + f", drawn for the emission (excited state to ground state). "
                    f"{color_of('EDD', 'positive', capital=True)}: region the electron returns to in the emission; "
                    f"{color_of('EDD', 'negative')}: region it leaves (the reverse of the absorption pictures); "
                    "each surface encloses 30% of the density gained or lost.")
        isContentAdded = True

    # States whose printed amplitudes are incomplete, then the transition density of each
    # emitting state (the transition of the emission)
    if isContentAdded:
        for log_idx, log_file in enumerate(log_files):
            if not any("_es" in jb for jb in (job_types[log_idx] or [])):
                continue
            emi_state = config_logfile(config, computed_data, log_idx).excited_state_number
            if emi_state:
                incomplete_amplitudes_warning(section, computed_data, log_idx, [emi_state - 1],
                                              f"the emission density difference ({log_file}) is approximate")

    for log_idx, log_file in enumerate(log_files):
        if not any("_es" in jb for jb in (job_types[log_idx] or [])):
            continue
        tdms = computed_data.derived.optimized_excited_states.transition_dipoles[log_idx]
        if render_transition_densities(section, output_dir, computed_data, log_idx, tdms, "emi-", log_file,
                                       emission=True):
            isContentAdded = True

    return isContentAdded    


def fill_optimized_excited_states_section(doc, output_dir, computed_data: ComputedData, report_type, config: Config, config_output: OutputInclude):
    
    """Optimized excited states section: emitting state, emission and CPL spectra, density differences."""
    section = Section('OPTIMIZED EXCITED STATES')
    
    added_optimized_es_transitions = False
    added_optimized_es_emission_spectrum = False
    added_optimized_es_CPL_spectrum = False
    added_optimized_es_EDD = False
    
    # Render optimized excited states transition for each available TD logfile
    if config_output.optimized_excited_states.transitions:
        added_optimized_es_transitions = render_optimized_es_transitions(section, computed_data, report_type, config)
        
    # Render optimized excited states Emission Spectrum for each available TD logfile
    if config_output.optimized_excited_states.emission_spectrum:
        added_optimized_es_emission_spectrum = render_optimized_es_emission_spectrum(section, output_dir, computed_data, report_type)
        # Phosphorescence of the optimized triplets with SOC (ORCA dosoc)
        added_optimized_es_emission_spectrum = render_phosphorescence(
            section, output_dir, computed_data, report_type) or added_optimized_es_emission_spectrum
        
    # Render optimized excited states CPL Spectrum for each available TD logfile
    if config_output.optimized_excited_states.CPL_spectrum:
        added_optimized_es_CPL_spectrum = render_optimized_es_CPL_spectrum(section, output_dir, computed_data, report_type)
        
    # Render optimized excited states EDD for each available TD logfile
    if config_output.optimized_excited_states.electron_density_difference:
        added_optimized_es_EDD = render_optimized_es_EDD(section, output_dir, computed_data, config, report_type)
    
    if (
        added_optimized_es_transitions
        or added_optimized_es_emission_spectrum
        or added_optimized_es_CPL_spectrum
        or added_optimized_es_EDD
    ):
        doc.append(section)


def fill_additional_molecule_representations_section(doc, computed_data: ComputedData):
    """Section of the topology pictures of the models whose topology differs from the reference."""
    ref_id = computed_data.metadata.ref_log_file_idx
    output_dir = computed_data.metadata.output_dir
    log_files = computed_data.metadata.log_files
    topology_groups = computed_data.derived.molecule.topology_groups
    
    if topology_groups is None:
        return
    
    # Skip if no other molecule representations than the reference
    if all((group is None or ref_id in group) for group in topology_groups):
        return
    
    with doc.create(Section('ADDITIONAL MOLECULE REPRESENTATION')):
    
        for i,_ in enumerate(log_files):
            group = topology_groups[i]
            
            # Skip if no data
            if group is None:
                continue
            
            log_idx = group[0]
            
            # Skip the molecule representation of the reference since it's done in the molecule section
            if (log_idx == ref_id):
                continue
            
            nomPng = f"img-log{log_idx + 1}-TOPOLOGY.png"
            
            if ( not os.path.isfile(output_dir + nomPng) ):
                print(f"{nomPng} not found. Molecule representation for the reference can not be added to the report.\n")
                return
            
            sources = ', '.join([log_files[i] for i in group])
            topology_figure(doc, output_dir, nomPng, f"(Source: {sources})")


def render_atomic_coordinates(doc, computed_data: ComputedData):
    """Coordinate tables of the distinct geometries, the reference first."""
    for coords, members in geometry_groups(computed_data):
        render_coordinates_group(doc, computed_data, coords, members)


def fill_atomic_coordinates_section(doc, computed_data: ComputedData):
    """Final section of the atomic coordinates: the distinct geometries (see geometry_groups)."""
    if not geometry_groups(computed_data):
        return

    with doc.create(Section('ATOMIC COORDINATES')):
        render_atomic_coordinates(doc, computed_data)


def render_report_document(reportFile, output_dir, doc, mode):
    #####################################################################
    #                                                                   #
    ## 3. compilation LaTeX                                             #
    #                                                                   #
    #####################################################################
    
    # on compile
    # on previent si le fichier PDF est present ou pas
    # par defaut dans pylatex tout ce qui concerne latex est efface
    
    # doc.generate_pdf(rapFile, clean_tex=True) # comportement en routine, avec False en mode dev/debug
    # OK sans SVG doc.generate_pdf(rapFile, clean_tex=False)
    # pas de chance !! doc.generate_pdf(rapFile, clean_tex=False,compiler="pdflatex  --shell-escape")
    
    """Write the .tex file and compile it with pdflatex in the output folder.

    Parameters
    ----------
    reportFile : str
        Path of the report without extension (``.tex`` and ``.pdf`` are added).
    output_dir : str
    doc : pylatex Document
    mode : str
        ``"clean"``: hide the pdflatex output and remove the auxiliary files.
    """
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    texFile = reportFile + ".tex"
    if (os.path.isfile(texFile)) :
        os.remove(texFile)
    
    pdfFile = reportFile + ".pdf"
    if (os.path.isfile(pdfFile)) :
        os.remove(pdfFile)
    
    current_dir = os.getcwd()
    try:
        os.chdir(output_dir)
        tex_filename = os.path.basename(reportFile)

        # Generating latex file
        doc.generate_tex(tex_filename)

        # LaTeX compilation
        cmd = f"pdflatex --shell-escape -interaction=nonstopmode {tex_filename}"
        if mode == "clean":
            cmd += " > /dev/null"
        os.system(cmd)
    finally:
        os.chdir(current_dir)
    
    # Cleaning auxilares files
    if mode=="clean" :
        if os.path.isfile(reportFile+'.aux') :
            os.remove(reportFile+'.aux')
        if os.path.isfile(reportFile+'.log') :
            os.remove(reportFile+'.log')
    #       if os.path.isfile(reportFile+'.tex') :
    #          os.remove(reportFile+'.tex')
    
    # Ending message
    if (os.path.isfile(pdfFile)) :
        print('Report file is ' + reportFile + ".pdf")
    else :
        print('Probably a LateX error.')


def generate_report(computed_data: ComputedData, config:Config, mode="unset"):
    
    """Build and compile the report.

    The report is ``<output_path>/<molecule_designated_name>/<name>_<verbosity>_report.txt.pdf``.

    Parameters
    ----------
    computed_data : ComputedData
    config : Config
    mode : str, optional
        Compilation mode; forced to ``"clean"`` unless ``logging.level`` is debug.
    """
    print("\n=================================================================")
    print(" Report Generation")
    print("=================================================================")
    
    # 3 report types are considered. 
    # Full report have an original layout. Pretty but inappropiate for pdf2word conversion.
    # They will print all considered data and needs discretization process: Full log verbosity. Orbkit calculations. 
    # SI mode will discard the layout. Better for the Supplementary inofrmation generation througth a pdf 2word conversion
    # SI mode will also not print the Fukui values and pictures, Analysis populations, Excited states dipoles. 
    # text mode reports are the most simple ones and discard all discretization pictures and calculations
    # Based on the SI mode, we remove the FMO, EDD pictures, CDFT and Charge transfer data.
    report_type = config.output.verbosity
    
    if config.logging.level != "debug":
        mode = "clean"

    # Get output_dir
    designated_name = config.output.molecule_designated_name
    output_dir = os.path.join(config.output.output_path, designated_name)
    if not output_dir.endswith(os.sep):
        output_dir += os.sep
    global IMAGE_DIR
    IMAGE_DIR = output_dir

    # Report file
    reportName = f"{sanitize_filename(designated_name)}_{sanitize_filename(report_type)}_report.txt"
    reportFile = output_dir + reportName
    
    config_output: OutputInclude = config.output.include
    
    # Generate Document (LaTeX for now)
    doc = generate_empty_document(report_type)
    
    # Molecule first, then the log files (Thomas Cauchy, 27/11/2026: the report opens on what
    # was computed; the charge and multiplicity of every log are in the table of the logs)
    fill_molecule_section(doc, output_dir, computed_data, designated_name, report_type, config_output)
    fill_logfiles_details_section(doc, computed_data, config)
    
    # Fill Computationals Details Section
    if config_output.computational_details:
        fill_computational_details_section(doc, computed_data)
    
    # Fill Wavefunction Section
    fill_wavefunction_section(doc, output_dir, computed_data, report_type, config_output)
    
    # Fill Frequencies Section
    fill_frequencies_section(doc, computed_data, config_output)
    
    # Fill NMR Section
    # TODO : fill_NMR_section(doc, computed_data, report_type)

    # Fill Reactivity Section
    fill_reactivity_section(doc, output_dir, computed_data, report_type, config_output)
    
    # Fill Excited States Section
    fill_excited_states_section(doc, output_dir, computed_data, report_type, config_output)
    
    # Fill Optimized Excited States Section
    fill_optimized_excited_states_section(doc, output_dir, computed_data, report_type, config, config_output)
    
    # Optionally fill with a new section for the additional molecule representations
    if config_output.molecule.molecule_representation:
        fill_additional_molecule_representations_section(doc, computed_data)
    
    # Optionally fill with a new section for the additional Atomic Coordinates
    if config_output.molecule.atomic_coordinates:
        fill_atomic_coordinates_section(doc, computed_data)
    
    # Generate Report Document (LaTeX for now)
    render_report_document(reportFile, output_dir, doc, mode)
