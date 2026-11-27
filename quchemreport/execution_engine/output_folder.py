"""Folder of the report: ``<output_path>/<molecule_designated_name>`` when it holds no previous
report, otherwise what ``output.if_exists`` says.

Why (27/11/2026, Thomas Cauchy): the folder was reused silently. A report launched again with
the same name overwrote the former one file by file, and a picture the new run did not make
(e.g. a CD spectrum no longer drawn) stayed and was put in the new report. The folder is now
checked at launch, before any computation:

- ``ask`` (default): in a terminal, the user chooses a new folder, to overwrite, or to stop;
  without a terminal (Slurm job), quchemreport stops with a message, rather than choose for
  the user (an automatic choice was rejected: a forgotten change of name would go unnoticed);
- ``new``: the first free folder ``<name>_2``, ``<name>_3``...;
- ``overwrite``: the files made by quchemreport (OWN_FILES) are deleted first, the other files
  of the folder are kept;
- ``stop``: stop with a message.
"""

import os
import re
import sys

# Files written by quchemreport in the folder of a report: pictures, spectra as text, the report
# itself and the files left by pdflatex
OWN_FILES = re.compile(r"^(img-.+\.(png|pov|ini)"
                       r"|log\d+-.+\.txt"
                       r"|.+_(full|si|text)_report\.txt\.(tex|pdf|aux|log|out|toc))$")
CHOICES = {"n": "new", "o": "overwrite", "s": "stop"}


class OutputFolderExists(SystemExit):
    """The folder holds a previous report and quchemreport stops: the message is printed on the
    standard error and the exit status is 1 (SystemExit with a message)."""


def own_files(folder):
    """Files made by quchemreport in folder (OWN_FILES), sorted; [] if the folder does not exist."""
    if not os.path.isdir(folder):
        return []
    return sorted(name for name in os.listdir(folder) if OWN_FILES.match(name))


def _stop_message(folder, reason):
    return (f"Error: {folder} already holds a report ({reason}). quchemreport stops so as not to overwrite it. "
            "Choose another output.molecule_designated_name or output.output_path, or set output.if_exists "
            "to new (a new folder <name>_2...) or overwrite.")


def _new_folder(folder):
    k = 2
    while own_files(f"{folder}_{k}"):
        k += 1
    return f"{folder}_{k}"


def resolve_output_folder(output_path, name, if_exists="ask", interactive=None, ask=input):
    """Folder (without the final separator) where the report of this run is written.

    Parameters
    ----------
    output_path, name : str
        output.output_path and output.molecule_designated_name.
    if_exists : str
        ask, new, overwrite or stop (see the module).
    interactive : bool, optional
        Whether the user can be asked; default: standard input and output are terminals.
    ask : callable
        Reads the answer (input; replaced in the tests).

    Raises
    ------
    OutputFolderExists
        The folder holds a report and the choice is to stop.
    """
    folder = os.path.join(output_path, name)
    existing = own_files(folder)
    if not existing:
        return folder
    reason = f"{len(existing)} files of a previous run, e.g. {existing[0]}"
    choice = if_exists
    if choice == "ask":
        if interactive is None:
            interactive = sys.stdin.isatty() and sys.stdout.isatty()
        if not interactive:
            raise OutputFolderExists(_stop_message(folder, reason) + " (No terminal to ask the user: "
                                     "output.if_exists is ask.)")
        print(f"{folder} already holds a report ({reason}).")
        choice = None
        while choice is None:
            answer = ask(f"[n] new folder {os.path.basename(_new_folder(folder))}, [o] overwrite, [s] stop? ")
            choice = CHOICES.get(answer.strip().lower()[:1])
    if choice == "stop":
        raise OutputFolderExists(_stop_message(folder, reason))
    if choice == "new":
        new = _new_folder(folder)
        print(f"Note: {folder} already holds a report; this report is written in {new}.")
        return new
    for file_name in existing:
        os.remove(os.path.join(folder, file_name))
    print(f"Note: the {len(existing)} files of the previous report in {folder} were deleted (output.if_exists: "
          "overwrite); the other files of the folder are kept.")
    return folder
