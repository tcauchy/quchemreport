"""Folder of the report when it already holds one (execution_engine/output_folder.py, 27/11/2026):
ask in a terminal, stop without one, or what output.if_exists says.

Run (see tests/README.md): tests/run_tests.sh -k output_folder
"""
import os

import pytest

from quchemreport.execution_engine.output_folder import OutputFolderExists, own_files, resolve_output_folder


def previous_report(folder):
    os.makedirs(folder, exist_ok=True)
    for name in ("img-log1-MEP.png", "log3-UV-Abso-Spectrum.txt", "mol_full_report.txt.pdf",
                 "mol_full_report.txt.tex", "notes.txt", "experiment.png"):
        open(os.path.join(folder, name), "w").close()


def test_free_folder_is_used(tmp_path):
    assert resolve_output_folder(str(tmp_path), "mol", "ask", interactive=False) == str(tmp_path / "mol")
    os.makedirs(tmp_path / "other")
    open(tmp_path / "other" / "notes.txt", "w").close()          # no report in it: used as is
    assert resolve_output_folder(str(tmp_path), "other", "stop") == str(tmp_path / "other")




def test_no_terminal_stops(tmp_path):
    previous_report(tmp_path / "mol")
    assert own_files(str(tmp_path / "mol")) == ["img-log1-MEP.png", "log3-UV-Abso-Spectrum.txt",
                                                "mol_full_report.txt.pdf", "mol_full_report.txt.tex"]
    with pytest.raises(OutputFolderExists, match="if_exists"):
        resolve_output_folder(str(tmp_path), "mol", "ask", interactive=False)
    with pytest.raises(OutputFolderExists):
        resolve_output_folder(str(tmp_path), "mol", "stop", interactive=True)


def test_terminal_answers(tmp_path):
    previous_report(tmp_path / "mol")
    answers = iter(["?", "n"])                                    # an unknown answer is asked again
    assert resolve_output_folder(str(tmp_path), "mol", "ask", interactive=True,
                                 ask=lambda _: next(answers)) == str(tmp_path / "mol_2")
    with pytest.raises(OutputFolderExists):
        resolve_output_folder(str(tmp_path), "mol", "ask", interactive=True, ask=lambda _: "s")
    assert resolve_output_folder(str(tmp_path), "mol", "ask", interactive=True,
                                 ask=lambda _: "o") == str(tmp_path / "mol")
    assert sorted(os.listdir(tmp_path / "mol")) == ["experiment.png", "notes.txt"]    # the user's files stay


def test_new_folder_numbering(tmp_path):
    previous_report(tmp_path / "mol")
    assert resolve_output_folder(str(tmp_path), "mol", "new") == str(tmp_path / "mol_2")
    previous_report(tmp_path / "mol_2")
    assert resolve_output_folder(str(tmp_path), "mol", "new") == str(tmp_path / "mol_3")


def test_config_option():
    import yaml
    from quchemreport.config.config import Output
    base = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "example.yaml")))["output"]
    assert Output(**base).if_exists == "ask"
    assert Output(**{**base, "if_exists": "New"}).if_exists == "new"
    with pytest.raises(ValueError):
        Output(**{**base, "if_exists": "rename"})
    out = Output(**{**base, "output_path": "/tmp/x", "molecule_designated_name": "mol"})
    assert out.report_folder() == "/tmp/x/mol/"
    out.set_report_folder("/tmp/x/mol_2")
    assert out.report_folder() == "/tmp/x/mol_2/"
