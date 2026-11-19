# QuChemReport

QuChemReport is a Python module for molecular quantum chemistry calculations results. 

The project was started in 2010 by Thomas CAUCHY, assistant professor in the MOLTECH-Anjou laboratory of the University of Angers (FRANCE).
At first a computational chemist program maintained by Thomas CAUCHY and Yohann MORILLE, the project boomed with the involvment of Benoit DA MOTA. 
Benoit DA MOTA is a computer scientist assistant professor of the LERIA laboratory of the University of Angers.

Installation
Dependencies
QuChemReport works with:

Python (>= 3.5)
NumPy
SciPy

For the parsing process:
requests
Openbabel (>= 2.4.1) (apt install openbabel-dev and pip3 install openbabel)
Cclib (>= 1.5)
Scikit-learn
scanlog (pip3 install scanlog)

For  the discretization process:
cython3
python3-h5py
Orbkit (git clone https://github.com/orbkit/orbkit.git, beware to export ORBKITPATH)

For the vizualization process:
latex (pdflatex)
pylatex (pip3 install pylatex)
PIL or on conda pillow
Matplotlib (>3.1)
Mayavi (pip3 install : beware of pyface install ! problem with Qt4, Qt5 in profile : export QT_API=pyqt, export ETS_TOOLKIT=qt4)

User installation

git clone https://github.com/BenoitDamota/QuChemReport.git

Project History

Natural transition orbitals (NTO)

Set `output.include.excited_states.natural_transition_orbitals.enabled` to
`true` in the YAML file to generate hole/electron NTO pairs for the states
selected by `excited_states.selection`. The example YAML documents the
maximum pairs per state, minimum relative weight, grid spacing in bohr and
number of ORBKIT workers. The feature is disabled by default, including for
older configuration files.

The calculation uses the printed excitation amplitudes X, omits TDDFT
de-excitation amplitudes Y, and normalizes the squared singular values over
all available X amplitudes for each state. Open-shell alpha and beta blocks
are decomposed separately and their weights share the same normalization.
Missing or truncated coefficients can change both the NTOs and their weights;
for Gaussian, `IOp(9/40=4)` requests more printed amplitudes. The reference
log must contain basis and MO coefficients, including all orbitals used by
the printed transitions. Its canonical orbitals must be the same orbitals,
in the same order and phases, as those used for the TD calculation. The code
checks available geometry, spin, basis, method and orbital energies, but these
checks cannot prove that orbital phases or rotations match. Prefer a TD log
containing its own basis and MO coefficients when available, or use an
explicitly shared reference/checkpoint. Incompatible data are skipped with a
warning. Green and gold in NTO images represent opposite orbital phases;
absolute phase signs are arbitrary.

Help and Support
Documentation

Communication

Citation



