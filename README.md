# PyCrystalField
Code to calculate the crystal field Hamiltonian of magnetic ions.

Created by Allen Scheie

   scheieao@ornl.gov

Please cite   A Scheie, "PyCrystalField: Software for Calculation, Analysis, and Fitting of Crystal Electric Field Hamiltonians", arXiv preprint arXiv:2006.15151 (2020).




## For documentation, see [here](https://github.com/asche1/PyCrystalField/wiki)

### Update (Oct. 15, 2020) to version 2.2

Added tables of radial integrals and spin orbit coupling constants for 3d and 4d transition ions.

Modified importCIF so that it will work for 3d transition ions, adding example (NCNF.py) showing how to do this.

### Update (Mar. 13, 2020) to version 2.1

Added importCIF function which imports a crystal structure from a .cif file, orients the axes along the appropriate symmetry-defined directions, and calculates a point charge model.
It returns a Ligands object and a CFLevels object for further adjustments or fitting. (Currently, it only works for rare earth ions.)

Also allows the user to specify the axes if the ion is too low-symmetry for PyCrystalField to identify a custom z or y axis.

### Update (Dec. 19, 2019) to version 2.0

Optimized the PyCrystalField neutron spectrum function for the effective J basis so that it runs over an order of magnitude faster. Fits that previously took 40 minutes now take 4 minutes.

# Description

This code is for doing single-ion crystal field calculations on Rare Earth ions using the Stevens Operators formalism. It can also do calculations on transition metal ions, though the code has not been cross-checked in that regime. The math is based on M. Hutchings, Solid State Physics, 16 , 227 (1964).

## Features

PyCrystalField can generate a crystal electric field (CEF) Hamiltonian in two ways: (1) calculating the Hamiltonian for a given set of CEF parameters, or (2) calculating the Hamiltonian from an effective point-charge model.

From the Hamiltonian, PyCrystalField can then calculate:
- Eigenvalues (energy spectrum) and eigenvectors
- Temperature and field-dependent single-ion magnetization
- Temperature and field-dependent magnetic susceptibility
- Temperature, energy, and momentum-dependent neutron spectrum (using the dipole approximation)

PyCrystalField can also fit the crystal field Hamiltonian to data. The user can define an arbitrary ChiSquared function based on any observables (neutron data, susceptibility, magnetization, or eigenvalues) and PyCrystalField will minimize the ChiSquared function by varying a defined set of CEF parameters. Alternatively, PyCrystalField can fit effective charges of a point charge model.

All calculations can be carried out in the strong spin-orbit coupling regime (usling effective J states), the intermediate spin-orbit coupling regime (treating spin orbit interactions non-perturbatively in the L and S basis), or the weak spin-orbit coupling regime (dealing with effective l states).
