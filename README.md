# PyCrystalField
Code to calculate the crystal field Hamiltonian of magnetic ions.

Created by Allen Scheie

   scheie@lanl.gov

Please cite  Scheie, A. "PyCrystalField: Software for Calculation, Analysis, and Fitting of Crystal Electric Field Hamiltonians", J. Appl. Cryst. 54. (2021).  https://doi.org/10.1107/S160057672001554X

For a method of determining uncertainty for crystal field fits, see Scheie, A. "Quantifying uncertainties in crystal electric field Hamiltonian fits to neutron data", SciPost Physics Core, 5(1), 018. (2022). https://doi.org/10.21468/SciPostPhysCore.5.1.018


#### To download and use PyCrystalField, run the following command in a prompt/terminal window:
```
pip install git+https://github.com/asche1/PyCrystalField.git@master
```
PyCrystalField requires Python 3; downloading Anaconda is recommended, as PyCrystalField requires scipy, matplotlib, and numba to be installed.

## For documentation, see [here](https://github.com/asche1/PyCrystalField/wiki)

### Update (April 18, 2025) to version 2.3.11

Added CFLevels.Lig.LatticeTransformM to the CFLevels object upon importing the .cif file. This matrix transforms from Cartesian space to ABC space.

### Update (March 21, 2025) to version 2.3.10b

Added many new transition metal ion radial integrals and spin orbit coupling constants. Also added new function `checkTMexist()` used to check if all three parameters required for calculation are known for a given transition metal ion.

### Update (October 10, 2024) to version 2.3.10

Fixed rounoff error in ligand locations which caused some CEF parameters to be nonzero when symmetry constrained them to be zero. 

### Update (December 21, 2023) to version 2.3.9

Fixed bug in printEigenvectors function, added U4+ and U3+ to list of ions available for the point charge model. 

### Update (March 10, 2023) to version 2.3.8

Fixed bug in importCIF function, can now handle some non-standard CIF files.

### Update (January 26, 2023) to version 2.3.6

Fixed roundoff error issue in point charge calculations.

### Update (March 16, 2022) to version 2.3.2

Fixed error in neutrion spectrum calculations for the LS_CFLevels class.

### Update (May 25, 2021) to version 2.3

Added ability of importCIF to handle "near-symmetries". If no rotation or mirror symmetries are found, it will search for near-rotation symmetries with continuous shape measures (CSM) in order to define the quantization (Z) axis. If no close symmetries are found, it calculates the moment of intertia tensor of the ligands and uses the principal axes to define the Z axis.

Other small bugs in importCIF were fixed as well.

### Update (May 14, 2021) to version 2.2.2

Fixed bug in g tensor calculations so that Lande g factor is included in rare earth ions.

Modified importCIF to (a) import only the first phase listed in a .cif file, (b) treat the first listed rare earth ion as the central ion if none is specified, (c) import multiply defined ligand positions, outputting all possible structures, (d) allow the user to specify a certain number of ions in the coordination sphere rather than categorizing them by ion type (which is still the default).

### Update (Dec. 1, 2020) to version 2.2.1

Fixed bug in magnetization calculations so that M_y is calculated properly. (Oct 15 update introduced a bug which set J_y imaginary components to real when calculating magnetization.)

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

