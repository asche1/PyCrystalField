# PyCrystalField
Code to calculate the crystal field Hamiltonian of magnetic ions.

Created by Allen Scheie

   scheie@jhu.edu

   https://sites.google.com/view/allen-scheie/



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
