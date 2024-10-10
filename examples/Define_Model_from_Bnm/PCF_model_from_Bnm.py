import numpy as np
import PyCrystalField as cef

'''This example shows how to create a CEF model from 
previously defined CEF parameters rather than from a .cif 
or point charge model. 
Allen Scheie, October 2024'''

############################################
## Example 1: Cubic symmetry

# First, create a dictionary of the parameters. For negative m, define e.g. "B4-4". 
Bnm = {'B20':-0.42, 'B40': 0.02, 'B44':0.13}  

# Next, use the Bdict class method to define a Hamiltonian.
# The ion and the dictionary must be specified. 
Nd3 = cef.CFLevels.Bdict(ion='Nd3+', Bdict=Bnm)

# Now, you can diagonalize, plot, fit, etc. just like any other PyCrystalField object. 
Nd3.printEigenvectors()


############################################
##  Example 2: three-fold symmetry: 

## Define Bnm
Bnm = {'B20':0.22, 'B40': 0.03, 'B43':0.011, 'B60': -0.0002, 'B63':0.0001, 'B66':0.00054}  
# Define Hamiltonian
Nd3 = cef.CFLevels.Bdict(ion='Nd3+', Bdict=Bnm)
# Print Eigenvectors
Nd3.printEigenvectors()