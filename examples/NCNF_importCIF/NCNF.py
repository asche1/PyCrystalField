import numpy as np
import matplotlib.pyplot as plt

import pycrystalfield as cef

########### Import CIF file

NCNFLig, Ni = cef.importCIF('NaCaNi2F7.cif','Ni1', ionS = 1, ionL = 3)

########### print eigenvectors

Ni.printEigenvectors()

