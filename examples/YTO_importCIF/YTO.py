import numpy as np
import matplotlib.pyplot as plt

import PyCrystalField as cef

########### Import CIF file

YTOLig, Yb = cef.importCIF('yto.cif','Yb1')
Yb.diagonalize()

########### print eigenvectors

Yb.printEigenvectors()


########### plot neutron spectrum

hw = np.linspace(0,100,200)
intens = Yb.neutronSpectrum(hw, Temp=20, Ei=120, ResFunc= lambda x: 4, gamma = 1)

plt.figure()
plt.plot(hw, intens)
plt.show()