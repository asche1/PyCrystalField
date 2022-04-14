import numpy as np
import matplotlib.pyplot as plt

import pycrystalfield as cef

########### Import CIF file

YTOLig, Yb = cef.importCIF('yto.cif')

########### print eigenvectors

Yb.printEigenvectors()


########### plot neutron spectrum

hw = np.linspace(0,100,200)
QQ = np.linspace(1,6,50)
intens = Yb.neutronSpectrum2D(hw, QQ, Temp=20, Ei=120, ResFunc= lambda x: 4, gamma = 1, Ion = 'Yb3+')

plt.figure()
plt.pcolormesh( QQ, hw, intens)
plt.show()