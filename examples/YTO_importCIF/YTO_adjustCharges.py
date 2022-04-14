import numpy as np
import matplotlib.pyplot as plt

import pycrystalfield as cef

########### Import CIF file

YTOLig, Yb1 = cef.importCIF('yto.cif','Yb1', MaxDistance=4)
print(YTOLig.bondlen)
Yb2 = YTOLig.PointChargeModel(LigandCharge = np.hstack(([-1.66]*8, [3.33]*18)))

# ########### print eigenvectors

Yb1.printEigenvectors()
Yb2.printEigenvectors()


########### plot neutron spectrum

hw = np.linspace(0,100,200)
intens1 = Yb1.neutronSpectrum(hw, Temp=20, Ei=120, ResFunc= lambda x: 4, gamma = 1)
intens2 = Yb2.neutronSpectrum(hw, Temp=20, Ei=120, ResFunc= lambda x: 4, gamma = 1)

plt.figure()
plt.plot(hw, intens1, intens2)
plt.show()