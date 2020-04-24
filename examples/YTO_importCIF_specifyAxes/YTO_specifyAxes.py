import numpy as np
import matplotlib.pyplot as plt

import PyCrystalField as cef

########### Import CIF file

# We can specify the Z axis of the point charge model if we choose.
#    PyCrystalField will then find a y axis orthogonal to it with a 
#    mirror plane (if one exists)
# You can also specify the y axis if you want with "Yaxis = [ , , ]"
YTOLig, Yb = cef.importCIF('yto.cif','Yb1', Zaxis = [1,1,0])

########### print eigenvectors

Yb.printEigenvectors()


########### plot neutron spectrum

hw = np.linspace(0,100,200)
intens = Yb.neutronSpectrum(hw, Temp=20, Ei=120, ResFunc= lambda x: 4, gamma = 1)

plt.figure()
plt.plot(hw, intens)
plt.show()