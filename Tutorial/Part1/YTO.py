import numpy as np
import matplotlib.pyplot as plt
import PyCrystalField as cef

########### Import CIF file

YTOLig, Yb = cef.importCIF('Yb2Ti2O7.cif')

########### print eigenvectors

Yb.printEigenvectors() 

########### plot neutron spectrum

hw = np.linspace(0,100,200)

intens = Yb.normalizedNeutronSpectrum(hw, Temp=20, 
	ResFunc= lambda x: 4, gamma = 1)

plt.figure()
plt.plot(hw, intens)
plt.show()


########### print g tensor

g = Yb.gtensor()
print('G tensor:\n',g)