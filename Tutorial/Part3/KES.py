import numpy as np
import matplotlib.pyplot as plt
import PyCrystalField as cef

########### Import CIF file

KESLig, Er = cef.importCIF('KErSe2.cif')

########### print eigenvectors

Er.printEigenvectors() 



########### calculate magnetization at 2 K

temp = 2 #temperature in K

FieldStrengths = np.linspace(-10,10,101)
magnetization = np.zeros((len(FieldStrengths), 3, 3))
ion='Er3+'
for i, fs in enumerate(FieldStrengths):
    magnetization[i,0] = Er.magnetization(ion, temp, [fs,0,0]) # field along X
    magnetization[i,1] = Er.magnetization(ion, temp, [0,fs,0]) # field along Y
    magnetization[i,2] = Er.magnetization(ion, temp, [0,0,fs]) # field along Z
    



########### Calculate temperature dependent susceptibility 
### in a 0.5 T field 


Temperatures = np.arange(2,300,2)

# # For single crystal susceptibility, Field is a vector
# CalcSuscep = Er.susceptibility(ion='Er3+', Temps=Temperatures, 
# 						       Field=[0.5, 0 ,0], deltaField=0.001)

# For powder average susceptibility, Field is a scalar
CalcSuscep = Er.susceptibility(ion='Er3+', Temps=Temperatures, 
								Field=0.5, deltaField=0.001)



######## Plot results:

f, ax = plt.subplots(1,2, figsize=(7,3))

# Plot only the diagonal components of magnetization
ax[0].plot(FieldStrengths, magnetization[:,0,0], label='X')
ax[0].plot(FieldStrengths, magnetization[:,1,1], label='Y')
ax[0].plot(FieldStrengths, magnetization[:,2,2], label='Z')

ax[0].legend()
ax[0].set_xlabel('$B$ (T)')
ax[0].set_ylabel('$M$ ($\\mu_B$)')

# Plot the powder average susceptibility
ax[1].plot(Temperatures, -1/CalcSuscep)
ax[1].set_xlabel('$T$ (K)')
ax[1].set_ylabel('$\\chi^{-1}$ $(\\mu_B / T\\cdot$ion$)^{-1}$')

plt.tight_layout()
plt.show()