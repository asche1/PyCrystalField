import numpy as np
import matplotlib.pyplot as plt

MTdata = np.genfromtxt('MT.dat', delimiter='\t', unpack=True, skip_header=2)

plt.figure()
plt.plot(MTdata[0], MTdata[2])
#plt.plot(MTdata[0], MTdata[5])
#plt.plot(MTdata[0], MTdata[9])
plt.show()
