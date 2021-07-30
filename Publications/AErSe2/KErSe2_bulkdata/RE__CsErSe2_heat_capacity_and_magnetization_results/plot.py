import numpy as np
import matplotlib.pyplot as plt

class PPMS:
	def __init__(self,infile):
		with open(infile) as f:
			lines = f.readlines()
			for i,l in enumerate(lines):
				if 'WEIGHT' in l:
					self.weight = float(l.split(',')[2].strip('mg\n'))
				if l.startswith('[Data]'):
					firstline = i+2
		
		self.data = np.genfromtxt(infile, skip_header=firstline, delimiter=',', 
			unpack=True)
		#print(self.data)


KESab = PPMS('KErSe2_Hparatoplate_12.07.2019.dc.dat')
KESc = PPMS('KErSe2_Hpertoplate_12.07.2019.dc.dat')


# ll, ul = 479, -1
# ll,ul = 127, -1
ll, ul = 4, 130-32


KESmagAB = [KESab.data[2][ll:ul], KESab.data[4][ll:ul]]
KESmagC = [KESc.data[2][ll:ul], KESc.data[4][ll:ul]]


for i in range(2):
	KESmagAB[i] = np.mean(KESmagAB[i].reshape(-1,2),axis=1)
	KESmagC[i] = np.mean(KESmagC[i].reshape(-1,2),axis=1)



# Scale
wt = 0.000560
molecularwt = 39.0983 + 167.259 + 2*78.971
Na = 6.02214076e23 
scalefact = molecularwt/wt*(1.07828221e20/Na)

KESmagAB[1] *= scalefact
KESmagC[1]*= scalefact
KESmagAB[0] *= 1e-4
KESmagC[0]*= 1e-4

np.savetxt('KES_Magnetization.txt', np.vstack([KESmagAB, KESmagC]))

print(np.vstack([KESmagAB, KESmagC]))

plt.figure()
# # susceptibility
# plt.plot(KESab.data[3][ll:ul], 1/KESab.data[4][ll:ul])
# plt.plot(KESc.data[3][ll:ul], 1/KESc.data[4][ll:ul])
# Magnetization
# plt.plot(KESab.data[2][ll:ul], KESab.data[4][ll:ul])
# plt.plot(KESc.data[2][ll:ul], KESc.data[4][ll:ul])

plt.plot(KESmagAB[0], KESmagAB[1], marker='o')
plt.plot(KESmagC[0], KESmagC[1], marker='o')
plt.show()