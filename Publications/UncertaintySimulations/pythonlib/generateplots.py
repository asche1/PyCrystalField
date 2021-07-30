import numpy as np
import matplotlib.pyplot as plt
import plotformat as pf


def sigfigs(unc):
	sigdigits = int(-np.log10(unc)+1)
	if unc > 10:
		if str(unc)[0] != '1':
			sigdigits -= 1
	if (unc >= 1) and (unc < 10):
		if str(unc)[0] == '1':
			sigdigits += 1
	return sigdigits

class bestfit:
	def __init__(self, material, ion):
		simdata = np.genfromtxt(material+'/SimulatedData/'+ion+'TO_simdat.txt', unpack=True)
		self.simdataX = simdata[0]
		self.simdata10 = simdata[np.array([1,3])]
		self.simdata200 = simdata[np.array([2,4])]

		bffile = material+'/SimulatedData/'+ion+'TO_BestFitDat.txt'
		fitdata = np.genfromtxt(bffile, unpack=True)
		self.bestfitX = fitdata[0]
		self.bestfit6 = fitdata[np.array([1,2,3])]
		self.bestfit200 = fitdata[np.array([4,5,6])]
		with open(bffile) as f:
			firstline = f.readline()[1:].split(',')

		self.G = [] # best fit G tensor
		for i in range(3):
			self.G.append([float(v) for v in firstline[i].replace(']','').replace('[','').split()])



# BF = bestfit('Pyrochlore','Yb')

# f, ax = plt.subplots(2,1)
# BF.plotfit(ax)
# plt.show()