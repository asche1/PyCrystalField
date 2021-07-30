import numpy as np
import matplotlib.pyplot as plt
import PyCrystalField as cef




ions = ['Sm3+','Pm3+','Nd3+','Ce3+','Dy3+','Ho3+','Tm3+','Pr3+','Er3+','Tb3+','Yb3+']

for ion in ions:
	filename = ion[:2]+'TO'
	####### Create copies of cif files 
	with open('kes.cif') as f1:
		with open(filename+'.cif', 'w') as f2:
			for line in f1.readlines():
				f2.write(line.replace('Er3+',ion).replace('Er1',ion[:2]+'1'))


	########### Import CIF file

	YTOLig, Yb = cef.importCIF(filename+'.cif',ion[:2]+'1')

	########### print eigenvectors

	Yb.printEigenvectors()

	maxE = np.max(Yb.eigenvalues)*1.1 + 5
	########### plot neutron spectrum
	RF = lambda x: 0.03*(maxE - x/1.3)

	hw = np.linspace(5,maxE,200)
	Temps = [10,200]
	intens1 = Yb.normalizedNeutronSpectrum(hw, Temp=10, ResFunc= RF, gamma = 0.5*maxE/100)
	intens2 = Yb.normalizedNeutronSpectrum(hw, Temp=200, ResFunc= RF, gamma = 1.5*maxE/100)

	intens = [intens1, intens2]
	simI = []
	simdI = []


	######## Simulate noise and error bars
	plt.figure()

	err = 0.004
	BgNoise = np.random.normal(0,0.01,len(hw))
	for i in range(len(intens)):
		DataNoise = np.random.normal(intens[i], np.sqrt(intens[i])/50 + err)
		DataErr = np.sqrt(intens[i])/50 + err
		plt.errorbar(hw, DataNoise,DataErr)
		simI.append(DataNoise)
		simdI.append(DataErr)


	## Export simulated data
	simdat = np.vstack((hw, simI, simdI))
	np.savetxt('../SimulatedData/'+filename+'_simdat.txt', simdat.T,
		header = YTOLig.ion + ' '+ ' '.join([str(T) for T in Temps]))

	print(YTOLig.ion)

	plt.savefig('../SimulatedData/'+filename+'_simdat.png')