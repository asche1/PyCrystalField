import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from Uncertainty import stringUncertainty, stringUncertaintyAniso
import PyCrystalField as cef

import sys
from pythonlib.functions import *

ions = ['Sm3+','Pm3+','Nd3+','Ce3+','Dy3+','Ho3+','Tm3+','Pr3+','Er3+','Tb3+','Yb3+']
nn = 3


for ion in ions:
	print('*'*40, ion)

	filename = ion[:2]+'TO'

	lowChisqP = np.load('fitresults/{}{}_BestFit_uncertainty_P.npy'.format(filename,nn))
	lowChisqChi = np.load('fitresults/{}{}_BestFit_uncertainty_Chi.npy'.format(filename,nn))
	lowChisqGtensor = np.load('fitresults/{}{}_BestFit_uncertainty_Gtensor.npy'.format(filename,nn))
	lowChisqEvals = np.load('fitresults/{}{}_BestFit_uncertainty_Evals.npy'.format(filename,nn))
	lowChisqEvecs = np.load('fitresults/{}{}_BestFit_uncertainty_Evecs.npy'.format(filename,nn))

	bestfitB = lowChisqP[0][3:]

	outputfile ='ProcessedOutput/{}{}_results.txt'.format(filename,nn)
	with open(outputfile, 'w') as fo:

		################# PLOT OUTPUT
		# f, ax = plt.subplots(4,3,figsize=(9,6), sharey=True)

		# sf = 1e3
		# markerprops = {'ls':'none', 'marker':'.', 'rasterized':True}

		# axes= ax.flatten()
		# for i, axx in enumerate(axes):
		# 	jind = i
		# 	try:
		# 		axx.plot(lowChisqP[:,jind]*sf, lowChisqChi, alpha=0.1, **markerprops)
		# 	except IndexError: pass

		# for i in range(3):
		# 	ax[i,0].set_ylabel('$\\chi^2_{red}$')


		################## OUTPUT LATEX TABLE
		print('#### CEF parameters', file=fo)

		## Output latex table
		ns = [2,4,4,6,6,6]
		ms = [0,0,3,0,3,6]

		columns = 2
		i = 0
		rows = int(6/columns)
		for jj in range(rows):
			for j in range(columns):

				try:
					J = bestfitB[i]
					unc = (np.max(lowChisqP[:,i]) - np.min(lowChisqP[:,i]))/2
				except IndexError: break
				    
				sigdigits = int(-np.log10(unc)+1)
				if -np.log10(unc)+1 < 0:
					sigdigits = int(-np.log10(unc))
				#             sigdigits += 1
				if str(np.around(unc,sigdigits))[0] == '1':
					sigdigits += 1
					if str(np.around(unc, sigdigits)*1e10)[0] == '9':
				 		sigdigits -= 1
				if sigdigits > 0:
					if str(np.around(unc,sigdigits))[-1] == '1':
						sigdigits += 1
				if sigdigits <= 0:
					print('$B_{'+str(ns[i])+'}^{'+str(ms[i])+'}=$  &  $', 
							int(np.around(J, sigdigits)), '\\pm', int(np.around(unc, sigdigits)), end = '$',
							file=fo)
				else:
					print('$B_{'+str(ns[i])+'}^{'+str(ms[i])+'}=$  &  $', 
						np.around(J, sigdigits), '\\pm', np.around(unc, sigdigits), end = '$', file=fo)

				if j < (columns-1): print(' & ', end = ' ', file=fo)
				i += rows
			i -= rows*columns
			i += 1
			print(' \\\\', file=fo)

		#####################
		## G tensor
		print('\n#### G-tensor', file=fo)
		try:
			optG = np.abs(lowChisqGtensor[0])
			maxG = np.max(np.abs(lowChisqGtensor), axis=0)
			minG = np.min(np.abs(lowChisqGtensor), axis=0)
			uncG = (maxG - minG)/2

			print(optG, maxG, minG)

			# for i in range(3):
			# 	print(' & '.join([stringUncertaintyAniso(optG[i,j], maxG[i,j], minG[i,j]) for j in range(3)]), 
			# 		end=' \\\\ \n', file=fo)
			for i in range(3):
				print(stringUncertaintyAniso(optG[i,i], maxG[i,i], minG[i,i]) , 
					end=' \n', file=fo)
		except IndexError: pass


		############################################################ Plot extreme fits
		###### Import data

		Data = SimDat('SimulatedData/{}_simdat.txt'.format(filename))

		#### GENERATE new Hamiltonian
		def simspec(x, T, pars):
			Blabels = ['B20', 'B40', 'B43', 'B60', 'B63', 'B66']
			YbGopt = cef.CFLevels.Bdict(Data.ion, dict(zip(Blabels,  pars[3:])))
			gammas= pars[:2]
			pref= pars[2]
			YbGopt.diagonalize()
			maxE = np.max(YbGopt.eigenvalues)*1.1 + 5
			RF = lambda x: 0.03*(maxE - x/1.3)
			return pref*YbGopt.normalizedNeutronSpectrum(Earray=x, Temp=T,
											ResFunc= RF, gamma=gammas[i])


		##### PLOT
		try:
			

			xxx = np.linspace(Data.hw[0], Data.hw[-1], 300)
			yyy = [xxx]
			f, ax = plt.subplots(len(Data.Temps),1, figsize=(7,3*len(Data.Temps)), sharey=True, sharex=True)
			for i,T in enumerate(Data.Temps):
				ax[i].errorbar(Data.hw, Data.II[i], Data.dII[i], marker='.', ls='none', color='k')

				simI = simspec(xxx, T, lowChisqP[0])

				simIminG = simspec(xxx, T, lowChisqP[np.argmin(np.abs(lowChisqGtensor[:,-1,-1]), axis=0)])
				simImaxG = simspec(xxx, T, lowChisqP[np.argmax(np.abs(lowChisqGtensor[:,-1,-1]), axis=0)])

				for j, simm in enumerate([simI, simIminG, simImaxG]):
					ax[i].plot(xxx, simm, zorder=3)
					yyy.append(simm)
					
			minGG = np.abs(lowChisqGtensor[np.argmin(np.abs(lowChisqGtensor[:,-1,-1]))])
			maxGG = np.abs(lowChisqGtensor[np.argmax(np.abs(lowChisqGtensor[:,-1,-1]))])
			optGG = np.abs(lowChisqGtensor[0])
			# print(np.diag(minG))
			#print(np.diag(minG))
			#print(np.diag(maxG))
			#print(np.diag(optG))
			header = str(np.diag(optG))  +', ' + str(np.diag(minG)) +', '+ str(np.diag(maxG)) 
			print(header)

			np.savetxt('SimulatedData/{}_BestFitDat.txt'.format(filename), np.array(yyy).T, header=header)
			f.subplots_adjust(wspace=0.01, hspace=0.01)
			plt.savefig('SimulatedData/{}_BestFitDat.png'.format(filename))
		except IndexError: pass



		######################## PRINT LATEX TABLE OF BEST FIT

		print('\n#### best fit Eigenvalues', file=fo)

		## Eigenvalues and eigenvectors
		optVal = lowChisqEvals[0]
		maxVal = np.max(np.abs(lowChisqEvals), axis=0)
		minVal = np.min(np.abs(lowChisqEvals), axis=0)
		uncVal = (maxVal - minVal)/2

		
		## Because of the Kramers doublet, the eigenvalue doesn't always come out the same way. 
		## We'll have to loop through and sort the eigenvectors...
		JZ = cef.Operator.Jz(cef.Jion[ion][-1])
		sortedLowChisqEvecs = deepcopy(lowChisqEvecs)
		for i in range(len(lowChisqEvals)):
			for j in range(len(lowChisqEvals[i])-1):
				if np.abs(lowChisqEvals[i][j] - lowChisqEvals[i][j+1]) < 1e-4:
					ev1 = lowChisqEvecs[i][j]
					ev2 = lowChisqEvecs[i][j+1]
					Jexp1 = np.real(np.dot(ev1,np.dot(JZ.O,ev1)))
					Jexp2 = np.real(np.dot(ev2,np.dot(JZ.O,ev2)))
					if Jexp1 < Jexp2:
						sortedLowChisqEvecs[i][j] = lowChisqEvecs[i][j+1]
						sortedLowChisqEvecs[i][j+1] = lowChisqEvecs[i][j]

		optVec = sortedLowChisqEvecs[0]

		maxVec = np.max(np.abs(sortedLowChisqEvecs), axis=0)
		minVec = np.min(np.abs(sortedLowChisqEvecs), axis=0)
		uncVec = (maxVec - minVec)/2

		### Print LaTex table		    
		printLaTexEigenvectors(optVal, uncVal, optVec, uncVec, fo)


		## Now, print just ground state ket
		print('\n#### Ground State Ket', file=fo)
		print(printGSket(optVec, uncVec), file=fo)


	### Print G tensor to file
