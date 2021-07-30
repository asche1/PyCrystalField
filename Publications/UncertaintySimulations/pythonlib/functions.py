import numpy as np
from Uncertainty import stringUncertainty

class SimDat:
	def __init__(self,infile):
		with open(infile) as f:
			header = f.readline()[2:].split(' ')
			self.ion = header[0]
			self.Temps = [float(T) for T in header[1:]]
		data= np.genfromtxt(infile, unpack=True)
		self.hw = data[0]
		self.II =  [data[i] for i in range(1, len(self.Temps)+1)]
		self.dII = [data[i] for i in range(len(self.Temps)+1, 2*len(self.Temps)+1)]



def printLaTexEigenvectors(EV, dEV, Vec, dVec, fo, precision=4):
	'''prints eigenvectors and eigenvalues in the output that Latex can read'''

	print('\\begin{table*}\n\\caption{Eigenvectors and Eigenvalues...}',file=fo)
	print('\\begin{ruledtabular}',file=fo)
	numev = len(EV)
	print('\\begin{tabular}{c|'+'c'*numev+'}',file=fo)
	if numev % 2 == 1:
		print('E (meV) &'+' & '.join(['$|'+str(int(kk))+'\\rangle$' for kk in 
					np.arange(-(numev-1)/2,numev/2)])
			+' \\tabularnewline\n \\hline ',file=fo)
	else:
		print('E (meV) &'+
			' & '.join(['$| -\\frac{'+str(abs(kk))+'}{2}\\rangle$' if kk <0
						 else '$| \\frac{'+str(abs(kk))+'}{2}\\rangle$'
						for kk in np.arange(-(numev-1),numev,2)])
			+' \\tabularnewline\n \\hline ',file=fo)
	    
	#sortinds = EV.argsort()
	sortEVal= np.around(EV,3)
	sortEVec= np.real(Vec)
	sortdEVal= np.around(dEV,3)
	sortdEVec= np.real(dVec)
	for i in range(numev):
		print(sortEVal[i], sortdEVal[i])
		print(stringUncertainty(sortEVal[i], sortdEVal[i]))
		print(stringUncertainty(sortEVal[i], sortdEVal[i]),'&', 
			' & '.join([stringUncertainty(eevv, sortdEVec[i][j]) 
						for j,eevv in enumerate(sortEVec[i])]), '\\tabularnewline',file=fo)
	print('\\end{tabular}\\end{ruledtabular}',file=fo)
	print('\\label{flo:Eigenvectors}\n\\end{table*}',file=fo)



def printGSket(Vec, dVec):
	'''Select three largest components of ground state. Ket2 is if it is a Kramers ion'''
	ket1, ket2 = np.real(Vec)[0], np.real(Vec)[1]
	dket1, dket2 = np.real(dVec)[0],  np.real(dVec)[1]
	lenk = len(ket1)
	if lenk%2 ==0: #kramers ion
		labelrange = np.arange(-(lenk-1), lenk, 2)
		labellist = ['|{}/2 \\rangle'.format(jj) for jj in labelrange]
	else:
		labelrange = np.arange(int(-(lenk-1)/2), int((lenk+1)/2), 1)
		labellist = ['|{} \\rangle'.format(jj) for jj in labelrange]

	max3 = np.sort(np.argsort(np.abs(ket1))[-3:])
	try:
		max3b = np.sort(np.argsort(np.abs(ket2))[-3:])
		ketlist1 = [stringUncertainty(ket1[m],dket1[m])+labellist[m] for m in max3]
		ketlist2 = [stringUncertainty(ket2[m],dket2[m])+labellist[m] for m in max3b[::-1]]

		return '\\psi_0+ = '+' + '.join(ketlist1) + ' ,  ' +'\\psi_0- = '+' + '.join(ketlist2)
	except TypeError:
		ketlist1 = [stringUncertainty(ket1[m],dket1[m])+labellist[m] for m in max3]

		return '\\psi_0 = '+' + '.join(ketlist1)