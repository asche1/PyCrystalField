import numpy as np
import PyCrystalField as cef
import glob
import subprocess
from lib.BatchProcessFunctions import exportSHAPEfile, SD, ligandDistances, centerofmass

## Find all the cif files in the /ciffiles directory
ciffiles = []
cifnames = []
#for file in glob.glob("ciffiles/*.cif"):
for file in glob.glob("CIFs-initset/*.cif"):
    ciffiles.append(file)
    cifnames.append(file.split('/')[1])
print(ciffiles)

## Import the labels for these
labelsfilename = 'CIFs-initset/Data_Selected_SIM_samples.tsv'
#labelsfilename = 'CIFs-initset/test-dataset-cifs.tsv'

cifIDs = []
with open(labelsfilename) as f:
	lines = f.readlines()
	for i,cn in enumerate(cifnames):
		for line in lines:
			dd = line.strip().split('\t')
			if cn in dd[2]:
				cifIDs.append(int(dd[0]))
print(cifIDs)

## Sort cifs by cif_ID
ciffiles = [ciffiles[jj] for jj in np.argsort(cifIDs)]
cifIDs = np.sort(cifIDs)


# 
# print(SD.data['coordination_number'])

###################################
###################################
###################################



def printGSket(ket1, ket2=None):
	'''Select three largest components of ground state. Ket2 is if it is a Kramers ion'''
	lenk = len(ket1)
	if lenk%2 ==0: #kramers ion
		labelrange = np.arange(-(lenk-1), lenk, 2)
		labellist = ['|{}/2>'.format(jj) for jj in labelrange]
	else:
		labelrange = np.arange(int(-(lenk-1)/2), int((lenk+1)/2), 1)
		labellist = ['|{}>'.format(jj) for jj in labelrange]
	
	max3 = np.sort(np.argsort(np.abs(ket1))[-3:])
	try:
		max3b = np.sort(np.argsort(np.abs(ket2))[-3:])
		print(ket1, ket2)
		ketlist1 = [ '({0:.3f})'.format(ket1[m])+labellist[m] for m in max3]
		ketlist2 = [ '({0:.3f})'.format(ket2[m])+labellist[m] for m in max3b[::-1]]

		return 'psi_0+ = '+' '.join(ketlist1) + ' ,  ' +'psi_0- = '+' '.join(ketlist2)
	except TypeError:
		ketlist1 = [ '({0:.3f})'.format(ket1[m])+labellist[m] for m in max3]

		return 'psi_0 = '+' '.join(ketlist1)
	 

def printOversimplifiedGSket(ket1, ket2=None):
	'''Select largest component of ground state if any is above 0.4. Ket2 is if it is a Kramers ion'''
	lenk = len(ket1)
	if lenk%2 ==0: #kramers ion
		labelrange = np.arange(-(lenk-1), lenk, 2)
		labellist = ['|{}/2>'.format(jj) for jj in labelrange]
	else:
		labelrange = np.arange(int(-(lenk-1)/2), int((lenk+1)/2), 1)
		labellist = ['|{}>'.format(jj) for jj in labelrange]
	
	max1 = np.argsort(np.abs(ket1))[-1]
	if np.abs(ket1[max1]) > 0.4:
		try:
			max1b = np.argsort(np.abs(ket2))[-1]
			print(ket1, ket2)
			ketlist1 = '({0:.3f})'.format(ket1[max1])+labellist[max1]
			ketlist2 = '({0:.3f})'.format(ket2[max1b])+labellist[max1b]

			return 'psi_0+ = '+ketlist1 + ' ,  ' +'psi_0- = '+ketlist2
		except TypeError:
			ketlist1 = '({0:.3f})'.format(ket1[max1])+labellist[max1]

			return 'psi_0 = '+ketlist1
	else: return ' no weights above 0.4'


def findMainModulus(ket1, ket2=None):
	'''Select largest component of ground state if any is above 0.4. Ket2 is if it is a Kramers ion'''
	lenk = len(ket1)
	if lenk%2 ==0: #kramers ion
		labelrange = np.arange(-(lenk-1), lenk, 2)
		labellist = ['{}/2'.format(jj) for jj in labelrange]
	else:
		labelrange = np.arange(int(-(lenk-1)/2), int((lenk+1)/2), 1)
		labellist = ['{}'.format(jj) for jj in labelrange]
	
	max1 = np.argsort(np.abs(ket1))[-1]
	if np.abs(ket1[max1]) > 0.4:
		return'{0:.4f}'.format(np.abs(ket1[max1])), labellist[max1]

	else: return ' no weights above 0.4'


####################################
####################################
####################################


outputfile = 'calculatedOutput.tsv'

with open(outputfile, 'w') as fo:
		fo.write('# Output for PyCrystalField batch processing of files\n'+\
					'# Allen Scheie \tOctober, 2021\n#'+\
					'cif name \t sample_ID \t highest excited CEF level (meV) \tGS eigenstate'+\
					'\toversimplified GS eigenstate'+ '\tmain modulus'+ '\tmain m_J'+\
					' \tClosest polyhedron \tCSM \tCoordination number \tcoordination sphere \n')

## Loop through the cif files, calculate the crystal fields

for i,cf in enumerate(ciffiles):
	#if i != (158-4): continue # Pick out specific file
	#if cf != 'cifs_corrected/1894450.cif': continue

	print('*'*50, cf)
	cifname = cf.split('/')[1]
	#YTOLig, Yb = cef.importCIF(cf, NumIonNeighbors=3)
	sample_ID = cifIDs[i]
	name = str(sample_ID)
	try:
		coordnum = int(SD.data['coordination_number'][sample_ID])
		if 'Carbon' in SD.data['coordination_elements'][sample_ID]: 
			raise ValueError
		elif 'carbon' in SD.data['coordination_elements'][sample_ID]: 
			raise ValueError
	except ValueError:
		coordnum = None

	print('CoordinationNumber:', coordnum)

	####### Here insert ad-hoc fixes: ########
	if cf == 'cifs_corrected/1579114.cif':
		coordnum = 12
	elif cf == 'cifs_corrected/1894450.cif':
		coordnum = 4*16-1
	###########################################

	try:
		### 
		try:
			PCFoutput = cef.importCIF(cf, NumIonNeighbors=1, CoordinationNumber=coordnum)
		except TypeError:
			print('Unknown ion --- trying to replace...')
			repIon = SD.data['Ln_ion'][sample_ID]
			modcfname = cf[:-4] +'_mod.cif'
			print(repIon)
			with open(cf, 'r') as cff, open(modcfname, 'w') as cfo:
				lines = cff.readlines()
				for i,line in enumerate(lines):
					newline = line.replace('Gd3+',repIon).replace('Y3+',repIon).replace('La3+',
						repIon).replace(' Gd '," "+repIon+" ").replace(' Y ',
						" "+repIon+" ").replace(' La '," "+repIon+" ")

					for ii in range(1,10):
						newline = newline.replace('Y{}'.format(ii), repIon[:-2]+'{}'.format(ii))
						newline = newline.replace('Gd{}'.format(ii), repIon[:-2]+'{}'.format(ii))
						newline = newline.replace('La{}'.format(ii), repIon[:-2]+'{}'.format(ii))
					cfo.write(newline)
			PCFoutput = cef.importCIF(modcfname, NumIonNeighbors=1, CoordinationNumber=coordnum)

		### See how many neighbors. If there's less than seven, redo.
		if coordnum == None:
			nin = 1
			while True:
				try:
					YTOLig, Yb = PCFoutput
				except ValueError:
					YTOLig = PCFoutput[0][0]
				if len(YTOLig.bonds) < 7:
					nin += 1
					PCFoutput = cef.importCIF(cf, NumIonNeighbors=nin, CoordinationNumber=coordnum)
				elif len(YTOLig.bonds) >= 20:
					break
				## Find the "center of mass" of the ligands. If greater than 25%, add a ligand.
				elif centerofmass(YTOLig) > 0.2:
					nin += 1
					PCFoutput = cef.importCIF(cf, NumIonNeighbors=nin, CoordinationNumber=coordnum)
				else: break

			## Calculated coordination number
			coordnum = len(YTOLig.bondlen)



		try:
			YTOLig, Yb = PCFoutput

			########### print eigenvectors
			Yb.diagonalize()
			try:
				print(Yb.gtensor()) # Issue here: what if GS is a singlet?
			except IndexError: pass
			except ValueError: pass
			Yb.printEigenvectors()

		except ValueError:
			YTOLigs = [po[0] for po in PCFoutput]
			Ybs = [po[1] for po in PCFoutput]

			Erange = []
			for Yb in Ybs:
				Yb.diagonalize()
				Erange.append(np.max(Yb.eigenvalues))
			## Check if anything has changed to within 0.2 meV
			if np.abs(Erange[-1] - Erange[-2]) < 0.2:
				Yb = Ybs[1]
				YTOLig = YTOLigs[1]
			# else: 
			# 	print('AAH!! skipping this file because it depends upon the ligand positions!!')
			# 	print('\t difference in energy range = ', Erange)
			# 	with open(outputfile, 'a+') as fo:
			# 		fo.write(' \t'.join([cifname, name, '{0:.2f}, or {1:.2f}'.format(Erange[-1],Erange[-2]),
			# 					 'NA', 'NA']) + '\n')

			# 	continue #skip this file
			# else:
			# 	print('This file because it depends upon the ligand positions;'+\
			# 		' taking the largest energy splitting...')
			# 	if Erange[1] > Erange[2]:
			# 		Yb = Ybs[1]
			# 		YTOLig = YTOLigs[1]
			# 		print('   E splitting', Erange[1])
			# 	else:
			# 		Yb = Ybs[2]
			# 		YTOLig = YTOLigs[2]
			# 		print('   E splitting', Erange[2])
			else:
				print('This file because it depends upon the ligand positions;'+\
					' taking the smallest energy splitting...')
				if Erange[1] < Erange[2]:
					Yb = Ybs[1]
					YTOLig = YTOLigs[1]
					print('   E splitting', Erange[1])
				else:
					Yb = Ybs[2]
					YTOLig = YTOLigs[2]
					print('   E splitting', Erange[2])


		### Calculate the N+1 max distance
		bondlenMin = np.min(YTOLig.bondlen)
		bondlenMax = np.max(YTOLig.bondlen)

		distlist = ligandDistances(cf)
		bondlenMax_Plus1 = distlist[coordnum + 1]
		print('BOND LENGTHS:', bondlenMin, bondlenMax, distlist[coordnum], bondlenMax_Plus1)


		#### Export to database file:
		# 0) Sample_ID
		# 1) Highest CEF excitation
		# 2) Ground state ket
		# 3) Oversimplified ground state ket
		# 4) Main modulus
		# 5) main mJ
		# 6) Shape file
		# 7) Coordination Sphere 

		
		## find highest energy level
		energylevelMAX = np.max(Yb.eigenvalues)

		## Find lowest energy eigenket 
		if len(Yb.eigenvectors)%2 == 0: #kramers
			GSket = printGSket(Yb.eigenvectors[0], Yb.eigenvectors[1])
			OSGSket = printOversimplifiedGSket(Yb.eigenvectors[0], Yb.eigenvectors[1])
			## Find lowest energy level
			energylevel1 = str(np.around(np.min(Yb.eigenvalues[Yb.eigenvalues > 1e-5]), 4))
		else:
			GSket = printGSket(Yb.eigenvectors[0])
			OSGSket = printOversimplifiedGSket(Yb.eigenvectors[0])
			energylevel1 = str(np.around(np.sort(Yb.eigenvalues[Yb.eigenvalues > 1e-5])[0],4)) +\
						', ' + str(np.around(np.sort(Yb.eigenvalues[Yb.eigenvalues > 1e-5])[1],4))

		try:
			mainmodulus, mainMJ = findMainModulus(Yb.eigenvectors[0])
		except ValueError:
			mainmodulus, mainMJ = 'NA', 'NA'


		## calculate closest polyhedron using SHAPE
		YTOLig.exportCif('shapefiles/'+name+'_ligands.cif')
		try:
			CSMstr = exportSHAPEfile(YTOLig, name, 'shapefiles/'+name+'ShapeFile.dat')
			print(CSMstr)
		except (KeyError, IndexError):
			print('\tSHAPE error; no shapes for',len(YTOLig.bonds),'atoms.')
			CSMstr = 'NA'
			coordnum = len(YTOLig.bondlen)

		## Create string of coordination sphere
		CoordSphere = ["'"+YTOLig.LigandNames[k]+"' " + str(np.around(YTOLig.bonds[k],4)) 
						for k in range(len(YTOLig.bonds))]

		## Write to file
		with open(outputfile, 'a+') as fo:
			fo.write(' \t'.join([cifname, name, str(np.around(energylevelMAX,4)), 
				GSket, OSGSket, mainmodulus, mainMJ, CSMstr.replace(', ','\t'), str(coordnum), 
				', '.join(CoordSphere)]) + '\n')
				#str(np.around(bondlenMax,5)), str(np.around(bondlenMax_Plus1,5))]) + '\n')

	except TypeError: 
		#print('Unknown ion --- skipping file.')
		#with open(outputfile, 'a+') as fo:
		#	fo.write(' \t'.join([cifname, name, 'NA', 'NA','NA','NA', 'NA', 'NA', 'NA', 'NA', 'NA']) + '\n')

		print('Unknown ion --- trying to replace Gd...')
