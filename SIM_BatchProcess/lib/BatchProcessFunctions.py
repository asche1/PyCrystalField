import numpy as np
import glob
import subprocess
import PyCrystalField as cef
from copy import deepcopy


###################################
###################################
###################################

def ligandDistances(ciffile):
	cif = cef.CifFile(ciffile)
	for asuc in cif.asymunitcell:
		if asuc[1].strip('3+') in ['Sm','Pm','Nd','Ce','Dy','Ho','Tm','Pr','Er','Tb','Yb']:
			onesite = asuc
			break

	## Check for multiply defined atoms
	differentPositionsA = []
	differentPositionsB = []
	for ii, at in enumerate(cif.unitcell):
		if at[4] < 0: print('negative atom!',ii, at)

		if at[0][-1] in ["'", "B", "b"]:
			differentPositionsA.append(at[0])
			differentPositionsB.append(at[0].replace("'","").replace("B","A").replace("b","a"))

	if len(differentPositionsA) > 0:
		cif_a = deepcopy(cif)
		cif_b = deepcopy(cif)

		unitcellA = []
		unitcellB = []
		for ii, at in enumerate(cif.unitcell):
			if at[0] in differentPositionsA:
				unitcellA.append(at)
			elif at[0] in differentPositionsB:
				unitcellB.append(at)
			else:
				unitcellA.append(at)
				unitcellB.append(at)

		cif_a.unitcell = unitcellA
		cif_b.unitcell = unitcellB

		cifs = [cif, cif_a, cif_b]
	else:
		cifs = [cif]


	unitcellshifts = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], 
										[-1,0,0],[0,-1,0],[0,0,-1],
										[1,1,0], [0,1,1], [1,0,1],
										[-1,1,0],[0,-1,1],[1,0,-1],
										[1,-1,0],[0,1,-1],[-1,0,1],
										[-1,-1,0],[0,-1,-1],[-1,0,-1],
								[1,1,1],    [1,1,-1],  [1,-1,1], [-1,1,1],
								[-1,-1,-1], [-1,-1,1], [-1,1,-1],[1,-1,-1]])
	#neighborlist = []
	distlist = []
	for ii, at in enumerate(cifs[0].unitcell):
		if at[4] < 0: print('negative atom!',ii, at)
		#if ion not in at[0]:
		for ucs in unitcellshifts:
			distVec0 = cifs[0].latt.cartesian(np.array(onesite[2:5]) - (np.array(at[2:5]) + ucs))
			#neighborlist.append([at[1], np.linalg.norm(distVec0), distVec0])
			distlist.append(np.linalg.norm(distVec0))
	return np.sort(distlist)


def centerofmass(ligobj):
	return np.linalg.norm(np.mean(ligobj.bonds, axis=0))/np.max(ligobj.bondlen)


###################################
###################################
###################################


# Import the SIMDAVIS database

class SIMDAVIS:
	def __init__(self, infile):
		self.data = {}
		self.datalabels = []
		rawdata = []
		with open(infile) as f:
			lines = f.readlines()
			for i,line in enumerate(lines):
				if i == 0:
					for dl in line.strip().split('\t'):
						self.data[dl] = [np.nan]
						self.datalabels.append(dl)

				else:
					for j,dat in enumerate(line.strip().split('\t')):
						dl = self.datalabels[j]
						self.data[dl].append(dat)

SD = SIMDAVIS('lib/SIMDAVIS_dataset.tsv')

###################################
###################################
###################################


shapeRefShapes = {2:3, 3:4, 4:4, 5:5, 6:5, 7:7, 8:13, 9:13, 10:13, 11:7, 12:13, 20:1, 24:2, 48:1, 60:1}

def exportSHAPEfile(obj,title,outfile):
	'''take a ligands object and output a shape file'''
	nlig = len(obj.bonds)
	with open(outfile,'w') as f:
		f.write('$ '+ title + ' \n')
		f.write('! Ligands  CentralIon \n')
		f.write(' '*5 + str(nlig) + ' '*5 + str(nlig+1) + '  \n')
		f.write('! Possible structures \n')
		for i in range(shapeRefShapes[nlig]):
			f.write(' '*8 + str(i+1))
		f.write(' \nIONLIST\n') # name of file

		# Write the ions to the file
		for i in range(nlig):  # Just call each ligand an oxygen even if it's not.
			f.write('  O   ' + '  '.join(['{0:0.4f}'.format(ll) for ll in obj.bonds[i]]) + ' \n')
		f.write('  '+ obj.ion[:-2] +'   '+ '  '.join(['{0:0.4f}'.format(ll) for ll in [0.,0.,0.]]) + ' \n')

	## Run shape
	cmd = ['shape_2.1_linux64', outfile]
	process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
	for line in process.stdout:
		print(line)

	## import shape output and get closest symmetry shape
	shapenamelist = []
	shapeoutfile = outfile[:-3] + 'tab'
	with open(shapeoutfile) as f:
		lines = f.readlines()
		ii = 0
		while ii < len(lines):
			line = lines[ii]

			## Import dictionary
			if line.startswith(title):
				ii += 1
				while True:
					ii += 1
					line = lines[ii]
					if line.startswith('\n'):
						break
					else:
						shapenamelist.append(line.strip().split('   ')[-1])

			## Find minimum continuous shape measure
			if 'IONLIST' in line:
				CSMs = [float(v) for v in line.split(',')[1:]]
				minCSM = np.argmin(CSMs)

			ii += 1
	return shapenamelist[minCSM] + ', CSM='+str(CSMs[minCSM])


