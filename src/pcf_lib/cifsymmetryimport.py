import numpy as np
import matplotlib.pyplot as plt
from pcf_lib.cif_import import CifFile
from pcf_lib.plotLigands import plotPCF
from pcf_lib.MomentOfIntertia import findZaxis

# def importCIF(ciffile, mag_ion):
# 	'''Call this function to generate a PyCrystalField point charge model
# 	from a cif file'''
# 	cif = CifFile(ciffile)
# 	for ii, at in enumerate(cif.unitcell):
# 		if at[4] < 0: print('negative atom!',ii, at)
# 	centralIon, ligandPositions, ligandCharge = FindPointGroupSymOps(cif, mag_ion)

# 	Lig = cef.Ligands(ion=centralIon, ionPos = [0,0,0], ligandPos = ligandPositions)
# 	# Create a point charge model, assuming that a mirror plane has been found.
# 	print('   Creating a point charge model...')
# 	PCM = Lig.PointChargeModel(printB = True, LigandCharge=ligandCharge[0], suppressminusm = True)

# 	return Lig, PCM


def FindPointGroupSymOps(self, ion, Zaxis = None, Yaxis = None, crystalImage = False, 
						NumIonNeighbors = 3, CoordinationNumber=None, maxDistance = None):
	# Step 1: identify the ion in the asymmetric unit cell
	site = []
	for i,auc in enumerate(self.asymunitcell):
		if ion in auc[0]:
			site.append(auc)

	if len(site) > 1:
	 	raise AttributeError("\033[41m"+"More than one "+ion+" ion in the .cif file.\n"+\
	 						'  Try specifying "'+'" and "'.join([s[0] for s in site])+'" individually.\n'+\
	 						"      For example, importCIF('filename.cif', '"+site[0][0]+"')\n"+\
	 						"                   importCIF('filename.cif', '"+site[1][0]+"')" + "\033[0m\n")
	onesite = site[0]
	centralIon = onesite[1]
	print('Central ion:', onesite[1], 'at', onesite[2:5])

	# Step 2: unfold the crystal
	PGS = []    # PGS is short for point group symmetry
	for sy in self.symops:
		new_at = self.SymOperate(sy,onesite)
		if np.all(new_at[2:5] == onesite[2:5]):
			PGS.append(sy)



	############### Now, find the nearest neighbors
	# Step 1: make a list of all the nearest neighbor distances and vectors
	unitcellshifts = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], 
										[-1,0,0],[0,-1,0],[0,0,-1],
										[1,1,0], [0,1,1], [1,0,1],
										[-1,1,0],[0,-1,1],[1,0,-1],
										[1,-1,0],[0,1,-1],[-1,0,1],
										[-1,-1,0],[0,-1,-1],[-1,0,-1],
								[1,1,1],    [1,1,-1],  [1,-1,1], [-1,1,1],
								[-1,-1,-1], [-1,-1,1], [-1,1,-1],[1,-1,-1]])
	neighborlist = []
	distlist = []
	for ii, at in enumerate(self.unitcell):
		if at[4] < 0: print('negative atom!',ii, at)
		#if ion not in at[0]:
		for ucs in unitcellshifts:
			distVec0 = self.latt.cartesian(np.array(onesite[2:5]) - (np.array(at[2:5]) + ucs))
			## Get rid of any ions which are closer than 0.4 \AA. This is unphysical.
			if np.linalg.norm(distVec0) > 0.4: 
				neighborlist.append([at[1], np.linalg.norm(distVec0), distVec0])
				distlist.append(np.linalg.norm(distVec0))
			else: 
				print('    AAAH! There is a super-close atom. Removing it...')
	
	# Step 2: sort the list in ascending order
	sortedNeighborArgs = np.argsort(distlist)

	############# If max distance is specified, use this
	if maxDistance != None:
		CoordinationNumber = np.sum(np.array(distlist) < maxDistance)

	########### If coordination number is specified, only take those ions
	if CoordinationNumber != None:
		minindex = np.min(np.where(np.sort(distlist) > 1e-4))
		nearestNeighbors = [neighborlist[v] for v in sortedNeighborArgs[minindex:CoordinationNumber+minindex]] 
		NNLigandList = [neighborlist[v][0] for v in sortedNeighborArgs[minindex:CoordinationNumber+minindex]]

		for i, nnll  in enumerate(list(set(NNLigandList))):
			numN = [nn[0] for nn in nearestNeighbors].count(nnll)
			print('   Identified', numN, nnll,'ligands.')


	# otherwise, we search through neighbors by common ions.
	else:
		nearestNeighbors = []
		### Find the nearest neighbor ligands that are not on the same site
		jjj, kkk = 0,0
		NNLigandList = []
		while len(NNLigandList) < NumIonNeighbors:
			if neighborlist[sortedNeighborArgs[jjj]][1] > 1e-4:
				#nearestLigand = neighborlist[sortedNeighborArgs[jjj]][0]
				#break
				try:
					if neighborlist[sortedNeighborArgs[jjj]][0] == NNLigandList[-1]:
						pass
					else:
						NNLigandList.append(neighborlist[sortedNeighborArgs[jjj]][0])
				except IndexError:  NNLigandList.append(neighborlist[sortedNeighborArgs[jjj]][0])
				# print(neighborlist[sortedNeighborArgs[jjj]][0], NNLigandList)
			else: 
				kkk += 1   #for keeping track of where the first ligand is.
			jjj+=1


		for i in range(NumIonNeighbors):
			print(' Next'*i+' Nearest ligand:', NNLigandList[i])
		#nearestLigand = NNLigandList[0]
		addedLigands = []
		for sna in sortedNeighborArgs[kkk:]:
			#if neighborlist[sna][0] == nearestLigand:
			if neighborlist[sna][0] in NNLigandList:

				try:
					if neighborlist[sna][0] == addedLigands[-1]: 
						pass
					else: addedLigands.append(neighborlist[sna][0])
				except IndexError:
					addedLigands.append(neighborlist[sna][0])
				if (len(addedLigands) > NumIonNeighbors): 
					break
				nearestNeighbors.append(neighborlist[sna])
			else: break
			#if len(nearestNeighbors) >= jjj: break

		for i in range(len(list(set(NNLigandList)))):
			numN = [nn[0] for nn in nearestNeighbors].count(NNLigandList[i])
			print('   Identified', numN, NNLigandList[i],'ligands.')


	###############

	# print(PGS)
	# Step 3: make the symmetry operations matrices and find the rotation axes
	inversion = False  #this is so that PyCrystalField knows whether to include -m terms
	RotAngles = []
	RotAxes = []
	Mirrors = []
	for pgs in PGS:
		mat = makeSymOpMatrix(self, pgs)
		rotmir = findRotationAxis(self, mat)
		#print(pgs, mat, rotmir)
		if rotmir[0] == 'rot':
			RotAngles.append(rotmir[1])
			RotAxes.append(rotmir[2])
		elif rotmir[0] == 'mirr':
			Mirrors.append(rotmir[1].flatten())
			inversion = True
		# elif rotmir[0] == 'inversion': pass
			


	## Step 3a: identify the axes
	if np.any(Zaxis == None) and np.any(Yaxis == None):

		# Step 4: Find the highest-fold rotation axes, and set to Z axis
		try:
			ZAXIS =  RotAxes[np.argmax(RotAngles)]
			print('   Found',int(np.around(np.max(RotAngles),0)),'fold axis about',ZAXIS)

				# Step 5: find a mirror plane orthogonal to the rotation axis and set to be the Y axis
			for i, M in enumerate(Mirrors):
				if np.dot(self.latt.cartesian(M),self.latt.cartesian(ZAXIS)) == 0:
					YAXIS = M
					print('   Found mirror plane:',YAXIS)
					break
		except ValueError: pass


		try:
			ZAXIS, YAXIS
		except UnboundLocalError:
			try:
				if len(Mirrors) > 0:
					YAXIS = Mirrors[0]
					#perpvec = np.cross(self.latt.cartesian(Mirrors[0]), 
					#					self.latt.cartesian(Mirrors[0]+np.array([-1,0,0])))
					## use CSM measures to try to find the z axis
					csmZAXIS, csmYAXIS = findZaxis([nn[2] for nn in nearestNeighbors])
					perpvec = csmZAXIS - YAXIS*np.dot(YAXIS,csmZAXIS)/\
								(np.linalg.norm(YAXIS)*np.linalg.norm(csmZAXIS))

					if np.sum(perpvec) == 0:
						perpvec = np.cross(self.latt.cartesian(Mirrors[0]), 
							self.latt.cartesian(Mirrors[0]+np.array([0,-1,0])))
					ZAXIS = self.latt.ABC(perpvec)
					print('   No mirror plane found orthogonal to a rotation axis.\n',
						'     Found mirror plane at', YAXIS, '\n'
						'     Using', ZAXIS, 'as the Z axis and', YAXIS, 'as the Y axis.')
				else:
					perpvec = np.cross(self.latt.cartesian(ZAXIS), self.latt.cartesian(ZAXIS+np.array([-1,0,0])))
					if np.sum(perpvec) == 0:
						perpvec = np.cross(self.latt.cartesian(ZAXIS), self.latt.cartesian(ZAXIS+np.array([0,-1,0])))
					YAXIS = self.latt.ABC(perpvec)
					print('    No mirror planes; using', ZAXIS,' as Z axis.')

			except (IndexError, UnboundLocalError): # No mirrors and no rotations
				print('    No mirror planes and no rotations in point group:', PGS)
				NoMirrorNoRotation = True
				YAXIS = np.array([0,1.,0])
				ZAXIS = np.array([0,0,1.])
				inversion= False

	elif Yaxis == None:  # User specified Z axis, but not Y axis
		ZAXIS = np.array(Zaxis) #np.array(Zaxis)/ np.linalg.norm(Zaxis)
		print('Given Z axis:', ZAXIS)
		roundedCartesianZ = np.around(self.latt.cartesian(ZAXIS),4)
		for i, M in enumerate(Mirrors):
			if np.dot(self.latt.cartesian(M),roundedCartesianZ) == 0:
				YAXIS = M
				print('   Found mirror plane:',YAXIS)
				break

		try: YAXIS
		except UnboundLocalError:
			print('   \033[43m WARNING: No mirror plane found orthogonal to the given Z axis axis.\n',
					'     User should specify the Y axis, but PyCrystalField will make a guess... \033[0m')
			perpvec = np.cross(self.latt.cartesian(ZAXIS), 
									self.latt.cartesian(ZAXIS+np.array([-1,0,0])))
			if np.sum(perpvec) == 0:
					perpvec = np.cross(self.latt.cartesian(ZAXIS), 
						self.latt.cartesian(ZAXIS+np.array([0,0,-1])))
			YAXIS = self.latt.ABC(perpvec)
			inversion= False

	else:
		print('    User-specifyied axes...')
		cartYax = self.latt.cartesian(np.array(Yaxis))
		cartZax = self.latt.cartesian(np.array(Zaxis))
		ZAXIS = np.array(Zaxis)/ np.linalg.norm(Zaxis)
		YAXIS = self.latt.ABC(cartYax/np.linalg.norm(cartYax) -\
					 cartZax*np.dot(cartYax, cartZax))

		mirrorAlongY = False
		for MM in Mirrors:
			if np.all(np.cross(YAXIS,MM) == 0): 
				print("    There's a mirror plane orthogonal to the specified Y axis. Suppressing -m terms.")
				mirrorAlongY = True
		if not mirrorAlongY: 
			# print("    No mirror plane orthogonal to the specified Y axis.")
			inversion=False





	###############

	## Step 2b: if there is no mirrors and no rotations about the central ion,
		# We use a moment of intertia calculation to identify the z axis
	try:
		if NoMirrorNoRotation:
			#print('\t using a moment of intertia calculation to set axes.')
			ZAXIS, YAXIS = findZaxis([nn[2] for nn in nearestNeighbors])
			XAXIS = np.cross(YAXIS, ZAXIS)
			XAXIS = self.latt.ABC(XAXIS)
			YAXIS = self.latt.ABC(YAXIS)
			ZAXIS = self.latt.ABC(ZAXIS)
	except UnboundLocalError:
		## Now, find the x axis as the cross product of the Yaxis and Z axis
		XAXIS = self.latt.ABC(np.cross(self.latt.cartesian(YAXIS), self.latt.cartesian(ZAXIS)))

	XAXIS = XAXIS/np.linalg.norm(XAXIS)
	YAXIS = YAXIS/np.linalg.norm(YAXIS)
	ZAXIS = ZAXIS/np.linalg.norm(ZAXIS)

	cartXAXIS = self.latt.cartesian(XAXIS)
	cartXAXIS /= np.linalg.norm(cartXAXIS)
	cartYAXIS = self.latt.cartesian(YAXIS)
	cartYAXIS /= np.linalg.norm(cartYAXIS)
	cartZAXIS = self.latt.cartesian(ZAXIS)
	cartZAXIS /= np.linalg.norm(cartZAXIS)
	# return XAXIS, YAXIS, ZAXIS



	## Step 3: rotate local axes so z axis is along the axis identified above



	ligandNames = []
	ligandPositions = []
	ligandCharge = []
	for nn in nearestNeighbors:
		ligandNames.append(nn[0])
		ligandPositions.append([np.dot(nn[2], cartXAXIS), 
								np.dot(nn[2], cartYAXIS), 
								np.dot(nn[2], cartZAXIS)])
		try:
			ligandCharge.append(int(nn[0][-2:][::-1]))
			NoCharges = False
		except ValueError:
			NoCharges = True
			if 'O' in nn[0]:
				ligandCharge.append(-2)
			elif 'S' in nn[0]:
				ligandCharge.append(-1)
			elif 'H' in nn[0]:
				ligandCharge.append(-1)
			elif 'N' in nn[0]:
				ligandCharge.append(-4)
			else: 
				ligandCharge.append(-2)
	
	fraccarbon = np.sum(['C' in at[0] for at in nearestNeighbors])/len(nearestNeighbors)

	if np.all(['C' in at[0] for at in nearestNeighbors]):
		print('Carbon in all:', nearestNeighbors)
		ligandCharge = np.array(ligandCharge)/3
	elif fraccarbon > 0.5:
		print('Carbon in most:', fraccarbon)
		ligandCharge = np.array(ligandCharge)/3
	elif np.all(['N' in at[0] for at in nearestNeighbors]):
		ligandCharge = np.array(ligandCharge)*2
	if NoCharges: 
		centralIon = centralIon.strip('3+').strip('+3') + '3+'
		print('    No charges found in cif file... guessing the '+\
			NNLigandList[0]+' ligands are charged',ligandCharge[0],','+\
						'\n       and assuming the central ion has a 3+ charge:', centralIon)


	## Print the X, Y and Z axes for the user
	print("\n\033[44m",    #95m
		" Axes for point charge model (in ABC space):\n",
		"       X axis =", XAXIS/np.max(np.abs(XAXIS)), '\n',
		"       Y axis =", YAXIS/np.max(np.abs(YAXIS)), '\n',
		"       Z axis =", ZAXIS/np.max(np.abs(ZAXIS)),
		"\033[0m\n" )


	if crystalImage:
		plotPCF(onesite, nearestNeighbors, XAXIS, YAXIS, ZAXIS)

	if not inversion:
		print('     \033[43m WARNING: there is no mirror symmetry along the Y axis, so \n'+
			'\033[0m     \033[43m   we must inlcude the -m terms, and the eigenkets will be complex.\033[0m\n')

	return centralIon, ligandPositions, ligandCharge, inversion, ligandNames





def findRotationAxis(self, matrix):
	'''For a given transformation matrix, find the rotation angle and axis 
	if it's a rotation maxtrix, and the mirror plane if it's a mirror matrix.'''
	if np.all(matrix == np.identity(3)):
		return ['identity']
	elif np.all(matrix == -np.identity(3)):
		return ['inversion']
	determinant = np.linalg.det(matrix)
	if determinant == 1:   # otherwise it's a reflection
		## The rotation angle can be found by the following formula: Trace(m)=1+2 cos(theta)
		rotangle = np.arccos((np.sum(np.diag(matrix))-1)/2)/(2*np.pi)
		

		if rotangle == 0:
			rotangle += 1
			rotaxis = np.zeros(3)
		elif rotangle == 0.5:
			x = np.sqrt((matrix[0,0]+1)/2)
			y = np.sqrt((matrix[1,1]+1)/2)
			z = np.sqrt((matrix[2,2]+1)/2)
			rotaxis= np.array([x,y,z])
		else:
			# find rotation axis:
			x = (matrix[2,1] - matrix[1,2])
			y = (matrix[0,2] - matrix[2,0])
			z = (matrix[1,0] - matrix[0,1])
			rotaxis= np.array([x,y,z])
		# print('\trotation angle =', rotangle ) 
		# print(rotaxis)
		return ['rot', 1/rotangle, rotaxis]

	else: #find mirror planes
		eva, evec = np.linalg.eig(matrix)
		#print(matrix, eva)
		if np.sum(np.imag(eva)**2) == 0:  # only real eigenvalues, no rotation
			indicesm1 = np.where(eva==-1)[0]
			if len(indicesm1) == 1:
				mirroraxis = evec.T[indicesm1]
				return ['mirr', mirroraxis]
			else:
				return ['mirr multiple', evec.T[indicesm1]]
		else: return ['mirr+rot']



def makeSymOpMatrix(self, symop):
	# Step 1: make a matrix:
	matrix = np.zeros((3,3))
	sym = symop.split(",")
	for i,s in enumerate(sym):
		multfact = 1
		for c in s:
			if c == '-':
				multfact = -1
			if c == 'x':
				matrix[i] += multfact*np.array([1,0,0])#multfact*self.latt.a/np.linalg.norm(self.latt.a)
			elif c == 'y':
				matrix[i] += multfact*np.array([0,1,0])#multfact*self.latt.b/np.linalg.norm(self.latt.b)
			elif c == 'z':
				matrix[i] += multfact*np.array([0,0,1])#multfact*self.latt.c/np.linalg.norm(self.latt.c)
	return matrix

