import numpy as np
from pcf_lib.cif_import import CifFile

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


def FindPointGroupSymOps(self, ion):
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

	#print(PGS)
	# Step 3: make the symmetry operations matrices and find the rotation axes
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
		YAXIS = Mirrors[0]
		perpvec = np.cross(self.latt.cartesian(Mirrors[0]), 
							self.latt.cartesian(Mirrors[0]+np.array([-1,0,0])))
		if np.sum(perpvec) == 0:
			perpvec = np.cross(self.latt.cartesian(Mirrors[0]), 
				self.latt.cartesian(Mirrors[0]+np.array([0,-1,0])))
		ZAXIS = self.latt.ABC(perpvec)
		print('   No mirror plane found orthogonal to rotation axis.\n',
			'     Using', ZAXIS, 'as the Z axis and', YAXIS, 'as the Y axis.')

	## Now, find the x axis as the cross product of the two
	XAXIS = self.latt.ABC(np.cross(self.latt.cartesian(YAXIS), self.latt.cartesian(ZAXIS)))
	XAXIS /= np.linalg.norm(XAXIS)
	YAXIS /= np.linalg.norm(YAXIS)
	ZAXIS /= np.linalg.norm(ZAXIS)

	cartXAXIS = self.latt.cartesian(XAXIS)
	cartXAXIS /= np.linalg.norm(cartXAXIS)
	cartYAXIS = self.latt.cartesian(YAXIS)
	cartYAXIS /= np.linalg.norm(cartYAXIS)
	cartZAXIS = self.latt.cartesian(ZAXIS)
	cartZAXIS /= np.linalg.norm(cartZAXIS)

	# return XAXIS, YAXIS, ZAXIS


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
		if ion not in at[0]:
			for ucs in unitcellshifts:
				distVec0 = self.latt.cartesian(np.array(onesite[2:5]) - (np.array(at[2:5]) + ucs))
				neighborlist.append([at[1], np.linalg.norm(distVec0), distVec0])
				distlist.append(np.linalg.norm(distVec0))

	
	# Step 2: sort the list in ascending order
	sortedNeighborArgs = np.argsort(distlist)

	nearestNeighbors = []
	nearestLigand = neighborlist[sortedNeighborArgs[0]][0]
	for sna in sortedNeighborArgs:
		if neighborlist[sna][0] == nearestLigand:
			nearestNeighbors.append(neighborlist[sna])
		else: break

	
	print('   Identified',len(nearestNeighbors),nearestNeighbors[0][0],'ligands.')

	## Step 3: rotate local axes so z axis is along the axis identified above
	ligandPositions = []
	ligandCharge = []
	for nn in nearestNeighbors:
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
			else: 
				ligandCharge.append(-2)
	if NoCharges: 
		print('    No charges found in cif file... guessing the ligands are charged',ligandCharge[0],','+\
						'\n       and assuming the central ion has a 3+ charge.')
		centralIon = centralIon + '3+'


	## Print the X, Y and Z axes for the user
	print("\n\033[44m",
		" Axes for point charge model (in ABC space):\n",
		"       X axis =", XAXIS, '\n',
		"       Y axis =", YAXIS, '\n',
		"       Z axis =", ZAXIS,
		"\033[0m\n" )


	return centralIon, ligandPositions, ligandCharge





def findRotationAxis(self, matrix):
	'''For a given transformation matrix, find the rotation angle and axis 
	if it's a rotation maxtrix, and the mirror plane if it's a mirror matrix.'''
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

