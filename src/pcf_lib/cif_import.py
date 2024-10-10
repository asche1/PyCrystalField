# Code to import a .cif file into a python object

import numpy as np
from pcf_lib.LatticeClass import lattice
import time
from copy import deepcopy

class CifFile:
	"""for importing .cif files into python"""
	def __init__(self,infile):

		f = open(infile)
		lines = f.readlines()
		f.close()
		i=0
		sites = []
		symops = []
		dataglobal = 0
		while i < len(lines):
			line = lines[i]

			# Check if we're in first phase:
			if 'data_global' in line:
				print(line, i, dataglobal)
				dataglobal += 1
				if dataglobal > 1: break #new phase!
			if '=END' in line: break

			#Find the unit cell parameters
			if line.startswith('_cell_length_a'):
				a = self._destringify( line.split()[1])
			elif line.startswith('_cell_length_b'):
				b = self._destringify( line.split()[1])
			elif line.startswith('_cell_length_c'):
				c = self._destringify( line.split()[1])
			elif line.startswith('_cell_angle_alpha'):
				aa = self._destringify( line.split()[1])
			elif line.startswith('_cell_angle_beta'):
				bb = self._destringify( line.split()[1])
			elif line.startswith('_cell_angle_gamma'):
				cc = self._destringify( line.split()[1])
				print('unit cell:', a,b,c,aa,bb,cc)

			# Find the atoms within the unit cell
			#elif (line.startswith("loop_") and lines[i+1].startswith(" _atom_site_label")):
			elif (line.startswith("loop_") and ((lines[i+1].strip().startswith("_atom_site_label")) or 
				lines[i+1].strip().startswith("_atom_site_type"))):
				print("Importing atoms")
				i+=1
				line = lines[i]
				
				jj = 0   # index for keeping track of labels
				while (line != " \r\n" and line != "\r\n" and line !='\n' and line.strip() != '' 
						and line !=' \n' and line !='loop_\n' and not line.startswith('#')): #loop until we hit a blank spot
					if '_atom' in line:
						sitesymorder = None
						if 'atom_site_fract_x' in line:
							fract_x = jj
						elif 'atom_site_fract_y' in line:
							fract_y = jj
						elif 'atom_site_fract_z' in line:
							fract_z = jj
						elif 'atom_site_occupancy' in line:
							occ = jj
						elif ('atom_site_site_symmetry_order' in line) or ('atom_site_symmetry_multipliticy' in line):
							sitesymorder = jj
						elif 'site_type_symbol' in line:
							sitetypesymbol = jj
						elif 'site_label' in line:
							sitelabel = jj
						i+=1
						jj +=1
					else:
						if line.startswith('_'): break
						site = line.split()#[0:9]
						#print('site', site)
						modsite = deepcopy(site)
						if len(modsite) < 3:  # If the line for some reason spills over into the next...
							i += 1
							try:
								line = lines[i]
							except IndexError: break
							continue
						modsite[0] = site[sitelabel]
						modsite[1] = site[sitetypesymbol]
						modsite[2] = self._destringify(site[fract_x])
						modsite[3] = self._destringify(site[fract_y])
						modsite[4] = self._destringify(site[fract_z])
						try:
							modsite[7] = self._destringify(site[occ])
						except IndexError:
							modsite.extend([self._destringify(site[occ])] * (len(modsite)-5))
						if sitesymorder != None:
							modsite[8] = int(site[sitesymorder])
						modsite.append(line.split()[-1])
						# Move to middle of unit cell
						for jj in range(2,5):
							modsite[jj] -= int(modsite[jj])
							if modsite[jj] < 0:
								modsite[jj] += 1
						sites.append(modsite)
						i+=1
					try:
						line = lines[i]
					except IndexError: break
				i -=1

			# Find the symmetry operations
			elif (line.startswith("loop_") and 
					(("_space_group_symop_id" in lines[i+1]
							or "_space_group_symop_operation_xyz" in lines[i+1]
							or "_symmetry_equiv_pos_site_id" in lines[i+1])
							or "_symmetry_equiv_pos_as_xyz" in lines[i+1])):
				while ('_' in line):  #jump ahead to where the symops are
					i+=1
					line = lines[i]
				while (line != " \r\n" and line != "\r\n" and line != "\n" and line != " \n" 
						and line != 'loop_\n' and line.strip() != ''): #loop until we hit a blank spot
					if line.startswith('#'):
						break
					if '\'' in line:
						quoteloc = [ii for ii, ltr in enumerate(line) if ltr == '\'']
						symops.append(line[quoteloc[0]+1:quoteloc[1]])
					else:
						lnsplit = line.split(' ')
						for ln in lnsplit:
							if ',' in ln:  # Identify the symmetry operation by the presence of commas
								symops.append(ln.strip())
								break
					i+=1
					line = lines[i]
				i -= 1
			i+=1
			
		# print(symops)

		if not sites:
			# sites list is empty
			# Without any atoms we can't do anything so this is a fatal error.
			raise RuntimeError("No atomic sites were found when importing cif file")

		if not symops:
			# symops list is empy
			# It may be that we don't have any symops (a supercell perhaps) so this is
			# a warning rather than an error.
			raise RuntimeWarning("No symmetry operations were found in the cif file")

		self.asymunitcell = list(sites)
		# Operate on asymmetric unit cell with all symops and then eliminate duplicates
		self.MakeUnitCell(sites,symops)
		self.symops = symops

		# Define unit cell using lattice class
		self.latt = lattice(a,b,c,aa,bb,cc)
		#print(a,b,c,aa,bb,cc)

		print(".cif import complete.")




	def SymOperate(self,symstring,atom):
		"""operate on atom positions with symmetry operation"""
		newatom = list(atom)
		newpos = [0.0, 0.0, 0.0]
		xpos, ypos, zpos = atom[2], atom[3], atom[4]
		symstring = symstring.replace(' ','')
		symstring = symstring.replace("x",str(xpos)).replace('y',str(ypos)).replace('z',str(zpos)).replace(
								"X",str(xpos)).replace('Y',str(ypos)).replace('Z',str(zpos))
		sym = symstring.split(",")
		newpos[0] = self._defractionify(sym[0])
		newpos[1] = self._defractionify(sym[1])
		newpos[2] = self._defractionify(sym[2])
		"""Translate the site back into the original unit cell"""
		for i in range(len(newpos)):
			if newpos[i] < -0.001:
				newpos[i] +=1
			if newpos[i] >= 1.00:
				newpos[i] -=1

		newatom[2:5] = newpos
		return newatom



	def MakeUnitCell(self,sites,symops):
		unitcell = list(sites)
		i = 0
		for sy in symops:
			for at in sites:
				new_at = self.SymOperate(sy,at)
				# Move inside unit cell
				for jj in range(2,5):
					new_at[jj] -= int(new_at[jj])
					if new_at[jj] < 0:
						new_at[jj] += 1

				# test if new atom is in array already
				if self._duplicaterow(new_at, unitcell):
					# Test if new atom is outside the unit cell
					unitcell.append(new_at)
					i+=1
		
		#print('  ', i, "atoms added")
		## Eliminate all atoms outside the unit cell
		self.unitcell = []
		for at in unitcell:
			if np.all(np.array(at[2:5])>=0):
				if np.all(np.array(at[2:5])<=1):
					self.unitcell.append(at)
		print('  ', len(self.unitcell), "atoms added")

		#print sum(sites[i][8] for i in xrange(len(sites))), "symmetry-equivalent sites"

		# Now, create a dictionary with symmetry equivalent atoms and sites
		atomnames = [auc[-1] for auc in self.asymunitcell]
		#atomnames = [auc[-1] for auc in self.asymunitcell]
		self.atomsunitcell = {}
		for ii, an in enumerate(atomnames):
			self.atomsunitcell[an] = []

		for uc in self.unitcell:
			self.atomsunitcell[uc[-1]].append(uc[2:5])
		for ii, an in enumerate(atomnames):
			self.atomsunitcell[an] = np.array(self.atomsunitcell[an])


	def StructureFactor(self,scatlength,maxrlv):
		"""Compute the nuclear structure factor.
		Note that the output is dependent upon the input form of the scattering lengths.
		If the lengths are pulled straight from the NCNR website, the structure factor
		will be in units of (fm)^2/sr, not barns/sr """
		# Make sure the input dictionary for scattering lengths is of the correct form
		elmts = self._NumElements(self.asymunitcell)
		try:
			for el in elmts:
				scatlength[el]
		except LookupError as err:
			print(err, "missing from 'scatlength' dictionary. Should contain "+ str(elmts)+"." +\
				'To look up values, see  https://www.ncnr.nist.gov/resources/n-lengths/ ,  coh b ')
			return 0

		# Build numpy array of a,b,c, scattering length, and occupancy
		atomlist = np.zeros((len(self.unitcell),5))
		i=0
		for a in self.unitcell:
			atomlist[i] = np.array([a[2],a[3],a[4],scatlength[a[1]], a[7]])
			i+=1

		# Create array of reciprocal lattice vectors
		taulim = np.arange(-maxrlv+1,maxrlv)
		xx, yy, zz = np.meshgrid(taulim,taulim,taulim)
		x = xx.flatten()
		y = yy.flatten()
		z = zz.flatten()
		#array of reciprocal lattice vectors; 4th column will be structure factor^2
		tau = np.array([x,y,z, np.zeros(len(x))]).transpose()  

		# Compute structure factor, weighted by occupancy (have not added Debye Waller factor yet...)
		i=0
		for t in tau:
			tau[i,3] = np.abs(np.sum(
				atomlist[:,3]*atomlist[:,4]*
				np.exp(2j*np.pi*np.inner(t[0:3],atomlist[:,0:3]))))**2
			i+=1
		# Eliminate tiny values
		tau[:,3][tau[:,3] < 1e-16] = 0.0
		self.SF = tau


	def MultipleScattering(self, ei, peak, xcut,ycut, threshold=0.05,):
		MS_SF = np.copy(self.SF)

		# Define the tau vectors found in the scattering plane
		normplane = np.cross(xcut,ycut)
		scatplanetau = self.SF[np.inner(normplane,self.SF[:,0:3])==0]

		# Find the incident wavevector
		k = self._kvector(ei)
		xcutnorm = xcut.astype(float)/np.linalg.norm(xcut)
		ycutnorm = ycut.astype(float)/np.linalg.norm(ycut)

		# Compute multiple scattering
		qvect = np.linalg.norm(self.latt.inverseA(vect = np.array(peak)))
		thet = np.arcsin(qvect/2/k)

		# put K_i in scattering plane defined by xcut and ycut in 
		k_h = k*(np.dot(np.array([1,0,0]), xcutnorm)*np.sin(thet) + np.dot(np.array([1,0,0]) , ycutnorm)*np.cos(thet))
		k_k = k*(np.dot(np.array([0,1,0]), xcutnorm)*np.sin(thet) + np.dot(np.array([0,1,0]) , ycutnorm)*np.cos(thet))
		k_l = k*(np.dot(np.array([0,0,1]), xcutnorm)*np.sin(thet) + np.dot(np.array([0,0,1]) , ycutnorm)*np.cos(thet))

		print(" Multiple scattering for", peak," peak, Ei =", ei,'meV:')
		print("---------------------------------------------")
		print("    dq \t \t intermediate peaks \t \t  SF^2")
		for t1 in self.SF:
			# Find length of scattering vector in inverse angstroms
			t1A = self.latt.inverseA(vect = t1[0:3])
			#calculate k-distance to tip of \vec{ki}
			dq  = np.sqrt((t1A[0]-k_h)**2+(t1A[1]-k_k)**2+(t1A[2]-k_l)**2)-k 

			if np.abs(dq) < threshold : 
				for t2 in self.SF:
					if np.array_equal((t2[0:3]+t1[0:3]), peak):
						t2sf = t2[3]
						t2pk = t2[0:3]
						break
				#netSF = t2sf * t1[3]
				try: netSF = t2sf * t1[3]
				except UnboundLocalError:
					netSF = 0
				
				if ((netSF != 0.0) 
					and not np.array_equal(t1[0:3], np.array([0,0,0])) 
					and not np.array_equal(t2pk, np.array([0,0,0]))
					and 1 > np.linalg.norm(t1[0:3])*2*np.pi/10./(2*k)
					and 1 > np.linalg.norm(t2pk)*2*np.pi/10./(2*k)):

					# report multiple multiple scattering match
					print("   {0:.4f}\t".format(dq) ,t1[0:3], ' ', t2pk, '\t', netSF)

		# Compute multiple scattering
		# for t1 in scatplanetau:
		# 	qvect = np.linalg.norm(self.latt.inverseA(vect = t1[0:3]))
		# 	thet = np.arcsin(qvect/2/k)

		# 	# put K_i in scattering plane defined by (00l) and (hh0) in 
		# 	k_h = k*(np.dot(np.array([1,0,0]), xcutnorm)*np.sin(thet) + np.dot(np.array([1,0,0]) , ycutnorm)*np.cos(thet))
		# 	k_k = k*(np.dot(np.array([0,1,0]), xcutnorm)*np.sin(thet) + np.dot(np.array([0,1,0]) , ycutnorm)*np.cos(thet))
		# 	k_l = k*(np.dot(np.array([0,0,1]), xcutnorm)*np.sin(thet) + np.dot(np.array([0,0,1]) , ycutnorm)*np.cos(thet))

		# 	for t2 in self.SF:
		# 		# Find length of scattering vector in inverse angstroms
		# 		t2A = self.latt.inverseA(vect = t2[0:3])

		# 		#calculate k-distance to tip of \vec{ki}
		# 		dq  = np.sqrt((t2A[0]-k_h)**2+(t2A[1]-k_k)**2+(t2A[2]-k_l)**2)-k 

		# 		if np.abs(dq) < 0.05 : 
		# 			# report multiple multiple scattering match
		# 			print t1[0:3],"{0:.4f}\t {1}  {2}  {3}".format(t2[0],t2[1],t2[2],dq) 





	def _destringify(self,string):
		"""automatically remove parenthesis from end of strings"""
		if isinstance(string, str):
			if '(' in string:
				numb = float(string[:string.find("(")])
			else: numb = float(string)

			## Replace 0.3333 with 1/3, etc.
			threshold = 5e-5
			if np.abs(numb - 1/3) < threshold:
				numb = 1/3
			elif np.abs(numb - 2/3) < threshold:
				numb = 2/3			
			elif np.abs(numb - 1/6) < threshold:
				numb = 1/6
			return numb
		else:   return string

	def _defractionify(self,string):
		"""convert string fraction to float"""
		return eval(string)	
		# if ('+' in string) and ('-' in string[1:]):
		# 	answer = 0
		# 	for st in string.split('+'):
		# 		for st2 in st.split('-'):
		# 			frac = st2.split('/')
		# 			if len(frac)==1:
		# 				answer -= float()
		# 			else:
		# 	return answer
		# elif '+' in string:
		# 	st = string.split('+')
		# 	frac = st[1].split('/')
		# 	if len(frac)==1:
		# 		return float(st[0]) + float(frac[0])
		# 	else:
		# 		return float(st[0]) + float(frac[0])/float(frac[1])
		# elif '-' in string[1:]:
		# 	st = string[1:].split('-')
		# 	st[0] = string[0]+st[0]
		# 	frac = st[1].split('/')
		# 	if len(frac)==1:
		# 		return float(st[0]) - float(frac[0])
		# 	else:
		# 		return float(st[0]) - float(frac[0])/float(frac[1])
		# else:
		# 	return float(string)

	def _duplicaterow(self, row, array):
		"""Determine whether a row exists in an array. Returns False if it does"""
		value = False
		for r in array:
			if (#r[0] == row[0] and
				r[1] == row[1] and
				round(r[2] - row[2],3) == 0 and
				round(r[3] - row[3],3) == 0 and
				round(r[4] - row[4],3) == 0  ):

				value = True
		return not value

	def _NumElements(self,sites):
		"""returns an array of unique atoms in the unit cell"""
		r = []
		for s in sites:
			r.append(s[1])
		return list(set(r))

	def _kvector(self,energy):
		"""returns the wave vector of neutrons of a given energy"""
		return 0.694693 * np.sqrt(energy)




###############################################################################
#Test with Yb2Ti2O7
#YbTiO = CifFile("StoichiometricYbTiO.cif")
#s_length = {'O2-': 5.803, 'Ti4+': -3.438, 'Yb3+': 12.43}
#YbTiO.StructureFactor(s_length,5)
#print(YbTiO.SF)
#print(' ')
#YbTiO.MultipleScattering(ei=1.0, threshold=0.1, peak = [0,0,2], xcut=np.array([1,1,1]), ycut = np.array([1,1,-2]))
#print(' ')
#YbTiO.MultipleScattering(ei=5, threshold=0.1, peak = [-4,2,2], xcut=np.array([1,-1,0]), ycut = np.array([1,1,-2]))
