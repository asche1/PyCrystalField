import numpy as np
import matplotlib.pyplot as plt


def plotPCF(onesite, nearestNeighbors, Xax, Yax, Zax):
	atomlist = [[[0,0,0]], [nn[2] for nn in nearestNeighbors]]

	plt.figure(figsize=(6,6))
	obj = atomplot(1.5, 0.2, atomlist)
	obj.plotatoms(plotlines = [1])
	obj.plotaxes(Xax, Yax, Zax)
	plt.savefig(onesite[0]+'_ligands.png')
	plt.close()



class atomplot:
	def __init__(self, theta, phi, atoms):
		self.theta = theta
		self.phi = phi
		self.plotX = np.array([np.cos(phi), np.sin(phi), 0])
		self.plotY = np.array([-np.sin(phi)*np.cos(theta), np.cos(phi)*np.cos(theta), np.sin(theta)])
		self.plotZ = np.cross(self.plotX, self.plotY)

		self.atoms = atoms
		# print(self.plotX, self.plotY, self.plotZ, np.dot(self.plotX, self.plotY), np.dot(self.plotY, self.plotZ))

	def plotatoms(self, plotlines  = []):
		plt.axis('off')
		colors = plt.cm.Set1(np.arange(8))
		for i,at in enumerate(self.atoms):
			for aa in at:
				plt.plot(np.dot(aa, self.plotX), np.dot(aa, self.plotY), 
					marker='o', markersize=100/(i+2), mec='k', color=colors[i], 
					zorder= np.dot(aa, self.plotZ))

			if i in plotlines:
				for a1 in at:
					dist = np.array(a1)-np.array(self.atoms[0][0])
					for a2 in at:
						vect = np.array(a1)-np.array(a2)
						# print(np.abs(np.dot(vect,  dist)))
						if np.abs(np.dot(vect,  dist)) < np.dot(dist,dist)*1.2:
							plt.plot([np.dot(a1, self.plotX), np.dot(a2, self.plotX)], 
								[ np.dot(a1, self.plotY), np.dot(a2, self.plotY)], 
								color='grey', lw='3', zorder = np.mean([np.dot(a1, self.plotZ), np.dot(a2, self.plotZ)]))

		xvals = [np.dot(a, self.plotX) for a in at for at in self.atoms]
		yvals = [np.dot(a, self.plotY) for a in at for at in self.atoms]
		plt.xlim(np.min(xvals+yvals)*1.25, np.max(xvals+yvals)*1.25)
		plt.ylim(np.min(xvals+yvals)*1.25, np.max(xvals+yvals)*1.25)

	def plotaxes(self, X, Y, Z):
		# X = np.array([1,0,0])
		# Y = np.array([0,1,0])
		# Z = np.array([0,0,1])
		arrowatributes = {'head_width':0.06, 'overhang':0.1, 'color':'k'}

		plt.arrow(0,0, *self._flatten(X/2), **arrowatributes)
		plt.arrow(0,0, *self._flatten(Y/2), **arrowatributes)
		plt.arrow(0,0, *self._flatten(Z/2), **arrowatributes)

		disp = np.array([0.04,0.04])
		plt.text(*self._flatten(X/2)+disp, 'X')
		plt.text(*self._flatten(Y/2)+disp, 'Y')
		plt.text(*self._flatten(Z/2)+disp, 'Z')


	def plotabc(self):
		X = np.array([1,0,0])
		Y = np.array([0,1,0])
		Z = np.array([0,0,1])
		arrowatributes = {'head_width':0.08, 'overhang':0.1}

		plt.arrow(0,0, *self._flatten(X/2), color='r', **arrowatributes)
		plt.arrow(0,0, *self._flatten(Y/2), color='g', **arrowatributes)
		plt.arrow(0,0, *self._flatten(Z/2), color='b', **arrowatributes)

		disp = np.array([0.04,0.04])
		plt.text(*self._flatten(X/2)+disp, 'X')
		plt.text(*self._flatten(Y/2)+disp, 'Y')
		plt.text(*self._flatten(Z/2)+disp, 'Z')


	def _flatten(self, vect):
		return np.dot(vect, self.plotX), np.dot(vect, self.plotY)




def exportLigandCif(self, filename):
	'''Takes a PyCrystalField Ligands object and exports a cif file with only the 
	central ion and the nearest neighbor ligands.'''
	if not filename.endswith('.cif'):
		filename = filename + '.cif' 

	with open(filename, 'w') as f:
		f.write('# Output from PyCrystalField showing the ligand environment\n\n')
		f.write('loop_\n'+\
				'_publ_author_name\n'+\
				"'Someone, A.'\n"+\
				"'Someone, B.'\n"+
				'_cell_length_a 10.0\n'+\
				'_cell_length_b 10.0\n'+\
				'_cell_length_c 10.0\n'+\
				'_cell_angle_alpha 90.\n'+\
				'_cell_angle_beta 90.\n'+\
				'_cell_angle_gamma 90.\n'+\
				'_cell_volume 1000.0\n'+\
				"_symmetry_space_group_name_H-M 'P 1'\n"+\
				'loop_\n'+\
				'_symmetry_equiv_pos_site_id\n'+\
				'_symmetry_equiv_pos_as_xyz\n'+\
				"1 'x, y, z'\n"+\
				'loop_\n'+\
				'_atom_type_symbol\n'+\
				self.ion + '\n'+\
				'S2-\n'+\
				'loop_\n'+\
				'_atom_site_label\n'+\
				'_atom_site_fract_x\n'+\
				'_atom_site_fract_y\n'+\
				'_atom_site_fract_z\n'+\
				'_atom_site_occupancy\n'+\
				self.ion +' 0.5 0.5 0.5 1. \n')
		for b in self.bonds:
			f.write('S1 '+ ' '.join([str(bi/10+0.5) for bi in b])+ ' 1. \n')