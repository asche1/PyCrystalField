import numpy as np

class mslice:
	def __init__(self, infile):
		# with open(infile) as f:
		# 	lines = f.readlines()
		# 	if lines[0].split(' ')[2] == 'Slice':
		# 		self.Slice = True
		# 	else:
		# 		self.Slice = False

		# if self.Slice:
		# 	self.x = data[0]
		# 	self.y = data[1]
		# 	self.i = data[2]
		# 	self.e = data[3]
		# else:
		# 	self.x = data[0]
		# 	self.i = data[1]
		# 	self.e = data[2]

		sdata = np.genfromtxt(infile+'_slice.xyie', unpack=True)
		self.sx = np.unique(sdata[0])
		self.sy = np.unique(sdata[1])
		self.si = sdata[2].reshape(len(self.sx), len(self.sy)).T
		self.se = sdata[3].reshape(len(self.sx), len(self.sy)).T

		cdata = np.genfromtxt(infile+'_cut.xie', unpack=True)
		self.cx = cdata[0]
		self.ci = cdata[1]
		self.ce = cdata[2]

		## Eliminate nans from cuts 
		self.cx = self.cx[~np.isnan(self.ci)]
		self.ci = self.ci[~np.isnan(self.ci)]
		self.ce = self.ce[~np.isnan(self.ce)]

	def normalize(self, scalefactor):
		self.si *= scalefactor
		self.ci *= scalefactor
		self.se *= scalefactor
		self.ce *= scalefactor

mslice('/home/1o1/Documents/KErSe2/CsErSe2/SavedData/CES_2K_9meV_0Tb')
