import numpy as np

class lattice:
	def __init__(self,a,b,c,alpha,beta,gamma):
		"""Define the unit cell"""
		self.anorm = a
		self.bnorm = b
		self.cnorm = c
		alpha = alpha*np.pi/180
		beta = beta*np.pi/180
		gamma = gamma*np.pi/180
		self.a = a*np.array([1,0,0])
		self.b = b*np.array([np.cos(gamma), np.sin(gamma), 0])
		self.c = c*np.array([np.cos(beta), 
			np.sin(beta)*np.cos(alpha)*np.sin(gamma), 
			np.sin(alpha)*np.sin(beta) ])
		#Round numbers (eliminate tiny numbers)
		self.b=np.around(self.b,decimals=8)
		self.c=np.around(self.c,decimals=8)

		#print(self.a, self.b, self.c)
		#print(np.arccos(np.dot(self.b, self.c)/self.bnorm/self.cnorm)*180/np.pi)
		#print(np.arccos(np.dot(self.a, self.c)/self.anorm/self.cnorm)*180/np.pi)
		#print(np.arccos(np.dot(self.a, self.b)/self.anorm/self.bnorm)*180/np.pi)

		#Define the reciprocal lattice
		self.reciplatt()


	def reciplatt(self):
		"""output the reciprocal lattice in np vectors"""
		cellvol = np.dot(self.a,np.cross(self.b,self.c))

		self.astar = 2*np.pi * np.cross(self.b,self.c) / cellvol
		self.bstar = 2*np.pi * np.cross(self.c,self.a) / cellvol
		self.cstar = 2*np.pi * np.cross(self.a,self.b) / cellvol

		#Round numbers (eliminate tiny numbers)
		self.bstar=np.around(self.bstar,decimals=8)
		self.cstar=np.around(self.cstar,decimals=8)

		self.V = cellvol
		self.Vstar = np.dot(self.astar,np.cross(self.bstar,self.cstar))

	def cartesian(self,vect,norm=True):
		"""Convert a vector from ABC space to Cartesian Space"""
		if vect.size == 3:
			if norm == True:
				return vect[0]*self.a + vect[1]*self.b + vect[2]*self.c
			else:
				return vect[0]*self.a/np.linalg.norm(self.a) \
					+ vect[1]*self.b/np.linalg.norm(self.b) \
					+ vect[2]*self.c/np.linalg.norm(self.c)
		elif len(vect[0]) == 3:
			return np.outer(vect[:,0],self.a) + np.outer(vect[:,1],self.b) + np.outer(vect[:,2],self.c)
		else:
			raise ValueError("vector must have three components") 


	def ABC(self, vect):
		"""Convert a vector from Cartesian Space to ABC space"""
		matrix = np.array([self.a, self.b, self.c])

		if vect.size == 3:
			return np.dot(vect,np.linalg.inv(matrix))
		else:
			raise ValueError("vector must have three components") 


	def inverseA(self,vect,norm=True):
		"""Convert a vector from RLU space to inverse Aangstroms abs value"""
		if vect.size == 3:
			return vect[0]*self.astar + vect[1]*self.bstar + vect[2]*self.cstar
		elif len(vect[0])==3:
			return np.outer(vect[:,0],self.astar) + np.outer(vect[:,1],self.bstar) + np.outer(vect[:,2],self.cstar)
		else:
			raise ValueError("vector must have three components") 
