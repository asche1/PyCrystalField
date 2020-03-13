# Code for computing crystal electric fields

import numpy as np
from scipy import optimize
import scipy.linalg as LA
#import numpy.linalg as LA
#from scipy.linalg import eig
from scipy.special import wofz
import sys, os
from pcf_lib.form_factors import RE_FormFactor
import pcf_lib.LatticeClass as lat
from pcf_lib.CreateFitFunction import makeFitFunction
from numba import njit, jitclass

# abspath = os.path.abspath(__file__)
# dname = os.path.dirname(abspath)
# os.chdir(dname)



class Ket():
    def __init__(self, array): 
        """give an array which defines the angular momentum eigenket in terms 
        of the available states. IE, write |1> + |2> as [0,0,0,1,1]"""
        self.ket = np.array(array)
        self.j = len(array)/2.0 - 0.5
        self.m = np.arange(-self.j,self.j+1,1)

    def Jz(self):
        return Ket( self.ket * self.m )

    def Jplus(self):
        newvals = np.sqrt((self.j-self.m)*(self.j+self.m+1)) * self.ket
        return Ket( np.roll(newvals,1) )

    def Jminus(self):
        newvals = np.sqrt((self.j+self.m)*(self.j-self.m+1)) * self.ket
        return Ket( np.roll(newvals,-1) )

    def Jx(self):
        return Ket(0.5*(self.Jplus().ket + self.Jminus().ket) )

    def Jy(self):
        return Ket(-1j*0.5*(self.Jplus().ket - self.Jminus().ket) )

    def R(self, alpha, beta, gamma):  # Rotation about general Euler Angles
        return self._Rz(alpha)._Ry(beta)._Rz(gamma)

    def _Rz(self,theta):  # Rotation about z axis
        newvals = np.zeros(len(self.ket), dtype=complex)
        for i in range(len(self.ket)):
           newvals[i] = self.ket[i]* np.exp(-1j*self.m[i] * theta)
        return Ket(newvals)

    def _Ry(self,beta):  # Rotation about y axis
        newvals = np.zeros(len(self.ket), dtype=complex)
        for i in range(len(self.ket)):
            mm = self.m[i]
            for j in range(len(self.ket)):
                mmp = self.m[j]
                newvals[j] += self.ket[i]* self._WignersFormula(mm,mmp,beta)
        return Ket(newvals)

    def _WignersFormula(self,m,mp,beta):
        """See Sakurai/Napolitano eq. 3.9.33. 
        This function was cross-checked with Mathematica's WignerD function."""

        # determine the limit of the sum over k
        kmin = np.maximum(0, m-mp)
        kmax = np.minimum(self.j+m, self.j-mp)

        d = 0
        for k in np.arange(kmin,kmax+1):
            d += (-1)**(k-m+mp) * np.sqrt(np.math.factorial(self.j+m) * np.math.factorial(self.j-m) *\
                                np.math.factorial(self.j+mp) * np.math.factorial(self.j-mp))/\
                (np.math.factorial(self.j+m-k) * np.math.factorial(k) * np.math.factorial(self.j-k-mp)*\
                 np.math.factorial(k-m+mp))*\
                np.cos(beta/2)**(2*self.j -2*k+m-mp) * np.sin(beta/2)**(2*k-m+mp)
        return d


    def __mul__(self,other):
        if isinstance(other, Ket):
            # Compute inner product
            return np.dot(np.conjugate(self.ket), other.ket)
        else:
            return Ket( np.dot(self.ket, other))

    def __add__(self,other):
        if isinstance(other, Ket):
            return Ket(self.ket + other.ket)
        else:
            print("other is not a ket")

        # def __rmul__(self,other):   #Doesn't work. not sure why.
        # 	if isinstance(other, Ket):
        # 		return np.vdot(other.ket, self.ket)
        # 	else:  # if not ket, try matrix multiplication.
        # 		#print "reverse multiply"
        # 		return np.dot(other, self.ket)



# spec = [
#     ('O', float32[:,:]),               # a simple scalar field
#     #('j', float32),
#     ('m', float32[:]),          # an array field
# ]

# @jitclass(spec)  # Doesn't work
class Operator():
    def __init__(self, J):
        self.O = np.zeros((int(2*J+1), int(2*J+1)))
        self.m = np.arange(-J,J+1,1)
        self.j = J

    @staticmethod
    def Jz(J):
        obj = Operator(J)
        for i in range(len(obj.O)):
            for k in range(len(obj.O)):
                if i == k:
                    obj.O[i,k] = (obj.m[k])
        return obj

    @staticmethod
    def Jplus(J):
        obj = Operator(J)
        for i in range(len(obj.O)):
            for k in range(len(obj.O)):
                if k+1 == i:
                    obj.O[i,k] = np.sqrt((obj.j-obj.m[k])*(obj.j+obj.m[k]+1))
        return obj

    @staticmethod
    def Jminus(J):
        obj = Operator(J)
        for i in range(len(obj.O)):
            for k in range(len(obj.O)):
                if k-1 == i:
                    obj.O[i,k] = np.sqrt((obj.j+obj.m[k])*(obj.j-obj.m[k]+1))
        return obj

    @staticmethod
    def Jx(J):
        objp = Operator.Jplus(J)
        objm = Operator.Jminus(J)
        return 0.5*objp + 0.5*objm

    @staticmethod
    def Jy(J):
        objp = Operator.Jplus(J)
        objm = Operator.Jminus(J)
        return -0.5j*objp + 0.5j*objm

    def __add__(self,other):
        newobj = Operator(self.j)
        if isinstance(other, Operator):
           newobj.O = self.O + other.O
        else:
           newobj.O = self.O + other*np.identity(int(2*self.j+1))
        return newobj

    def __radd__(self,other):
        newobj = Operator(self.j)
        if isinstance(other, Operator):
            newobj.O = self.O + other.O
        else:
            newobj.O = self.O + other*np.identity(int(2*self.j+1))
        return newobj

    def __sub__(self,other):
        newobj = Operator(self.j)
        if isinstance(other, Operator):
            newobj.O = self.O - other.O
        else:
            newobj.O = self.O - other*np.identity(int(2*self.j+1))
        return newobj

    def __mul__(self,other):
        newobj = Operator(self.j)
        if (isinstance(other, int) or isinstance(other, float) or isinstance(other, complex)):
           newobj.O = other * self.O
        else:
           newobj.O = np.dot(self.O, other.O)
        return newobj

    def __rmul__(self,other):
        newobj = Operator(self.j)
        if (isinstance(other, int) or isinstance(other, float)  or isinstance(other, complex)):
           newobj.O = other * self.O
        else:
           newobj.O = np.dot(other.O, self.O)
        return newobj

    def __pow__(self, power):
        newobj = Operator(self.j)
        newobj.O = self.O
        for i in range(power-1):
            newobj.O = np.dot(newobj.O,self.O)
        return newobj

    def __neg__(self):
        newobj = Operator(self.j)
        newobj.O = -self.O
        return newobj

    def __repr__(self):
        return repr(self.O)








def StevensOp(J,n,m):
    """generate stevens operator for a given total angular momentum
    and a given n and m state"""
    Jz = Operator.Jz(J=J)
    Jp = Operator.Jplus(J=J)
    Jm = Operator.Jminus(J=J)
    X = J*(J+1.)

    if [n,m] == [0,0]:
        return np.zeros((int(2*J+1), int(2*J+1)))
    elif [n,m] == [1,0]:
        matrix = Jz
    elif [n,m] == [1,1]:
        matrix = 0.5 *(Jp + Jm)
    elif [n,m] == [1,-1]:
        matrix = -0.5j *(Jp - Jm)

    elif [n,m] == [2,2]:
        matrix = 0.5 *(Jp**2 + Jm**2)
    elif [n,m] == [2,1]:
        matrix = 0.25*(Jz*(Jp + Jm) + (Jp + Jm)*Jz)
    elif [n,m] == [2,0]:
        matrix = 3*Jz**2 - X
    elif [n,m] == [2,-1]:
        matrix = -0.25j*(Jz*(Jp - Jm) + (Jp - Jm)*Jz)
    elif [n,m] == [2,-2]:
        matrix = -0.5j *(Jp**2 - Jm**2)

    elif [n,m] == [3,3]:
        matrix = 0.5 *(Jp**3 + Jm**3)
    elif [n,m] == [3,2]:
        matrix = 0.25 *((Jp**2 + Jm**2)*Jz + Jz*(Jp**2 + Jm**2))
    elif [n,m] == [3,1]:
        matrix = 0.25*((Jp + Jm)*(5*Jz**2 - X - 0.5) + (5*Jz**2 - X - 0.5)*(Jp + Jm))
    elif [n,m] == [3,0]:
        matrix = 5*Jz**3 - (3*X-1)*Jz
    elif [n,m] == [3,-1]:
        matrix = -0.25j*((Jp - Jm)*(5*Jz**2 - X - 0.5) + (5*Jz**2 - X - 0.5)*(Jp - Jm))
    elif [n,m] == [3,-2]:
        matrix = -0.25j*(Jz*(Jp**2 - Jm**2) + (Jp**2 - Jm**2)*Jz)
    elif [n,m] == [3,-3]:
        matrix = -0.5j *(Jp**3 - Jm**3)

    elif [n,m] == [4,4]:
        matrix = 0.5 *(Jp**4 + Jm**4)
    elif [n,m] == [4,3]:
        matrix = 0.25 *((Jp**3 + Jm**3)*Jz + Jz*(Jp**3 + Jm**3))
    elif [n,m] == [4,2]:
        matrix = 0.25 *((Jp**2 + Jm**2)*(7*Jz**2 -X -5) + (7*Jz**2 -X -5)*(Jp**2 + Jm**2))
    elif [n,m] == [4,1]:
        matrix = 0.25 *((Jp + Jm)*(7*Jz**3 -(3*X+1)*Jz) + (7*Jz**3 -(3*X+1)*Jz)*(Jp + Jm))
    elif [n,m] == [4,0]:
        matrix = 35*Jz**4 - (30*X -25)*Jz**2 + 3*X**2 - 6*X
    elif [n,m] == [4,-4]:
        matrix = -0.5j *(Jp**4 - Jm**4)
    elif [n,m] == [4,-3]:
        matrix = -0.25j *((Jp**3 - Jm**3)*Jz + Jz*(Jp**3 - Jm**3))
    elif [n,m] == [4,-2]:
        matrix = -0.25j *((Jp**2 - Jm**2)*(7*Jz**2 -X -5) + (7*Jz**2 -X -5)*(Jp**2 - Jm**2))
    elif [n,m] == [4,-1]:
        matrix = -0.25j *((Jp - Jm)*(7*Jz**3 -(3*X+1)*Jz) + (7*Jz**3 -(3*X+1)*Jz)*(Jp - Jm))

    elif [n,m] == [6,6]:
        matrix = 0.5 *(Jp**6 + Jm**6)
    elif [n,m] == [6,5]:
        matrix = 0.25*((Jp**5 + Jm**5)*Jz + Jz*(Jp**5 + Jm**5))
    elif [n,m] == [6,4]:
        matrix = 0.25*((Jp**4 + Jm**4)*(11*Jz**2 -X -38) + (11*Jz**2 -X -38)*(Jp**4 + Jm**4))
    elif [n,m] == [6,3]:
        matrix = 0.25*((Jp**3 + Jm**3)*(11*Jz**3 -(3*X+59)*Jz) + (11*Jz**3 -(3*X+59)*Jz)*(Jp**3 + Jm**3))
    elif [n,m] == [6,2]:
        matrix = 0.25*((Jp**2 + Jm**2)*(33*Jz**4 -(18*X+123)*Jz**2 +X**2 +10*X +102) +\
                    (33*Jz**4 -(18*X+123)*Jz**2 +X**2 +10*X +102)*(Jp**2 + Jm**2))
    elif [n,m] == [6,1]:
        matrix = 0.25*((Jp +Jm)*(33*Jz**5 -(30*X-15)*Jz**3 +(5*X**2 -10*X +12)*Jz) +\
                    (33*Jz**5 -(30*X-15)*Jz**3 +(5*X**2 -10*X +12)*Jz)*(Jp+ Jm))
    elif [n,m] == [6,0]:
        matrix = 231*Jz**6 - (315*X-735)*Jz**4 + (105*X**2 -525*X +294)*Jz**2 -\
                 5*X**3 + 40*X**2 - 60*X
    elif [n,m] == [6,-6]:
        matrix = -0.5j *(Jp**6 - Jm**6)
    elif [n,m] == [6,-5]:
        matrix = -0.25j*((Jp**5 - Jm**5)*Jz + Jz*(Jp**5 - Jm**5))
    elif [n,m] == [6,-4]:
        matrix = -0.25j*((Jp**4 - Jm**4)*(11*Jz**2 -X -38) + (11*Jz**2 -X -38)*(Jp**4 - Jm**4))
    elif [n,m] == [6,-3]:
        matrix = -0.25j*((Jp**3 - Jm**3)*(11*Jz**3 -(3*X+59)*Jz) + (11*Jz**3 -(3*X+59)*Jz)*(Jp**3 - Jm**3))
    elif [n,m] == [6,-2]:
        matrix = -0.25j*((Jp**2 - Jm**2)*(33*Jz**4 -(18*X+123)*Jz**2 +X**2 +10*X +102) +\
                    (33*Jz**4 -(18*X+123)*Jz**2 +X**2 +10*X +102)*(Jp**2 - Jm**2))
    elif [n,m] == [6,-1]:
        matrix = -0.25j*((Jp - Jm)*(33*Jz**5 -(30*X-15)*Jz**3 +(5*X**2 -10*X +12)*Jz) +\
                 (33*Jz**5 -(30*X-15)*Jz**3 +(5*X**2 -10*X +12)*Jz)*(Jp - Jm))

    return matrix.O
#   return matrix


def LS_StevensOp(L,S,n,m):
    """generate stevens operator for a given total angular momentum
    and a given n and m state, but in the LS basis"""
    lmatrix = StevensOp(L,n,m)

    fullmatrix = np.hstack(np.hstack(np.multiply.outer(lmatrix, np.identity(int(2*S+1)))))
    return fullmatrix





#### Point Charge Approximation stuff


directory = os.path.dirname(os.path.realpath(__file__))+'/'
#Import prefactors
coef = np.genfromtxt(directory+'pcf_lib/TessHarmConsts.txt',delimiter = ',')
# Make into callable dictionary
keys=[str(int(c[0]))+','+str(int(c[1])) for c in coef]
prefac = dict(zip(keys,np.abs(coef[:,2])))


def Constant(n,m):
    '''Returns the constant in front of the tesseral harmonic'''
    nm = str(n)+','+str(m)
    return prefac[nm]


def TessHarm(n,m,x,y,z):
    """These functions have been cross-checked with mathematica's functions"""
    nm = str(n)+','+str(m)
    r = np.sqrt(x**2 + y**2 + z**2)

    if nm == '0,0':
        value = 1
    elif nm == '1,1':
        value = x/r
    elif nm == '1,0':
        value = z/r
    elif nm == '1,-1':
        value = y/r


    elif nm == '2,-2':
        value = 2*x*y/(r**2)
    elif nm == '2,-1':
        value = (y*z)/(r**2)
    elif nm == '2,0':
        value = (3*z**2 - r**2)/(r**2)
    elif nm == '2,1':
        value = x*z/(r**2)
    elif nm == '2,2':
        value = (x**2 - y**2)/(r**2)


    elif nm == '3,-3':
        value = (3*x**2 *y - y**3)/(r**3)
    elif nm == '3,-2':
        value = (2*x*y*z)/(r**3)
    elif nm == '3,-1':
        value = y*(5*z**2 - r**2)/(r**3)
    elif nm == '3,0':
        value = z*(5*z**2 - 3*r**2)/(r**3)
    elif nm == '3,1':
        value = x*(5*z**2 - r**2)/(r**3)
    elif nm == '3,2':
        value = z*(x**2 - y**2)/(r**3)
    elif nm == '3,3':
        value = (x**3 - 3*x*y**2)/(r**3)


    elif nm == '4,-4':
        value = 4*(x**3*y - x*y**3)/(r**4)
    elif nm == '4,-3':
        value = (3*x**2 *y - y**3)*z/(r**4)
    elif nm == '4,-2':
        value = 2*x*y*(7*z**2 - r**2)/(r**4)
    elif nm == '4,-1':
        value = y*z*(7*z**2 - 3*r**2)/(r**4)
    elif nm == '4,0':
        value = (35*z**4 - 30*z**2 *r**2 + 3*r**4)/(r**4)
    elif nm == '4,1':
        value = x*z*(7*z**2 - 3*r**2)/(r**4)
    elif nm == '4,2':
        value = (x**2 - y**2)*(7*z**2 - r**2)/(r**4)
    elif nm == '4,3':
        value = (x**3 - 3*x*y**2)*z/(r**4)
    elif nm == '4,4':
        value = (x**4 - 6*x**2*y**2 + y**4)/(r**4)

    # skipping 5 because it's almost never used.
    if n == 5:
        value = 0

    elif nm == '6,-6':
        value = (6*x**5*y - 20*x**3*y**3 + 6*x*y**5)/(r**6)
    elif nm == '6,-5':
        value = (5*x**4*y - 10*x**2*y**3 + y**5)*z/(r**6)
    elif nm == '6,-4':
        value = 4*(x**3*y - x*y**3)*(11*z**2 - r**2)/(r**6)
    elif nm == '6,-3':
        value = (3*x**2*y - y**3)*(11*z**3 - 3*z*r**2)/(r**6)
    elif nm == '6,-2':
        value = 2*x*y*(33*z**4 - 18*z**2 *r**2 + r**4)/(r**6)
    elif nm == '6,-1':
        value = y*z*(33*z**4 - 30*z**2 *r**2 + 5*r**4)/(r**6)
    elif nm == '6,0':
        value = (231*z**6 - 315*z**4 *r**2 + 105*z**2 *r**4 - 5*r**6)/(r**6)
    elif nm == '6,1':
        value = x*z*(33*z**4 - 30*z**2 *r**2 + 5*r**4)/(r**6)
    elif nm == '6,2':
        value = (x**2 - y**2)*(33*z**4 - 18*z**2 *r**2 + r**4)/(r**6)
    elif nm == '6,3':
        value = (x**3 - 3*x*y**2)*(11*z**3 - 3*z*r**2)/(r**6)
    elif nm == '6,4':
        value = (x**4 - 6*x**2*y**2 + y**4)*(11*z**2 - r**2)/(r**6)
    elif nm == '6,5':
        value = (x**5 - 10*x**3*y**2 + 5*x*y**4)*z/(r**6)
    elif nm == '6,6':
        value = (x**6 - 15.*x**4*y**2 + 15.*x**2*y**4 - y**6)/(r**6)

    return prefac[nm]*value


# #Used for testing purposes
# for i in xrange(7):
#   for j in xrange(i+1):
#       print i,',',j,',', TessHarm(i,j,1.1,0.1,0.3)

#Import Radial Integrals
radialI = {}
for line in open(directory+'pcf_lib/RadialIntegrals.txt'):
    if not line.startswith('#'):
        l = line.split(',')
        radialI[l[0]] = [float(v) for v in l[1:]]


def RadialIntegral(ion,n):
    """Returns the radial integral of a rare earth ion plus self-shielding"""
    shielding = 1- radialI[ion][int(n/2-1)]
    return radialI[ion][int(n/2-1) + 3] * shielding

########################################################################
# Multiplicative factor from Hutchings, Table VII

def PFalpha(L,S,l,halffilled=True):
    aaa = 2.*(2.*l+1.-4.*S)/((2.*l-1)*(2.*l+3.)*(2.*L-1.))
    if halffilled:  return aaa
    else:           return -aaa


def PFbeta(L,S,l,halffilled=True):
    bbb = 3.*(2.*l+1.-4.*S)*(-7.*(l-2.*S)*(l-2.*S+1.)+3.*(l-1.)*(l+2.))/\
        ((2.*l-3)*(2.*l-1)*(2.*l+3.)*(2.*l+5.)*(L-1.)*(2.*L-1.)*(2.*L-3.))
    if halffilled:  return bbb
    else:           return -bbb


def PFgamma(L,nvalence):
    '''We assume l=6 because only l=6 even has a gamma term.'''
    def O_06(L,Lz):
        X = L*(L+1.)
        return 231.*Lz**6 - (315.*X-735.)*Lz**4 + (105.*X**2 -525.*X +294.)*Lz**2 -\
                 5.*X**3 + 40.*X**2 - 60*X
    LLzexp = O_06(L,L)
    gamma6 = -4./3861. #from integration over spherical harmonics in l=3,m=3 state

    # calculate individual electron wave function expectation values:
    lzvalues = np.tile(np.arange(-3,4),2)
    IndividualElectron = 0
    for i in range(nvalence):
        IndividualElectron += O_06(3,lzvalues[i])

    return IndividualElectron/LLzexp*gamma6

########################################################################
# The following lookup table was generated from the functions above.
# This was done to save time in computation steps.
LSThet = {}
LSThet['Sm3+'] = [0.0148148148148, 0.0003848003848, -2.46666913334e-05]
LSThet['Pm3+'] = [0.0040404040404, 0.000122436486073, 1.12121324243e-05]
LSThet['Nd3+'] = [-0.0040404040404, -0.000122436486073, -1.12121324243e-05]
LSThet['Ce3+'] = [-0.0444444444444, 0.0040404040404, -0.001036001036]
LSThet['Dy3+'] = [-0.0148148148148, -0.0003848003848, 2.46666913334e-05]
LSThet['Ho3+'] = [-0.0040404040404, -0.000122436486073, -1.12121324243e-05]
LSThet['Tm3+'] = [0.0148148148148, 0.0003848003848, -2.46666913334e-05]
LSThet['Pr3+'] = [-0.0148148148148, -0.0003848003848, 2.46666913334e-05]
LSThet['Er3+'] = [0.0040404040404, 0.000122436486073, 1.12121324243e-05]
LSThet['Tb3+'] = [-0.0444444444444, 0.0040404040404, -0.001036001036]
LSThet['Yb3+'] = [0.0444444444444, -0.0040404040404, 0.001036001036]
def LStheta(ion,n):
    if isinstance(ion, str):
        return LSThet[ion][int(n/2-1)]



# Multiplicative factor for rare earth ground state multiplet
# from Hutchings, Table VI
 # Cross-checked by hand with calculator.
Thet = {}
Thet['Ce3+'] = [-2./(5*7), 2./(3*3*5*7), 0]
Thet['Pr3+'] = [-2.*2*13/(3*3*5*5*11), -2.*2/(3*3*5*11*11), 2.**4*17/(3**4*5*7*11**2*13)]
Thet['Nd3+'] = [-7./(3**2*11**2) , -2.**3*17/(3**3*11**3*13), -5.*17*19/(3**3*7*11**3*13**2)]
Thet['Pm3+'] = [2*7./(3*5*11**2), 2.**3*7*17/(3**3*5*11**3*13), 2.**3*17*19/(3**3*7*11**2*13**2)]
Thet['Sm3+'] = [13./(3**2*5*7) , 2.*13/(3**3*5*7*11), 0]
Thet['Tb3+'] = [-1./(3**2*11), 2./(3**3*5*11**2), -1./(3**4*7*11**2*13)]
Thet['Dy3+'] = [-2./(3**2*5*7) , -2.**3/(3**3*5*7*11*13), 2.*2/(3**3*7*11**2*13**2)]
Thet['Ho3+'] = [-1./(2*3*3*5*5), -1./(2*3*5*7*11*13), -5./(3**3*7*11**2*13**2)]
Thet['Er3+'] = [2.*2/(3*3*5*5*7) , 2./(3.**2*5*7*11*13), 2.*2*2/(3.**3*7*11**2*13**2)]
Thet['Tm3+'] = [1./(3**2*11) , 2.**3/(3**4*5*11**2), -5./(3**4*7*11**2*13)]
Thet['Yb3+'] = [2./(3**2*7) , -2./(3*5*7*11), 2.*2/(3**3*7*11*13)]
def theta(ion,n):
    return Thet[ion][int(n/2-1)]



Jion = {}   # [S, L, J]
Jion['Ni2+'] = [1., 3.]
Jion['Ni3+'] = [1., 2.]
# Rare earths
Jion['Ce3+'] = [0.5, 3., 2.5]
Jion['Pr3+'] = [1., 5., 4.]
Jion['Nd3+'] = [1.5, 6., 4.5]
Jion['Pm3+'] = [2., 6., 4.]
Jion['Sm3+'] = [2.5, 5, 2.5]
Jion['Eu3+'] = [3, 3, 0]
Jion['Gd3+'] = [7/2, 0, 7/2]
Jion['Tb3+'] = [3., 3., 6.]
Jion['Dy3+'] = [2.5, 5., 7.5]
Jion['Ho3+'] = [2., 6., 8.]
Jion['Er3+'] = [1.5, 6., 7.5]
Jion['Tm3+'] = [1., 5., 6.]
Jion['Yb3+'] = [0.5, 3., 3.5]
# def ionJ(ion):
#     return Jion[ion]


def LandeGFactor(ion):
    s, l, j = Jion[ion]
    return 1.5 + (s*(s+1.) - l*(l+1.))/(2.*j*(j+1.))



class Ligands:
    """For doing point-charge calculations"""
    def __init__(self,ion,ligandPos, latticeParams=None, ionPos=[0,0,0]):
        """Creates array of ligand bonds in cartesian coordinates"""
        lp = latticeParams
        if lp == None:
            self.latt = lat.lattice(1,1,1,90,90,90)
        elif len(lp) != 6:
            raise LookupError("latticeParams needs to have 6 components: a,b,c,alpha,beta,gamma")
        else:
            self.latt = lat.lattice(lp[0], lp[1], lp[2], lp[3], lp[4], lp[5])

        self.bonds = np.array([O - np.array(ionPos) for O in ligandPos])
        self.bonds = self.latt.cartesian(self.bonds).astype('float')
        self.bondlen = np.linalg.norm(self.bonds, axis=1)
        self.ion = ion

    def rotateLigands(self, oldaxis, newaxis):
        '''rotates the ligand bonds so that the new axis is in the direction of the old axis'''
        rotationAxis = np.cross(newaxis,oldaxis)
        rotationAngle = np.arccos(np.dot(newaxis,oldaxis)/(np.linalg.norm(newaxis)*np.linalg.norm(oldaxis)))
        self.bonds = np.array([self._rotateMatrix(b,rotationAxis,rotationAngle) for b in self.bonds])

    def rotateLigandsZ(self, oldaxis):
        '''rotates the ligand bonds around the z axis so that oldaxis 
        becomes the x axis'''
        zrotation = np.arctan(oldaxis[1]/oldaxis[0])
        self.bonds = np.array([self._rotateMatrix(b,np.array([0,0,1]),-zrotation) for b in self.bonds])


    def _rotateMatrix(self,matrixin,axis,angle):
        """taken from http://inside.mines.edu/fs_home/gmurray/ArbitraryAxisRotation/"""
        u, v, w = axis[0], axis[1], axis[2]
        norm = u**2 + v**2 + w**2
        
        rotmatrix = np.zeros((3,3))
        rotmatrix[0,0] = (u**2 +(v**2 + w**2)*np.cos(angle)) / norm
        rotmatrix[0,1] = (u*v*(1- np.cos(angle)) - w*np.sqrt(norm)*np.sin(angle)) / norm
        rotmatrix[0,2] = (u*w*(1- np.cos(angle)) + v*np.sqrt(norm)*np.sin(angle)) / norm
        rotmatrix[1,0] = (u*v*(1- np.cos(angle)) + w*np.sqrt(norm)*np.sin(angle)) / norm
        rotmatrix[1,1] = (v**2 +(u**2 + w**2)*np.cos(angle)) / norm
        rotmatrix[1,2] = (v*w*(1- np.cos(angle)) - u*np.sqrt(norm)*np.sin(angle)) / norm
        rotmatrix[2,0] = (u*w*(1- np.cos(angle)) - v*np.sqrt(norm)*np.sin(angle)) / norm
        rotmatrix[2,1] = (v*w*(1- np.cos(angle)) + u*np.sqrt(norm)*np.sin(angle)) / norm
        rotmatrix[2,2] = (w**2 +(v**2 + u**2)*np.cos(angle)) / norm

        # Simple matrix multiplication of matrixin is a vector
        if matrixin.size == 3:
            return np.dot(rotmatrix, matrixin)
        # R*m*R^T if matrixin is a matrix
        elif matrixin.size == 9:
            return np.dot(rotmatrix, np.dot(matrixin, rotmatrix.transpose() ))



    def PointChargeModel(self, symequiv=None, LigandCharge=-2,IonCharge=3, printB = True, 
                            suppressminusm = False, ionL=None):
        '''Create point charge model of the crystal fields of a rare-earth ion.
        Returns a CFLevels object with the hamiltonian defined.
        Define LigandCharge in units of e.'''

        self.IonCharge = IonCharge
        # Lock suppressmm into whatever it was when PointChargeModel was first called.
        try: self.suppressmm
        except AttributeError:
            self.suppressmm = suppressminusm

        if symequiv == None:
            # charge = IonCharge*[LigandCharge]*len(self.bonds)
            charge = [LigandCharge]*len(self.bonds)

        else:
            charge = [0]*len(self.bonds)
            for i,se in enumerate(symequiv):
                #charge[i] = IonCharge*LigandCharge[se]
                charge[i] = LigandCharge[se]

        
        
        ion=self.ion
        if ionL == None:
            ionJ = Jion[ion][2]
        else: ionJ = ionL

        # # print factors used:
        # print "#---------------------------------------"
        # print "# Stevens Factors \tRadial Integrals (a_0)"
        # for n in range(2,8,2):
        #     print ' ', theta(ion,n), '\t ', RadialIntegral(ion,n)
        # print '#---------------------------------------\n'

        ahc = 1.43996e4  #Constant to get the energy in units of meV = alpha*hbar*c
        a0 = 0.52917721067    #Bohr radius in \AA

        self.H = np.zeros((int(2*ionJ+1), int(2*ionJ+1)),dtype = complex)
        self.B = []

        if self.suppressmm == False:  nmrange = [[n,m] for n in range(2,8,2) for m in range(-n,n+1)]
        elif self.suppressmm == True:   nmrange = [[n,m] for n in range(2,8,2) for m in range(0,n+1)]
        #for n,m in [[n,m] for n in range(2,8,2) for m in range(-n,n+1)]:
        for n,m in nmrange:
            # 1)  Compute gamma
            gamma = 0
            for i in range(len(self.bonds)):
                gamma += 4*np.pi/(2*n+1)*charge[i] *\
                            TessHarm(n,m, self.bonds[i][0], self.bonds[i][1], self.bonds[i][2])/\
                            (self.bondlen[i]**(n+1))

            # 2)  Compute CEF parameter
            B = -gamma * ahc* a0**n * Constant(n,m) * RadialIntegral(ion,n) * theta(ion,n)
            if printB ==True: print('B_'+str(n),m,' = ',np.around(B,decimals=8))
            #print cef.StevensOp(ionJ,n,m)
            #self.H += np.around(B,decimals=15)*StevensOp(ionJ,n,m)
            self.H += B*StevensOp(ionJ,n,m)
            self.B.append(B)
        self.B = np.array(self.B)

        return CFLevels.Hamiltonian(self.H)



    def FitChargesNeutrons(self, chisqfunc, fitargs, method='Powell', **kwargs):
        '''fits neutron data'''

        # Define function to be fit
        fun, p0, resfunc = makeFitFunction(chisqfunc, fitargs, **dict(kwargs, LigandsObject=self) )

        print('\tFitting...')
        ############## Fit, using error function  #####################
        p_best = optimize.minimize(fun, p0, method=method)
        #p_best = optimize.minimize(fun, p0, method='Nelder-Mead')
        ###############################################################

        initialChisq, finalChisq = fun(p0), fun(p_best.x)

        # split back into values
        finalvals = resfunc(p_best.x)
        finalCharges = finalvals['LigandCharge']

        # Print results
        print("\n#*********************************")
        print("# Final Stevens Operator Values")
        newH = self.PointChargeModel(kwargs['symequiv'], finalCharges, printB=True)
        newH.diagonalize()
        print("\nFinal Charges: ", finalCharges)
        print('Final EigenValues: ', np.around(np.sort(newH.eigenvalues.real),3))

        return newH, finalvals


#######################################################################################


class CFLevels:
    """For calculating and fitting crystal field levels for an ion"""
    def __init__(self, StevensOperators, Parameters):
        """add Stevens operators to make a single hamiltonian matrix"""
        self.H = np.sum([a*b for a,b in zip(StevensOperators, Parameters)], axis=0)
        self.O = StevensOperators  #save these for a fit
        self.Ci = Parameters
        try:
            self.J = (len(self.H) -1.)/2
            self.opttran = opttransition(Operator.Jx(self.J).O, Operator.Jy(self.J).O.imag, Operator.Jz(self.J).O)
        except TypeError: pass


    @classmethod
    def Hamiltonian(cls, Hamil):
        newcls = cls([0,0],[0,0])  # Create empty class so we can just define Hamiltonian
        newcls.H = Hamil
        newcls.J = (len(Hamil) -1.)/2
        newcls.opttran = opttransition(Operator.Jx(newcls.J).O.real, Operator.Jy(newcls.J).O.imag, Operator.Jz(newcls.J).O.real)
        return newcls

    def diagonalize(self, Hamiltonian=None):
        """A Hamiltonian can be passed to the function (used for data fits)
        or the initially defined hamiltonian is used."""
        if Hamiltonian is None:
            Hamiltonian = self.H
        else:
            self.H = Hamiltonian
        diagonalH = LA.eigh(Hamiltonian)

        #self.eigenvaluesNoNorm = diagonalH[0]
        self.eigenvalues = diagonalH[0] - np.amin(diagonalH[0])
        self.eigenvectors = diagonalH[1].T
        # set very small values to zero
        tol = 1e-15
        self.eigenvalues[abs(self.eigenvalues) < tol] = 0.0
        self.eigenvectors[abs(self.eigenvectors) < tol] = 0.0


    def diagonalize_banded(self, Hamiltonian=None):
        '''same as above, but using the Scipy eig_banded function'''
        if Hamiltonian is None:
            Hamiltonian = self.H
        else:
            self.H = Hamiltonian

        bands = self._findbands(Hamiltonian)
        diagonalH = LA.eig_banded(bands, lower=True)

        #self.eigenvaluesNoNorm = diagonalH[0]
        self.eigenvalues = diagonalH[0] - np.amin(diagonalH[0])
        self.eigenvectors = diagonalH[1].T
        # set very small values to zero
        tol = 1e-15
        self.eigenvalues[abs(self.eigenvalues) < tol] = 0.0
        self.eigenvectors[abs(self.eigenvectors) < tol] = 0.0

    def _findbands(self, matrix):
        '''used in the diagonalize_banded function'''
        diags = np.zeros((len(matrix),len(matrix)))
        for i in range(len(matrix)):
            diag = matrix.diagonal(i)
            if i == 0:
                diags[i] = diag
            else:
                diags[i][:-i] = diag
            if np.count_nonzero(np.around(diag,10)) > 0:
                nonzerobands = i
        return diags[:nonzerobands+1]




    def neutronSpectrum(self, Earray, Temp, Ei, ResFunc, gamma = 0):
        # make angular momentum ket object
        #eigenkets = [Ket(ei) for ei in self.eigenvectors]
        eigenkets = self.eigenvectors.real
        intensity = np.zeros(len(Earray))

        # for population factor weights
        beta = 1/(8.61733e-2*Temp)  # Boltzmann constant is in meV/K
        Z = sum([np.exp(-beta*en) for en in self.eigenvalues])

        for i, ket_i in enumerate(eigenkets):
            # compute population factor
            pn = np.exp(-beta *self.eigenvalues[i])/Z
            if pn > 1e-3:  #only compute for transitions with enough weight
                for j, ket_j in enumerate(eigenkets):
                    # compute amplitude
                    #mJn = self._transition(ket_i,ket_j)   # Old: slow
                    mJn = self.opttran.transition(ket_i,ket_j)
                    deltaE = self.eigenvalues[j] - self.eigenvalues[i]
                    GausWidth = ResFunc(deltaE)  #peak width due to instrument resolution
                    intensity += ((pn * mJn * self._voigt(x=Earray, x0=deltaE, alpha=GausWidth, 
                                                        gamma=gamma)).real).astype('float64')
                #intensity += ((pn * mJn * self._lorentzian(Earray, deltaE, Width)).real).astype('float64')

        ## List comprehension: turns out this way was slower.
        # intensity = np.sum([
        #     np.exp(-beta *self.eigenvalues[i])/Z *\
        #     self._transition(eigenkets[i], eigenkets[j]) *\
        #     self._voigt(x=Earray, x0=(self.eigenvalues[j] - self.eigenvalues[i]), 
        #         alpha=ResFunc(self.eigenvalues[j] - self.eigenvalues[i]), gamma=gamma)
        #     for i in range(len(eigenkets)) for j in range(len(eigenkets))
        #     ], axis = 0)

        kpoverk = np.sqrt((Ei - Earray)/Ei) #k'/k = sqrt(E'/E)
        return intensity * kpoverk

    def normalizedNeutronSpectrum(self, Earray, Temp, ResFunc, gamma = 0):
        '''1D neutron spectrum without the Kf/Ki correction'''
        # make angular momentum ket object
        # eigenkets = [Ket(ei) for ei in self.eigenvectors]
        eigenkets = self.eigenvectors.real
        intensity = np.zeros(len(Earray))

        # for population factor weights
        beta = 1/(8.61733e-2*Temp)  # Boltzmann constant is in meV/K
        Z = sum([np.exp(-beta*en) for en in self.eigenvalues])

        for i, ket_i in enumerate(eigenkets):
            # compute population factor
            pn = np.exp(-beta *self.eigenvalues[i])/Z
            if pn > 1e-3:  #only compute for transitions with enough weight
                for j, ket_j in enumerate(eigenkets):
                    # compute amplitude
                    #mJn = self._transition(ket_i,ket_j)  # Old: slow
                    mJn = self.opttran.transition(ket_i,ket_j)
                    deltaE = self.eigenvalues[j] - self.eigenvalues[i]
                    GausWidth = ResFunc(deltaE)  #peak width due to instrument resolution
                    intensity += ((pn * mJn * self._voigt(x=Earray, x0=deltaE, alpha=GausWidth, 
                                                        gamma=gamma)).real).astype('float64')
        return intensity


    def neutronSpectrum2D(self, Earray, Qarray, Temp, Ei, ResFunc, gamma, DebyeWaller, Ion):
        intensity1D = self.neutronSpectrum(Earray, Temp, Ei, ResFunc,  gamma)

        # Scale by Debye-Waller Factor
        DWF = np.exp(1./3. * Qarray**2 * DebyeWaller**2)
        # Scale by form factor
        FormFactor = RE_FormFactor(Qarray,Ion)
        return np.outer(intensity1D, DWF*FormFactor)


    def _transition(self,ket1,ket2):  ## Correct, but slow.
        """Computes \sum_a |<|J_a|>|^2"""
        # Jx = Operator.Jx(ket1.j)
        # Jy = Operator.Jy(ket1.j)
        # Jz = Operator.Jz(ket1.j)
        # ax = np.dot(ket1.ket,np.dot(Jx.O,ket2.ket)) * np.dot(ket2.ket,np.dot(Jx.O,ket1.ket))
        # ay = np.dot(ket1.ket,np.dot(Jy.O,ket2.ket)) * np.dot(ket2.ket,np.dot(Jy.O,ket1.ket))
        # az = np.dot(ket1.ket,np.dot(Jz.O,ket2.ket)) * np.dot(ket2.ket,np.dot(Jz.O,ket1.ket))

        ax = (ket1*ket2.Jx() )*( ket2*ket1.Jx() )
        ay = (ket1*ket2.Jy() )*( ket2*ket1.Jy() )
        az = (ket1*ket2.Jz() )*( ket2*ket1.Jz() )
        # eliminate tiny values
        ax, ay, az = np.around(ax, 10), np.around(ay, 10), np.around(az, 10)
        if (ax + ay + az).imag == 0:
            return ((ax + ay + az).real).astype(float)
        else:
            print(ax, ay, az)
            raise ValueError("non-real amplitude. Error somewhere.")
            
    def _lorentzian(self, x, x0, gamma):
        return 1/np.pi * (0.5*gamma)/((x-x0)**2 + (0.5*gamma)**2)

    def _voigt(self, x, x0, alpha, gamma):
        """ Return the Voigt line shape at x with Lorentzian component FWHM gamma
        and Gaussian component FWHM alpha."""
        sigma = (0.5*alpha) / np.sqrt(2 * np.log(2))
        return np.real(wofz(((x-x0) + 1j*(0.5*gamma))/sigma/np.sqrt(2))) / sigma\
                                                            /np.sqrt(2*np.pi)

    def _Re(self,value):
        thresh = 1e-9
        if np.size(value) == 1 & isinstance(value, complex):
            if value.imag <= thresh:
                return (value.real).astype(float)
            else: 
                return value
        else:
            if np.all(value.imag < thresh):
                return (value.real)
            else: return value

    def printEigenvectors(self):
        '''prints eigenvectors and eigenvalues in a matrix'''
        print('\n Eigenvalues \t Eigenvectors')
        print('\t\t'+'-------'*(len(self.eigenvalues)+1))
        sortinds = self.eigenvalues.argsort()
        sortEVal= np.around(self.eigenvalues[sortinds],5)
        sortEVec= np.around(self.eigenvectors[sortinds],3)
        for i in range(len(sortinds)):
            print(format(self._Re(sortEVal[i]), '.5f'),'\t| ', self._Re(sortEVec[i]),' |')
        print('\t\t'+'-------'*(len(self.eigenvalues)+1) + '\n')

    def printLaTexEigenvectors(self, precision = 4):
        '''prints eigenvectors and eigenvalues in the output that Latex can read'''
        print('\\begin{table*}\n\\caption{Eigenvectors and Eigenvalues...}')
        print('\\begin{ruledtabular}')
        numev = len(self.eigenvalues)
        print('\\begin{tabular}{c|'+'c'*numev+'}')
        if numev % 2 == 1:
            print('E (meV) &'+' & '.join(['$|'+str(int(kk))+'\\rangle$' for kk in 
                        np.arange(-(numev-1)/2,numev/2)])
                +' \\tabularnewline\n \\hline ')
        else:
            print('E (meV) &'+
                ' & '.join(['$| -\\frac{'+str(abs(kk))+'}{2}\\rangle$' if kk <0
                            else '$| \\frac{'+str(abs(kk))+'}{2}\\rangle$'
                            for kk in np.arange(-(numev-1),numev,2)])
                +' \\tabularnewline\n \\hline ')
        sortinds = self.eigenvalues.argsort()
        sortEVal= np.around(self.eigenvalues[sortinds],3)
        sortEVec= np.around(self.eigenvectors[sortinds],precision)
        for i in range(len(sortinds)):
            print(format(self._Re(sortEVal[i]), '.3f'),'&', 
                ' & '.join([str(eevv) for eevv in self._Re(sortEVec[i])]), '\\tabularnewline')
        print('\\end{tabular}\\end{ruledtabular}')
        print('\\label{flo:Eigenvectors}\n\\end{table*}')

    def gsExpectation(self):
        """Prints <J_x>, <J_y>, and <J_z> for the ground state"""
        zeroinds = np.where(np.around(self.eigenvalues,7)==0)
        gsEVec = self.eigenvectors[zeroinds]
        print('\t Ground State Expectation Values:')
        for ev in gsEVec:
            vv = Ket(ev)
            jjxx = self._Re( vv*vv.Jx() )
            jjyy = self._Re( vv*vv.Jy() )
            jjzz = self._Re( vv*vv.Jz() )
            print('  <J_x> =',jjxx,'\t<J_y> =',jjyy,'\t<J_z> =',jjzz)
        print(' ')


    def magnetization(self, ion, Temp, Field):
        '''field should be a 3-component vector. Temps may be an array.'''
        if len(Field) != 3: 
            raise TypeError("Field needs to be 3-component vector")

        # A) Define magnetic Hamiltonian

        Jx = Operator.Jx(self.J)
        Jy = Operator.Jy(self.J)
        Jz = Operator.Jz(self.J)

        #print(Jx)
        #print(Jy)
        if isinstance(ion, str):
            gJ = LandeGFactor(ion)
        else: gJ = ion
        muB = 5.7883818012e-2  # meV/T
        #mu0 = np.pi*4e-7       # T*m/A
        JdotB = gJ*muB*(Field[0]*Jx + Field[1]*Jy + Field[2]*Jz)

        # B) Diagonalize full Hamiltonian
        FieldHam = self.H + JdotB.O
        diagonalH = LA.eigh(FieldHam)

        minE = np.amin(diagonalH[0])
        evals = diagonalH[0] - minE
        evecs = diagonalH[1].T
        # These ARE actual eigenvalues.

        # C) Compute expectation value along field
        JexpVals = np.zeros((len(evals),3))
        for i, ev in enumerate(evecs):
            kev = Ket(ev)
            # print np.real(np.dot(ev,kev.Jy().ket)), np.real(np.dot(ev,np.dot(Jy.O,ev)))
            # print np.real(kev*kev.Jy()) - np.real(np.dot(ev,np.dot(Jy.O,ev)))
            JexpVals[i] =[np.real(kev*kev.Jx()),
                          np.real(kev*kev.Jy()),
                          np.real(kev*kev.Jz())]
        k_B = 8.6173303e-2  # meV/K

        if (isinstance(Temp, int) or isinstance(Temp, float)):
            Zz = np.sum(np.exp(-evals/(k_B*Temp)))
            JexpVal = np.dot(np.exp(-evals/(k_B*Temp)),JexpVals)/Zz
            return gJ*JexpVal
        else:
            expvals, temps = np.meshgrid(evals, Temp)
            ZZ = np.sum(np.exp(-expvals/temps/k_B), axis=1)
            JexpValList = np.repeat(JexpVals.reshape((1,)+JexpVals.shape), len(Temp), axis=0)
            JexpValList = np.sum(np.exp(-expvals/temps/k_B)*\
                                np.transpose(JexpValList, axes=[2,0,1]), axis=2) / ZZ
            # if np.isnan(JexpValList).any():
            #     print -expvals[0]/temps[0]/k_B
            #     print np.exp(-expvals/temps/k_B)[0]
            #     raise ValueError('Nan in result!')
            return np.nan_to_num(gJ*JexpValList.T)


    def susceptibility(self, ion, Temps, Field, deltaField):
        '''Computes susceptibility numerically with a numerical derivative.
        deltaField needs to be a scalar value.'''
        if not isinstance(deltaField, float):
            raise TypeError("Deltafield needs to be a scalar")

        if isinstance(Field, float):
            # Assume we are computing a powder average
            VecField = Field * np.array([1,0,0])
            Delta = deltaField*np.array(VecField)/Field
            Mplus1 = self.magnetization(ion, Temps, VecField + Delta)
            Mminus1= self.magnetization(ion, Temps, VecField - Delta)
            Mplus2 = self.magnetization(ion, Temps, VecField + 2*Delta)
            Mminus2= self.magnetization(ion, Temps, VecField - 2*Delta)

            dMdH_x = (8*(Mplus1 - Mminus1) - (Mplus2 - Mminus2))/(12*deltaField)

            VecField = Field * np.array([0,1,0])
            Delta = deltaField*np.array(VecField)/Field
            Mplus1 = self.magnetization(ion, Temps, VecField + Delta)
            Mminus1= self.magnetization(ion, Temps, VecField - Delta)
            Mplus2 = self.magnetization(ion, Temps, VecField + 2*Delta)
            Mminus2= self.magnetization(ion, Temps, VecField - 2*Delta)

            dMdH_y = (8*(Mplus1 - Mminus1) - (Mplus2 - Mminus2))/(12*deltaField)

            VecField = Field * np.array([0,0,1])
            Delta = deltaField*np.array(VecField)/Field
            Mplus1 = self.magnetization(ion, Temps, VecField + Delta)
            Mminus1= self.magnetization(ion, Temps, VecField - Delta)
            Mplus2 = self.magnetization(ion, Temps, VecField + 2*Delta)
            Mminus2= self.magnetization(ion, Temps, VecField - 2*Delta)

            dMdH_z = (8*(Mplus1 - Mminus1) - (Mplus2 - Mminus2))/(12*deltaField)

            return (dMdH_x[:,0]+dMdH_y[:,1]+dMdH_z[:,2])/3.

        elif len(Field) == 3:
            Delta = deltaField*np.array(Field)/np.linalg.norm(Field)
            Mplus1 = self.magnetization(ion, Temps, Field + Delta)
            Mminus1= self.magnetization(ion, Temps, Field - Delta)
            Mplus2 = self.magnetization(ion, Temps, Field + 2*Delta)
            Mminus2= self.magnetization(ion, Temps, Field - 2*Delta)

            dMdH = (8*(Mplus1 - Mminus1) - (Mplus2 - Mminus2))/(12*deltaField)
            #dMdH = (Mplus1 - Mminus1)/(2*deltaField)

            return dMdH

    def susceptibilityPert(self, ion, Temps):
        # Compute susceptibility from perturbation theory, using MesotFurer eq. 11
        gJ = LandeGFactor(ion)
        muB = 5.7883818012e-2  # meV/T
        k_B = 8.6173303e-2  # meV/K

        # In this case, we assume powder average.

        Jx = Operator.Jx(self.J)
        Jy = Operator.Jy(self.J)
        Jz = Operator.Jz(self.J)


        expvals, temps = np.meshgrid(self.eigenvalues, Temps)
        ZZ = np.sum(np.exp(-expvals/temps/k_B), axis=1)

        suscept = np.zeros(len(Temps))
        for i, ev1 in enumerate(self.eigenvectors):
            kev = Ket(ev1)
            suscept += (np.exp(-self.eigenvalues[i]/Temps/k_B)/ ZZ/ k_B/ Temps) *\
                            np.mean([np.real((kev*kev.Jx()) * (kev*kev.Jx())),
                                    np.real((kev*kev.Jy()) * (kev*kev.Jy())),
                                    np.real((kev*kev.Jz()) * (kev*kev.Jz()))])

            for j, ev2 in enumerate(self.eigenvectors):
                if i == j: continue
                #elif (self.eigenvalues[i]- self.eigenvalues[j]) > 1e-14:
                else:
                    kev2 = Ket(ev2)
                    suscept += ((np.exp(-self.eigenvalues[j]/Temps/k_B)- 
                        np.exp(-self.eigenvalues[i]/Temps/k_B))/ ZZ)/\
                            (self.eigenvalues[i]- self.eigenvalues[j]) *\
                            np.mean([np.real((kev2*kev.Jx()) * (kev*kev2.Jx())),
                                    np.real((kev2*kev.Jy()) * (kev*kev2.Jy())),
                                    np.real((kev2*kev.Jz()) * (kev*kev2.Jz()))])
        return gJ*gJ*muB*suscept


    def gtensor(self):
        '''Returns g tensor computed numerically'''

        self.diagonalize_banded()

        def eliminateimag(number):
            num = np.around(number, 10)
            if num.imag == 0:
                return (num.real).astype(float)
            else:
                return number

        zeroinds = np.where(np.around(self.eigenvalues,10)==0)
        gsEVec = self.eigenvectors[zeroinds]
        vv1 = np.around(gsEVec[0],10)
        vv2 = np.around(gsEVec[1],10)
        Jx = Operator.Jx(self.J).O
        Jy = Operator.Jy(self.J).O
        Jz = Operator.Jz(self.J).O
        #print(vv1,'\n',vv2)
        jz01 = eliminateimag( np.dot(vv1,np.dot(Jz,vv2)) )
        jz10 = eliminateimag( np.dot(vv2,np.dot(Jz,vv1)) )
        jz00 = eliminateimag( np.dot(vv1,np.dot(Jz,vv1)) )
        jz11 = eliminateimag( np.dot(vv2,np.dot(Jz,vv2)) )
        
        
        jx01 = eliminateimag( np.dot(vv1,np.dot(Jx,vv2)) )
        jx10 = eliminateimag( np.dot(vv2,np.dot(Jx,vv1)) )
        jx00 = eliminateimag( np.dot(vv1,np.dot(Jx,vv1)) )
        jx11 = eliminateimag( np.dot(vv2,np.dot(Jx,vv2)) )
        
        jy01 = eliminateimag( np.dot(vv1,np.dot(Jy,vv2)) )
        jy10 = eliminateimag( np.dot(vv2,np.dot(Jy,vv1)) )
        jy00 = eliminateimag( np.dot(vv1,np.dot(Jy,vv1)) )
        jy11 = eliminateimag( np.dot(vv2,np.dot(Jy,vv2)) )
        
        gg = 2*np.array([[np.real(jx01), np.imag(jx01), jx00],
                         [np.real(jy01), np.imag(jy01), jy00],
                         [np.real(jz01), np.imag(jz01), np.abs(jz00)]])
        return gg


    # def gtensor(self, field=0.1, Temp=0.1):
    #     '''Returns g tensor computed numerically from zeeman splitting'''
    #     Jx = Operator.Jx(self.J)
    #     Jy = Operator.Jy(self.J)
    #     Jz = Operator.Jz(self.J)

    #     #print(Jx)
    #     #print(Jy)
    #     muB = 5.7883818012e-2  # meV/T
    #     #mu0 = np.pi*4e-7       # T*m/A

    #     gg = np.zeros(3)
    #     #loop through x,y,z
    #     for i,Field in enumerate([[field,0,0], [0,field,0], [0,0,field]]):
    #         JdotB = muB*(Field[0]*Jx + Field[1]*Jy + Field[2]*Jz)

    #         # B) Diagonalize full Hamiltonian
    #         FieldHam = self.H + JdotB.O
    #         diagonalH = LA.eigh(FieldHam)

    #         minE = np.amin(diagonalH[0])
    #         evals = diagonalH[0] - minE
    #         evecs = diagonalH[1].T

    #         DeltaZeeman = evals[1]-evals[0]
    #         print(DeltaZeeman)

    #         # Now find the expectation value of J
    #         JexpVals = np.zeros((len(evals),3))
    #         for ii, ev in enumerate(evecs):
    #             kev = Ket(ev)
    #             JexpVals[ii] =[np.real(kev*kev.Jx()),
    #                           np.real(kev*kev.Jy()),
    #                           np.real(kev*kev.Jz())]
    #         k_B = 8.6173303e-2  # meV/K

    #         Zz = np.sum(np.exp(-evals/(k_B*Temp)))
    #         JexpVal = np.dot(np.exp(-evals/(k_B*Temp)),JexpVals)/Zz

    #         expectationJ = JexpVal[i]

    #         # calculate g values
    #         gg[i] = DeltaZeeman/(muB*field*expectationJ)
            
    #     return gg


    def fitdata(self, chisqfunc, fitargs, method='Powell', **kwargs):
        '''fits data to CEF parameters'''

        # define parameters
        # if len(self.Ci) != len(kwargs['coeff']):
        #     raise ValueError('coeff needs to have the same length as self.Ci')

        # Define function to be fit
        fun, p0, resfunc = makeFitFunction(chisqfunc, fitargs, **dict(kwargs, CFLevelsObject=self) )

        ############## Fit, using error function  #####################
        p_best = optimize.minimize(fun, p0, method=method)
        ###############################################################

        print(fun(p_best.x))
        print(chisqfunc(self, **kwargs))
        initialChisq, finalChisq = chisqfunc(self, **kwargs), fun(p_best.x)
        print('\rInitial err =', initialChisq, '\tFinal err =', finalChisq)
        
        result = resfunc(p_best.x)
        #print '\nFinal values: ', result
        result['Chisq'] = finalChisq
        return result

    def fitdata_GlobalOpt(self, chisqfunc, fitargs, **kwargs):
        '''fits data to CEF parameters using the basin hopping algorithm'''

        # Define function to be fit
        fun, p0, resfunc = makeFitFunction(chisqfunc, fitargs, **dict(kwargs, CFLevelsObject=self) )

        ############## Fit, using error function  #####################
        p_best = optimize.basinhopping(fun, p0, niter=100, T = 1e5)
        ###############################################################

        print(fun(p_best.x))
        print(chisqfunc(self, **kwargs))
        initialChisq, finalChisq = chisqfunc(self, **kwargs), fun(p_best.x)
        print('\rInitial err =', initialChisq, '\tFinal err =', finalChisq)
        
        result = resfunc(p_best.x)
        #print '\nFinal values: ', result
        result['Chisq'] = finalChisq
        return result


    def testEigenvectors(self):
        """Tests if eigenvectors are really eigenvectors"""
        print('testing eigenvectors... (look for large values)')
        for i in range(len(self.eigenvalues)):
            print(np.around(
                np.dot(self.H,self.eigenvectors[i]) - self.eigenvectors[i]*self.eigenvaluesNoNorm[i],
                10))

        print('\n Sum rule (two values should be equal):')
        TotalTransition = 0
        for i, ev in enumerate(self.eigenvectors):
            TotalTransition += self._transition(Ket(self.eigenvectors[1]),Ket(ev))
        print(TotalTransition, '  ', self.J*(self.J+1))




from numba import njit, jitclass
from numba import float64

import warnings
warnings.filterwarnings('ignore')

spec = [ 
    ('Jx', float64[:,:]),          # an array field
    ('Jy', float64[:,:]),
    ('Jz', float64[:,:])
]

@jitclass(spec)
class opttransition(object):
    def __init__(self, optJx, optJy, optJz):
        self.Jx = np.zeros((len(optJx),len(optJx)), dtype=np.float64)
        self.Jy = np.zeros((len(optJx),len(optJx)), dtype=np.float64)
        self.Jz = np.zeros((len(optJx),len(optJx)), dtype=np.float64)
        self.Jx = optJx
        self.Jy = optJy
        self.Jz = optJz

    def transition(self,ket1, ket2):
        ax = np.dot(ket1, np.dot(self.Jx, ket2))**2
        ay = np.dot(ket1, np.dot(self.Jy, ket2))**2
        az = np.dot(ket1, np.dot(self.Jz, ket2))**2
        return ax + ay + az




def rescaleCEF(ion1, ion2, B, n):
    '''Uses the point charge model to scale a CEF parameter (B)
    from one ion to another. Assumes precisely the same ligand environment
    with a different magnetic ion thrown in.'''
    scalefact = (RadialIntegral(ion2,n)*theta(ion2,n))/(RadialIntegral(ion1,n)*theta(ion1,n))
    return B*scalefact











####################################3
##   
##    ##                #######
##    ##             #############
##    ##             ###       ###
##    ##             ###
##    ##              ####
##    ##                #####
##    ##                   #####
##    ##                     ####
##    ##                       ###
##    ##             ###       ###
##    ###########     ####   ####
##    ###########       #######
##
######################################









class LSOperator():
    '''This is for a full treatment in the intermediate coupling scheme'''
    def __init__(self, L, S):
        self.O = np.zeros((int((2*L+1)*(2*S+1)), int((2*L+1)*(2*S+1)) ))
        self.L = L
        self.S = S
        lm = np.arange(-L,L+1,1)
        sm = np.arange(-S,S+1,1)
        self.Lm = np.repeat(lm, len(sm))
        self.Sm = np.tile(sm, len(lm))
    
    @staticmethod
    def Lz(L, S):
        obj = LSOperator(L, S)
        for i in range(len(obj.O)):
            for k in range(len(obj.O)):
                if i == k:
                    obj.O[i,k] = (obj.Lm[k])
        return obj

    @staticmethod
    def Lplus(L, S):
        obj = LSOperator(L, S)
        for i, lm1 in enumerate(obj.Lm):
            for k, lm2 in enumerate(obj.Lm):
                if (lm1 - lm2 == 1) and (obj.Sm[i] == obj.Sm[k] ):
                    obj.O[i,k] = np.sqrt((obj.L-obj.Lm[k])*(obj.L+obj.Lm[k]+1))
        return obj

    @staticmethod
    def Lminus(L, S):
        obj = LSOperator(L, S)
        for i, lm1 in enumerate(obj.Lm):
            for k, lm2 in enumerate(obj.Lm):
                if (lm2 - lm1 == 1)  and (obj.Sm[i] == obj.Sm[k]):
                    obj.O[i,k] = np.sqrt((obj.L+obj.Lm[k])*(obj.L-obj.Lm[k]+1))
        return obj

    @staticmethod
    def Lx(L, S):
        objp = LSOperator.Lplus(L, S)
        objm = LSOperator.Lminus(L, S)
        return 0.5*objp + 0.5*objm

    @staticmethod
    def Ly(L, S):
        objp = LSOperator.Lplus(L, S)
        objm = LSOperator.Lminus(L, S)
        return -0.5j*objp + 0.5j*objm

    ##################################
    # Spin operators
    @staticmethod
    def Sz(L, S):
        obj = LSOperator(L, S)
        for i in range(len(obj.O)):
            for k in range(len(obj.O)):
                if i == k:
                    obj.O[i,k] = (obj.Sm[k])
        return obj

    @staticmethod
    def Splus(L, S):
        obj = LSOperator(L, S)
        for i, sm1 in enumerate(obj.Sm):
            for k, sm2 in enumerate(obj.Sm):
                if (sm1 - sm2 == 1) and (obj.Lm[i] == obj.Lm[k]):
                    obj.O[i,k] = np.sqrt((obj.S-obj.Sm[k])*(obj.S+obj.Sm[k]+1))
        return obj

    @staticmethod
    def Sminus(L, S):
        obj = LSOperator(L, S)
        for i, sm1 in enumerate(obj.Sm):
            for k, sm2 in enumerate(obj.Sm):
                if (sm2 - sm1 == 1) and (obj.Lm[i] == obj.Lm[k]):
                    obj.O[i,k] = np.sqrt((obj.S+obj.Sm[k])*(obj.S-obj.Sm[k]+1))
        return obj

    @staticmethod
    def Sx(L, S):
        objp = LSOperator.Splus(L, S)
        objm = LSOperator.Sminus(L, S)
        return 0.5*objp + 0.5*objm

    @staticmethod
    def Sy(L, S):
        objp = LSOperator.Splus(L, S)
        objm = LSOperator.Sminus(L, S)
        return -0.5j*objp + 0.5j*objm


    def __add__(self,other):
        newobj = LSOperator(self.L, self.S)
        try:
            newobj.O = np.add(self.O, other.O)
        except AttributeError:
            newobj.O = self.O + other*np.identity(len(self.O))
        return newobj

    def __radd__(self,other):
        newobj = LSOperator(self.L, self.S)
        try:
            newobj.O = np.add(other.O, self.O)
        except AttributeError:
            newobj.O = self.O + other*np.identity(len(self.O))
        return newobj

    def __sub__(self,other):
        newobj = LSOperator(self.L, self.S)
        try:
            newobj.O = self.O - other.O
        except AttributeError:
            newobj.O = self.O - other*np.identity(len(self.O))
        return newobj

    def __mul__(self,other):
        newobj = LSOperator(self.L, self.S)
        try:
            newobj.O = np.dot(self.O, other.O)
        except AttributeError:
            newobj.O = other * self.O
        return newobj

    def __rmul__(self,other):
        newobj = LSOperator(self.L, self.S)
        try:
            newobj.O = np.dot(other.O, self.O)
        except AttributeError:
            newobj.O = other * self.O
        return newobj

    def __pow__(self, power):
        newobj = LSOperator(self.L, self.S)
        newobj.O = self.O
        for i in range(power-1):
            newobj.O = np.dot(newobj.O,self.O)
        return newobj

    def __neg__(self):
        newobj = LSOperator(self.L, self.S)
        newobj.O = -self.O
        return newobj

    def __repr__(self):
        return repr(self.O)


    # Computing magnetization and susceptibility

    def magnetization(self, Temp, Field):
        '''field should be a 3-component vector. Temps may be an array.'''
        if len(Field) != 3: 
            raise TypeError("Field needs to be 3-component vector")

        # A) Define magnetic Hamiltonian
        Lx = LSOperator.Lx(self.L, self.S)
        Ly = LSOperator.Ly(self.L, self.S)
        Lz = LSOperator.Lz(self.L, self.S)
        Sx = LSOperator.Sx(self.L, self.S)
        Sy = LSOperator.Sy(self.L, self.S)
        Sz = LSOperator.Sz(self.L, self.S)

        g0 = 2.002319
        Jx = Lx + g0*Sx
        Jy = Ly + g0*Sy
        Jz = Lz + g0*Sz

        muB = 5.7883818012e-2  # meV/T
        #mu0 = np.pi*4e-7       # T*m/A
        JdotB = muB*((Field[0]*Lx + Field[1]*Ly + Field[2]*Lz) +\
                        (Field[0]*Sx + Field[1]*Sy + Field[2]*Sz))

        # B) Diagonalize full Hamiltonian
        FieldHam = self.O + JdotB.O
        diagonalH = LA.eigh(FieldHam)

        minE = np.amin(diagonalH[0])
        evals = diagonalH[0] - minE
        evecs = diagonalH[1].T
        # These ARE actual eigenvalues.

        # C) Compute expectation value along field
        JexpVals = np.zeros((len(evals),3))
        for i, ev in enumerate(evecs):
            #print np.real(np.dot(ev, np.dot( self.O ,ev))), diagonalH[0][i]
            #print np.real(np.dot( FieldHam ,ev)), np.real(diagonalH[0][i]*ev)
            JexpVals[i] =[np.real(np.dot(np.conjugate(ev), np.dot( Jx.O ,ev))),
                          np.real(np.dot(np.conjugate(ev), np.dot( Jy.O ,ev))),
                          np.real(np.dot(np.conjugate(ev), np.dot( Jz.O ,ev)))]
        k_B = 8.6173303e-2  # meV/K

        if (isinstance(Temp, int) or isinstance(Temp, float)):
            Zz = np.sum(np.exp(-evals/(k_B*Temp)))
            JexpVal = np.dot(np.exp(-evals/(k_B*Temp)),JexpVals)/Zz
            return np.real(JexpVal)
        else:
            expvals, temps = np.meshgrid(evals, Temp)
            ZZ = np.sum(np.exp(-expvals/temps/k_B), axis=1)
            JexpValList = np.repeat(JexpVals.reshape((1,)+JexpVals.shape), len(Temp), axis=0)
            JexpValList = np.sum(np.exp(-expvals/temps/k_B)*\
                                np.transpose(JexpValList, axes=[2,0,1]), axis=2) / ZZ
            # if np.isnan(JexpValList).any():
            #     print -expvals[0]/temps[0]/k_B
            #     print np.exp(-expvals/temps/k_B)[0]
            #     raise ValueError('Nan in result!')
            return np.nan_to_num(JexpValList.T)


    def susceptibility(self, Temps, Field, deltaField):
        '''Computes susceptibility numerically with a numerical derivative.
        deltaField needs to be a scalar value.'''
        if not isinstance(deltaField, float):
            raise TypeError("Deltafield needs to be a scalar")

        if isinstance(Field, float):
            # Assume we are computing a powder average
            VecField = Field * np.array([1,0,0])
            Delta = deltaField*np.array(VecField)/Field
            Mplus1 = self.magnetization(Temps, VecField + Delta)
            Mminus1= self.magnetization(Temps, VecField - Delta)
            Mplus2 = self.magnetization(Temps, VecField + 2*Delta)
            Mminus2= self.magnetization(Temps, VecField - 2*Delta)

            dMdH_x = (8*(Mplus1 - Mminus1) - (Mplus2 - Mminus2))/(12*deltaField)

            VecField = Field * np.array([0,1,0])
            Delta = deltaField*np.array(VecField)/Field
            Mplus1 = self.magnetization(Temps, VecField + Delta)
            Mminus1= self.magnetization(Temps, VecField - Delta)
            Mplus2 = self.magnetization(Temps, VecField + 2*Delta)
            Mminus2= self.magnetization(Temps, VecField - 2*Delta)

            dMdH_y = (8*(Mplus1 - Mminus1) - (Mplus2 - Mminus2))/(12*deltaField)

            VecField = Field * np.array([0,0,1])
            Delta = deltaField*np.array(VecField)/Field
            Mplus1 = self.magnetization(Temps, VecField + Delta)
            Mminus1= self.magnetization(Temps, VecField - Delta)
            Mplus2 = self.magnetization(Temps, VecField + 2*Delta)
            Mminus2= self.magnetization(Temps, VecField - 2*Delta)

            dMdH_z = (8*(Mplus1 - Mminus1) - (Mplus2 - Mminus2))/(12*deltaField)

            return (dMdH_x[:,0]+dMdH_y[:,1]+dMdH_z[:,2])/3.

        elif len(Field) == 3:
            Delta = deltaField*np.array(Field)/np.linalg.norm(Field)
            Mplus1 = self.magnetization(Temps, Field + Delta)
            Mminus1= self.magnetization(Temps, Field - Delta)
            Mplus2 = self.magnetization(Temps, Field + 2*Delta)
            Mminus2= self.magnetization(Temps, Field - 2*Delta)

            dMdH = (8*(Mplus1 - Mminus1) - (Mplus2 - Mminus2))/(12*deltaField)
            #dMdH = (Mplus1 - Mminus1)/(2*deltaField)

            return dMdH












### Same class, but in the LS basis

class LS_Ligands:
    """For doing point-charge calculations in LS basis"""
    def __init__(self,ion,latticeParams,ionPos,ligandPos, SpinOrbitCoupling):
        """Creates array of ligand bonds in cartesian coordinates.
        'ion' can either be the name of the ion or a list specifying L and S."""
        lp = latticeParams
        if len(lp) != 6:
            raise LookupError("latticeParams needs to have 6 components: a,b,c,alpha,beta,gamma")
        self.latt = lat.lattice(lp[0], lp[1], lp[2], lp[3], lp[4], lp[5])

        self.bonds = np.array([O - ionPos for O in ligandPos])
        self.bonds = self.latt.cartesian(self.bonds)
        self.bondlen = np.linalg.norm(self.bonds, axis=1)

        if isinstance(ion, str):
            self.ion = ion
            self.ionS = Jion[ion][0]
            self.ionL = Jion[ion][1]
        else:
            self.ionS = ion[0]
            self.ionL = ion[1]

        # Now, define the spin orbit coupling (so we don't have to re-define it 
        # every time we build the point charge model).
        Sx = LSOperator.Sx(self.ionL, self.ionS)
        Sy = LSOperator.Sy(self.ionL, self.ionS)
        Sz = LSOperator.Sz(self.ionL, self.ionS)
        Lx = LSOperator.Lx(self.ionL, self.ionS)
        Ly = LSOperator.Ly(self.ionL, self.ionS)
        Lz = LSOperator.Lz(self.ionL, self.ionS)

        self.H_SOC = Lx*Sx + Ly*Sy + Lz*Sz
        LdotS = self.H_SOC.O*1.0
        if np.sum(LdotS.imag) == 0: LdotS = LdotS.real
        self.H_SOC.O = SpinOrbitCoupling*LdotS

    def rotateLigands(self, oldaxis, newaxis):
        '''rotates the ligand bonds so that the new axis is in the direction of the old axis'''
        rotationAxis = np.cross(newaxis,oldaxis)
        rotationAngle = np.arccos(np.dot(newaxis,oldaxis)/(np.linalg.norm(newaxis)*np.linalg.norm(oldaxis)))
        self.bonds = np.array([self._rotateMatrix(b,rotationAxis,rotationAngle) for b in self.bonds])

    def rotateLigandsZ(self, oldaxis):
        '''rotates the ligand bonds around the z axis so that oldaxis 
        becomes the x axis'''
        zrotation = np.arctan(oldaxis[1]/oldaxis[0])
        self.bonds = np.array([self._rotateMatrix(b,np.array([0,0,1]),-zrotation) for b in self.bonds])


    def _rotateMatrix(self,matrixin,axis,angle):
        """taken from http://inside.mines.edu/fs_home/gmurray/ArbitraryAxisRotation/"""
        u, v, w = axis[0], axis[1], axis[2]
        norm = u**2 + v**2 + w**2
        
        rotmatrix = np.zeros((3,3))
        rotmatrix[0,0] = (u**2 +(v**2 + w**2)*np.cos(angle)) / norm
        rotmatrix[0,1] = (u*v*(1- np.cos(angle)) - w*np.sqrt(norm)*np.sin(angle)) / norm
        rotmatrix[0,2] = (u*w*(1- np.cos(angle)) + v*np.sqrt(norm)*np.sin(angle)) / norm
        rotmatrix[1,0] = (u*v*(1- np.cos(angle)) + w*np.sqrt(norm)*np.sin(angle)) / norm
        rotmatrix[1,1] = (v**2 +(u**2 + w**2)*np.cos(angle)) / norm
        rotmatrix[1,2] = (v*w*(1- np.cos(angle)) - u*np.sqrt(norm)*np.sin(angle)) / norm
        rotmatrix[2,0] = (u*w*(1- np.cos(angle)) - v*np.sqrt(norm)*np.sin(angle)) / norm
        rotmatrix[2,1] = (v*w*(1- np.cos(angle)) + u*np.sqrt(norm)*np.sin(angle)) / norm
        rotmatrix[2,2] = (w**2 +(v**2 + u**2)*np.cos(angle)) / norm

        # Simple matrix multiplication of matrixin is a vector
        if matrixin.size == 3:
            return np.dot(rotmatrix, matrixin)
        # R*m*R^T if matrixin is a matrix
        elif matrixin.size == 9:
            return np.dot(rotmatrix, np.dot(matrixin, rotmatrix.transpose() ))



    def PointChargeModel(self,  symequiv=None, LigandCharge=-2,IonCharge=1,
                        printB = True, suppressminusm = False):
        '''Create point charge model of the crystal fields of a rare-earth ion.
        Returns a CFLevels object with the hamiltonian defined.
        Define LigandCharge in units of e.'''

        # Lock suppressmm into whatever it was when PointChargeModel was first called.
        self.IonCharge = IonCharge
        try: self.suppressmm
        except AttributeError:
            self.suppressmm = suppressminusm

        if symequiv == None:
            charge = IonCharge*[LigandCharge]*len(self.bonds)
        else:
            charge = [0]*len(self.bonds)
            for i,se in enumerate(symequiv):
                charge[i] = IonCharge*LigandCharge[se]
        self.symequiv = symequiv
        
        ion=self.ion
        # ionS = Jion[ion][0]
        # ionL = Jion[ion][1]

        # # print factors used:
        # print "#---------------------------------------"
        # print "# Stevens Factors \tRadial Integrals (a_0)"
        # for n in range(2,8,2):
        #     print ' ', theta(ion,n), '\t ', RadialIntegral(ion,n)
        # print '#---------------------------------------\n'

        ahc = 1.43996e4  #Constant to get the energy in units of meV = alpha*hbar*c
        a0 = 0.52917721067    #Bohr radius in \AA

        H = np.zeros((int(2*self.ionL+1), int(2*self.ionL+1)),dtype = complex)
        self.B = []

        self.H_nocharge = [[]]
        if self.suppressmm == False:  nmrange = [[n,m] for n in range(2,8,2) for m in range(-n,n+1)]
        elif self.suppressmm == True:   nmrange = [[n,m] for n in range(2,8,2) for m in range(0,n+1)]
        #for n,m in [[n,m] for n in range(2,8,2) for m in range(-n,n+1)]:
        for n,m in nmrange:
            # 1)  Compute gamma
            gamma = 0
            for i in range(len(self.bonds)):

                gamma += 4*np.pi/(2*n+1)*charge[i] *\
                            TessHarm(n,m, self.bonds[i][0], self.bonds[i][1], self.bonds[i][2])/\
                            (self.bondlen[i]**(n+1))

            # 2)  Compute CEF parameter
            B = -gamma * ahc* a0**n * Constant(n,m) * RadialIntegral(ion,n) * LStheta(ion,n)
            if printB ==True: print('B_'+str(n),m,' = ',np.around(B,decimals=8))
            #print cef.StevensOp(ionJ,n,m)
            #self.H += np.around(B,decimals=15)*StevensOp(ionJ,n,m)
            H += B*StevensOp(self.ionL,n,m)
            self.B.append(B)
        self.B = np.array(self.B)

        # Convert Hamiltonian to full LS basis
        H_CEF_O = np.hstack(np.hstack(np.multiply.outer(H, np.identity(int(2*self.ionS+1)))))
        self.H_CEF = LSOperator(self.ionL, self.ionS)
        self.H_CEF.O = H_CEF_O

        #self.H = self.H_CEF + self.H_LS

        return LS_CFLevels.Hamiltonian(self.H_CEF, self.H_SOC, self.ionL, self.ionS)


    def TMPointChargeModel(self, RadialIntegrals,  halffilled=True, l=2,
                        symequiv=None, LigandCharge= -2, IonCharge=1,
                        printB = True, suppressminusm = False):
        ''' For transition metals:
        Create point charge model of the crystal fields of a rare-earth ion.
        Returns a CFLevels object with the hamiltonian defined.
        Define LigandCharge in units of e.'''

        self.IonCharge = IonCharge
        # Lock suppressmm into whatever it was when PointChargeModel was first called.
        try: self.suppressmm
        except AttributeError:
            self.suppressmm = suppressminusm

        if symequiv == None:
            charge = [LigandCharge]*len(self.bonds)
        else:
            charge = [0]*len(self.bonds)
            for i,se in enumerate(symequiv):
                charge[i] = LigandCharge[se]
        self.symequiv = symequiv

        # # print factors used:
        # print "#---------------------------------------"
        # print "# Stevens Factors \tRadial Integrals (a_0)"
        # for n in range(2,8,2):
        #     print ' ', theta(ion,n), '\t ', RadialIntegral(ion,n)
        # print '#---------------------------------------\n'

        ahc = 1.43996e4  #Constant to get the energy in units of meV = alpha*hbar*c
        a0 = 0.52917721067    #Bohr radius in \AA

        H = np.zeros((int(2*self.ionL+1), int(2*self.ionL+1)),dtype = complex)
        self.B = []

        TM_LStheta = {2: PFalpha(self.ionL,self.ionS,l,halffilled), 
                    4: PFbeta(self.ionL,self.ionS,l,halffilled)}

        self.H_nocharge = [[]]
        if self.suppressmm == False:  nmrange = [[n,m] for n in range(2,6,2) for m in range(-n,n+1)]
        elif self.suppressmm == True:   nmrange = [[n,m] for n in range(2,6,2) for m in range(0,n+1)]
        #for n,m in [[n,m] for n in range(2,8,2) for m in range(-n,n+1)]:
        for n,m in nmrange:
            # 1)  Compute gamma
            gamma = 0
            for i in range(len(self.bonds)):

                gamma += 4*np.pi/(2*n+1)*charge[i] *\
                            TessHarm(n,m, self.bonds[i][0], self.bonds[i][1], self.bonds[i][2])/\
                            (self.bondlen[i]**(n+1))

            # 2)  Compute CEF parameter
            B = -gamma * ahc* a0**n * Constant(n,m) * RadialIntegrals[n] * TM_LStheta[n]
            if printB ==True: print('B_'+str(n),m,' = ',np.around(B,decimals=8))
            #print cef.StevensOp(ionJ,n,m)
            #self.H += np.around(B,decimals=15)*StevensOp(ionJ,n,m)
            H += B*StevensOp(self.ionL,n,m)
            self.B.append(B)
        self.B = np.array(self.B)

        # Convert Hamiltonian to full LS basis
        H_CEF_O = np.hstack(np.hstack(np.multiply.outer(H, np.identity(int(2*self.ionS+1)))))
        self.H_CEF = LSOperator(self.ionL, self.ionS)
        self.H_CEF.O = H_CEF_O

        #self.H = self.H_CEF + self.H_LS

        return LS_CFLevels.Hamiltonian(self.H_CEF, self.H_SOC, self.ionL, self.ionS)



    def ReMakePointChargeModel(newcharges):
        # make charges into list
        if self.symequiv == None:
            charge = self.IonCharge*[newcharges]*len(self.bonds)
        else:
            charge = [0]*len(self.bonds)
            for i,se in enumerate(self.symequiv):
                charge[i] = self.IonCharge*newcharges[se]

    def FitChargesNeutrons(self, chisqfunc, fitargs, method='Powell', **kwargs):
        '''fits neutron data'''

        # Define function to be fit
        fun, p0, resfunc = makeFitFunction(chisqfunc, fitargs, **dict(kwargs, LigandsObject=self) )

        print('\tFitting...')
        ############## Fit, using error function  #####################
        p_best = optimize.minimize(fun, p0, method=method)
        #p_best = optimize.minimize(fun, p0, method='Nelder-Mead')
        ###############################################################

        initialChisq, finalChisq = fun(p0), fun(p_best.x)

        # split back into values
        finalvals = resfunc(p_best.x)
        finalCharges = finalvals['LigandCharge']

        # Print results
        print("\n#*********************************")
        print("# Final Stevens Operator Values")
        newH = self.PointChargeModel(kwargs['symequiv'], finalCharges, printB=True)
        newH.diagonalize()
        print("\nFinal Charges: ", finalCharges)
        print('Final EigenValues: ', np.around(np.sort(newH.eigenvalues.real),3))

        return newH, finalvals






#######################################################################################


class LS_CFLevels:
    """For calculating and fitting crystal field levels for an ion"""
    def __init__(self, StevensOperators, Parameters, L,S, SpinOrbitCoupling):
        """add Stevens operators to make a single hamiltonian matrix.
        Assumes that the Hamiltonian has been built with the LS_StevensOp function"""
        self.H_CEF = LSOperator(L, S)
        HcefJ = np.sum([a*b for a,b in zip(StevensOperators, Parameters)], axis=0)
        #self.H_CEF.O = np.hstack(np.hstack(np.multiply.outer(HcefJ, np.identity(int(2*S+1)))))
        self.H_CEF.O = HcefJ
        self.O = StevensOperators  #save these for a fit
        self.Ci = Parameters
        self.S = S
        self.L = L

        # Define spin orbit coupling Hamiltonian
        Sx = LSOperator.Sx(L, S)
        Sy = LSOperator.Sy(L, S)
        Sz = LSOperator.Sz(L, S)
        Lx = LSOperator.Lx(L, S)
        Ly = LSOperator.Ly(L, S)
        Lz = LSOperator.Lz(L, S)

        self.H_SOC = Lx*Sx + Ly*Sy + Lz*Sz
        LdotS = self.H_SOC.O*1.0
        if np.sum(LdotS.imag) == 0: LdotS = LdotS.real
        self.H_SOC.O = SpinOrbitCoupling*LdotS
        self.spinorbitcoupling = SpinOrbitCoupling
        #print(self.spinorbitcoupling)

        # Define J operators for use later
        g0 = 2.002319
        self.Jx = Sx + Lx
        self.Jy = Sy + Ly
        self.Jz = Sz + Lz
        self.Jxg0 = g0*Sx + Lx
        self.Jyg0 = g0*Sy + Ly
        self.Jzg0 = g0*Sz + Lz

    @classmethod
    def Hamiltonian(cls, CEF_Hamil, SOC_Hamil, L, S):
        newcls = cls([0,0],[0,0], L, S, 0) # Create empty class so we can just define Hamiltonian
        newcls.H_CEF = CEF_Hamil  # Crystal electric fields
        newcls.H_SOC = SOC_Hamil  # Spin Orbit Coupling
        return newcls

    ### DEPRECIATED 1/3/20: eig_banded is faster and less susceptible to roundoff errors
    # def diagonalize(self, CEF_Hamiltonian=None):
    #     """A Hamiltonian can be passed to the function (used for data fits)
    #     or the initially defined hamiltonian is used."""
    #     if CEF_Hamiltonian is None:
    #         CEF_Hamiltonian = self.H_CEF.O
    #     else:
    #         self.H_CEF.O = CEF_Hamiltonian
    #     diagonalH = LA.eigh(CEF_Hamiltonian + self.H_SOC.O)

    #     self.eigenvaluesNoNorm = diagonalH[0]
    #     self.eigenvalues = diagonalH[0] - np.amin(diagonalH[0])
    #     self.eigenvectors = diagonalH[1].T
    #     # set very small values to zero
    #     tol = 1e-15
    #     self.eigenvalues[abs(self.eigenvalues) < tol] = 0.0
    #     self.eigenvectors[abs(self.eigenvectors) < tol] = 0.0


    def diagonalize(self, CEF_Hamiltonian=None):
        '''same as above, but using the Scipy eig_banded function'''
        if CEF_Hamiltonian is None:
            CEF_Hamiltonian = self.H_CEF.O
        else:
            self.H_CEF.O = CEF_Hamiltonian

        bands = self._findbands(CEF_Hamiltonian + self.H_SOC.O)
        diagonalH = LA.eig_banded(bands, lower=True)

        #self.eigenvaluesNoNorm = diagonalH[0]
        self.eigenvalues = diagonalH[0] - np.amin(diagonalH[0])
        self.eigenvectors = diagonalH[1].T
        # set very small values to zero
        tol = 1e-15
        self.eigenvalues[abs(self.eigenvalues) < tol] = 0.0
        self.eigenvectors[abs(self.eigenvectors) < tol] = 0.0

    def _findbands(self, matrix):
        '''used in the diagonalize_banded function'''
        diags = np.zeros((len(matrix),len(matrix)))
        for i in range(len(matrix)):
            diag = matrix.diagonal(i)
            if i == 0:
                diags[i] = diag
            else:
                diags[i][:-i] = diag
            if np.count_nonzero(np.around(diag,10)) > 0:
                nonzerobands = i
        return diags[:nonzerobands+1]



    def neutronSpectrum(self, Earray, Temp, Ei, ResFunc, gamma = 0):
        maxtransition = 12 # because we can't see the others

        # make angular momentum ket object
        eigenkets = [Ket(ei) for ei in self.eigenvectors[:maxtransition]]
        intensity = np.zeros(len(Earray))

        # for population factor weights
        beta = 1/(8.61733e-2*Temp)  # Boltzmann constant is in meV/K
        Z = sum([np.exp(-beta*en) for en in self.eigenvalues])

        for i, ket_i in enumerate(eigenkets):
            # compute population factor
            pn = np.exp(-beta *self.eigenvalues[i])/Z

            for j, ket_j in enumerate(eigenkets):
                # compute amplitude
                mJn = self._transition(ket_i,ket_j)
                deltaE = self.eigenvalues[j] - self.eigenvalues[i]
                GausWidth = ResFunc(deltaE)  #peak width due to instrument resolution
                intensity += ((pn * mJn * self._voigt(x=Earray, x0=deltaE, alpha=GausWidth, 
                                                    gamma=gamma)).real).astype('float64')
                #intensity += ((pn * mJn * self._lorentzian(Earray, deltaE, Width)).real).astype('float64')

        ## List comprehension: turns out this way was slower.
        # intensity = np.sum([
        #     np.exp(-beta *self.eigenvalues[i])/Z *\
        #     self._transition(eigenkets[i], eigenkets[j]) *\
        #     self._voigt(x=Earray, x0=(self.eigenvalues[j] - self.eigenvalues[i]), 
        #         alpha=ResFunc(self.eigenvalues[j] - self.eigenvalues[i]), gamma=gamma)
        #     for i in range(len(eigenkets)) for j in range(len(eigenkets))
        #     ], axis = 0)

        kpoverk = np.sqrt((Ei - Earray)/Ei) #k'/k = sqrt(E'/E)
        return intensity * kpoverk

    def neutronSpectrum2D(self, Earray, Qarray, Temp, Ei, ResFunc, gamma, DebyeWaller, Ion):
        intensity1D = self.neutronSpectrum(Earray, Temp, Ei, ResFunc,  gamma)

        # Scale by Debye-Waller Factor
        DWF = np.exp(1./3. * Qarray**2 * DebyeWaller**2)
        # Scale by form factor
        FormFactor = RE_FormFactor(Qarray,Ion)
        return np.outer(intensity1D, DWF*FormFactor)


    def _transition(self,ket1,ket2):
        """Computes \sum_a |<|J_a|>|^2 = \sum_a |<|L_a + S_a|>|^2"""
        ax = np.dot(np.conjugate(ket1.ket),np.dot(self.Jx.O,ket2.ket)) *\
                np.dot(np.conjugate(ket2.ket),np.dot(self.Jx.O,ket1.ket))
        ay = np.dot(np.conjugate(ket1.ket),np.dot(self.Jy.O,ket2.ket)) *\
                np.dot(np.conjugate(ket2.ket),np.dot(self.Jy.O,ket1.ket))
        az = np.dot(np.conjugate(ket1.ket),np.dot(self.Jz.O,ket2.ket)) *\
                np.dot(np.conjugate(ket2.ket),np.dot(self.Jz.O,ket1.ket))

        # eliminate tiny values
        ax, ay, az = np.around(ax, 10), np.around(ay, 10), np.around(az, 10)
        if (ax + ay + az).imag == 0:
            return ((ax + ay + az).real).astype(float)
        else:
            print(ax, ay, az)
            raise ValueError("non-real amplitude. Error somewhere.")
            
    def _lorentzian(self, x, x0, gamma):
        return 1/np.pi * (0.5*gamma)/((x-x0)**2 + (0.5*gamma)**2)

    def _voigt(self, x, x0, alpha, gamma):
        """ Return the Voigt line shape at x with Lorentzian component FWHM gamma
        and Gaussian component FWHM alpha."""
        sigma = (0.5*alpha) / np.sqrt(2 * np.log(2))
        return np.real(wofz(((x-x0) + 1j*(0.5*gamma))/sigma/np.sqrt(2))) / sigma\
                                                            /np.sqrt(2*np.pi)

    def _Re(self,value):
        thresh = 1e-9
        if np.size(value) == 1 & isinstance(value, complex):
            if value.imag <= thresh:
                return (value.real).astype(float)
            else: 
                return value
        else:
            if np.all(value.imag < thresh):
                return (value.real)
            else: return value

    def printEigenvectors(self):
        '''prints eigenvectors and eigenvalues in a matrix'''
        print('\n Eigenvalues \t Eigenvectors')
        print('\t\t'+'-------'*(len(self.eigenvalues)+1))
        sortinds = self.eigenvalues.argsort()
        sortEVal= np.around(self.eigenvalues[sortinds],5)
        sortEVec= np.around(self.eigenvectors[sortinds],3)
        for i in range(len(sortinds)):
            print(format(self._Re(sortEVal[i]), '.5f'),'\t| ', self._Re(sortEVec[i]),' |')
        print('\t\t'+'-------'*(len(self.eigenvalues)+1) + '\n')

    def gsExpectation(self):
        """Prints <J_x>, <J_y>, and <J_z> for the ground state"""
        zeroinds = np.where(np.around(self.eigenvalues,7)==0)
        gsEVec = self.eigenvectors[zeroinds]
        print('\t Ground State Expectation Values:')
        for ev in gsEVec:
            jjxx = self._Re(np.dot(ev,np.dot(self.Jxg0.O,ev)))
            jjyy = self._Re(np.dot(ev,np.dot(self.Jyg0.O,ev)))
            jjzz = self._Re(np.dot(ev,np.dot(self.Jzg0.O,ev)))
            print('  <J_x> =',jjxx,'\t<J_y> =',jjyy,'\t<J_z> =',jjzz)
        print(' ')


    def magnetization(self, Temp, Field):
        '''field should be a 3-component vector. Temps may be an array.'''
        if len(Field) != 3: 
            raise TypeError("Field needs to be 3-component vector")

        # A) Define magnetic Hamiltonian
        muB = 5.7883818012e-2  # meV/T
        #mu0 = np.pi*4e-7       # T*m/A
        JdotB = muB*(Field[0]*self.Jxg0 + Field[1]*self.Jyg0 + Field[2]*self.Jzg0)

        # B) Diagonalize full Hamiltonian
        FieldHam = self.H_CEF.O + self.H_SOC.O + JdotB.O
        diagonalH = LA.eigh(FieldHam)

        minE = np.amin(diagonalH[0])
        evals = diagonalH[0] - minE
        evecs = diagonalH[1].T
        # These ARE actual eigenvalues.

        # C) Compute expectation value along field
        JexpVals = np.zeros((len(evals),3))
        for i, ev in enumerate(evecs):
            #print np.real(np.dot(ev, np.dot( self.O ,ev))), diagonalH[0][i]
            #print np.real(np.dot( FieldHam ,ev)), np.real(diagonalH[0][i]*ev)
            JexpVals[i] =[np.real(np.dot(np.conjugate(ev), np.dot( self.Jxg0.O ,ev))),
                          np.real(np.dot(np.conjugate(ev), np.dot( self.Jyg0.O ,ev))),
                          np.real(np.dot(np.conjugate(ev), np.dot( self.Jzg0.O ,ev)))]
        k_B = 8.6173303e-2  # meV/K

        if (isinstance(Temp, int) or isinstance(Temp, float)):
            Zz = np.sum(np.exp(-evals/(k_B*Temp)))
            JexpVal = np.dot(np.exp(-evals/(k_B*Temp)),JexpVals)/Zz
            return np.real(JexpVal)
        else:
            expvals, temps = np.meshgrid(evals, Temp)
            ZZ = np.sum(np.exp(-expvals/temps/k_B), axis=1)
            JexpValList = np.repeat(JexpVals.reshape((1,)+JexpVals.shape), len(Temp), axis=0)
            JexpValList = np.sum(np.exp(-expvals/temps/k_B)*\
                                np.transpose(JexpValList, axes=[2,0,1]), axis=2) / ZZ
            # if np.isnan(JexpValList).any():
            #     print -expvals[0]/temps[0]/k_B
            #     print np.exp(-expvals/temps/k_B)[0]
            #     raise ValueError('Nan in result!')
            return np.nan_to_num(JexpValList.T)


    def susceptibility(self, Temps, Field, deltaField):
        '''Computes susceptibility numerically with a numerical derivative.
        deltaField needs to be a scalar value.'''
        if not isinstance(deltaField, float):
            raise TypeError("Deltafield needs to be a scalar")

        if isinstance(Field, float):
            # Assume we are computing a powder average
            VecField = Field * np.array([1,0,0])
            Delta = deltaField*np.array(VecField)/Field
            Mplus1 = self.magnetization(Temps, VecField + Delta)
            Mminus1= self.magnetization(Temps, VecField - Delta)
            Mplus2 = self.magnetization(Temps, VecField + 2*Delta)
            Mminus2= self.magnetization(Temps, VecField - 2*Delta)

            dMdH_x = (8*(Mplus1 - Mminus1) - (Mplus2 - Mminus2))/(12*deltaField)

            VecField = Field * np.array([0,1,0])
            Delta = deltaField*np.array(VecField)/Field
            Mplus1 = self.magnetization(Temps, VecField + Delta)
            Mminus1= self.magnetization(Temps, VecField - Delta)
            Mplus2 = self.magnetization(Temps, VecField + 2*Delta)
            Mminus2= self.magnetization(Temps, VecField - 2*Delta)

            dMdH_y = (8*(Mplus1 - Mminus1) - (Mplus2 - Mminus2))/(12*deltaField)

            VecField = Field * np.array([0,0,1])
            Delta = deltaField*np.array(VecField)/Field
            Mplus1 = self.magnetization(Temps, VecField + Delta)
            Mminus1= self.magnetization(Temps, VecField - Delta)
            Mplus2 = self.magnetization(Temps, VecField + 2*Delta)
            Mminus2= self.magnetization(Temps, VecField - 2*Delta)

            dMdH_z = (8*(Mplus1 - Mminus1) - (Mplus2 - Mminus2))/(12*deltaField)

            return (dMdH_x[:,0]+dMdH_y[:,1]+dMdH_z[:,2])/3.

        elif len(Field) == 3:
            Delta = deltaField*np.array(Field)/np.linalg.norm(Field)
            Mplus1 = self.magnetization(Temps, Field + Delta)
            Mminus1= self.magnetization(Temps, Field - Delta)
            Mplus2 = self.magnetization(Temps, Field + 2*Delta)
            Mminus2= self.magnetization(Temps, Field - 2*Delta)

            dMdH = (8*(Mplus1 - Mminus1) - (Mplus2 - Mminus2))/(12*deltaField)
            #dMdH = (Mplus1 - Mminus1)/(2*deltaField)

            return dMdH

    def susceptibilityDeriv(self, Temps, Field, deltaField):
        """Computes susceptibility with 
        $\chi = \frac{\partial M}{\partial H} = - \frac{\partial^2 F}{\partial H^2}$"""
        if not isinstance(deltaField, float):
            raise TypeError("Deltafield needs to be a scalar")

        if (isinstance(Field, float) or isinstance(Field, int)):
            # Assume we are computing a powder average
            VecField = Field * np.array([1,0,0])
            Delta = deltaField*np.array(VecField)/Field
            Mplus1 = self.magnetizationDeriv(Temps, VecField + Delta, deltaField)
            Mminus1= self.magnetizationDeriv(Temps, VecField - Delta, deltaField)
            Mplus2 = self.magnetizationDeriv(Temps, VecField + 2*Delta, deltaField)
            Mminus2= self.magnetizationDeriv(Temps, VecField - 2*Delta, deltaField)

            dMdH_x = (8*(Mplus1 - Mminus1) - (Mplus2 - Mminus2))/(12*deltaField)

            VecField = Field * np.array([0,1,0])
            Delta = deltaField*np.array(VecField)/Field
            Mplus1 = self.magnetizationDeriv(Temps, VecField + Delta, deltaField)
            Mminus1= self.magnetizationDeriv(Temps, VecField - Delta, deltaField)
            Mplus2 = self.magnetizationDeriv(Temps, VecField + 2*Delta, deltaField)
            Mminus2= self.magnetizationDeriv(Temps, VecField - 2*Delta, deltaField)

            dMdH_y = (8*(Mplus1 - Mminus1) - (Mplus2 - Mminus2))/(12*deltaField)

            VecField = Field * np.array([0,0,1])
            Delta = deltaField*np.array(VecField)/Field
            Mplus1 = self.magnetizationDeriv(Temps, VecField + Delta, deltaField)
            Mminus1= self.magnetizationDeriv(Temps, VecField - Delta, deltaField)
            Mplus2 = self.magnetizationDeriv(Temps, VecField + 2*Delta, deltaField)
            Mminus2= self.magnetizationDeriv(Temps, VecField - 2*Delta, deltaField)

            dMdH_z = (8*(Mplus1 - Mminus1) - (Mplus2 - Mminus2))/(12*deltaField)

            return (dMdH_x[:,0]+dMdH_y[:,1]+dMdH_z[:,2])/3.

        else: return 0

    def magnetizationDeriv(self, Temp, Field, deltaField):
        '''field should be a 3-component vector. Temps may be an array.
        Field should be in Tesla, and Temp in Kelvin. Returns magnetization in mu_B'''
        if len(Field) != 3: 
            raise TypeError("Field needs to be 3-component vector")

        # A) Define magnetic Hamiltonian
        muB = 5.7883818012e-2  # meV/T
        #mu0 = np.pi*4e-7       # T*m/A
        JdotB_0 = muB*(Field[0]*self.Jxg0 + Field[1]*self.Jyg0 + Field[2]*self.Jzg0)

        Delta = deltaField*np.array([1,0,0])
        FieldPD = Field + Delta
        JdotB_p1x = muB*(FieldPD[0]*self.Jxg0 + FieldPD[1]*self.Jyg0 + FieldPD[2]*self.Jzg0)
        FieldPD = Field - Delta
        JdotB_m1x = muB*(FieldPD[0]*self.Jxg0 + FieldPD[1]*self.Jyg0 + FieldPD[2]*self.Jzg0)

        Delta = deltaField*np.array([0,1,0])
        FieldPD = Field + Delta
        JdotB_p1y = muB*(FieldPD[0]*self.Jxg0 + FieldPD[1]*self.Jyg0 + FieldPD[2]*self.Jzg0)
        FieldPD = Field - Delta
        JdotB_m1y = muB*(FieldPD[0]*self.Jxg0 + FieldPD[1]*self.Jyg0 + FieldPD[2]*self.Jzg0)

        Delta = deltaField*np.array([0,0,1])
        FieldPD = Field + Delta
        JdotB_p1z = muB*(FieldPD[0]*self.Jxg0 + FieldPD[1]*self.Jyg0 + FieldPD[2]*self.Jzg0)
        FieldPD = Field - Delta
        JdotB_m1z = muB*(FieldPD[0]*self.Jxg0 + FieldPD[1]*self.Jyg0 + FieldPD[2]*self.Jzg0)

        # B) Diagonalize full Hamiltonian
        # first do the Delta=0 field:
        FieldHam = self.H_CEF.O + self.H_SOC.O + JdotB_0.O
        diagonalH = LA.eigh(FieldHam)

        minE = np.amin(diagonalH[0])
        evals = diagonalH[0] - minE

        # Now do the Delta =/= 0:
        Evals_pm = []
        for JdotB in [JdotB_p1x, JdotB_m1x, JdotB_p1y, JdotB_m1y, JdotB_p1z, JdotB_m1z]:   
            FieldHam = self.H_CEF.O + self.H_SOC.O + JdotB.O
            diagonalH = LA.eigh(FieldHam)

            
            Evals_pm.append(diagonalH[0])
        minE = np.amin(Evals_pm)
        Evals_pm -= minE

        Evals_pm = np.array(Evals_pm).T

        # C) Compute derivative of energy w.r.t. field:
        Mderivs = np.zeros((len(Evals_pm),3))
        for i, ev in enumerate(Evals_pm):
            Mderivs[i]=[(ev[0] - ev[1])/(2*deltaField),  
                        (ev[2] - ev[3])/(2*deltaField),  
                        (ev[4] - ev[5])/(2*deltaField)]
        k_B = 8.6173303e-2  # meV/K
        #print(Mderivs)

        if (isinstance(Temp, int) or isinstance(Temp, float)):
            Zz = np.sum(np.exp(-evals/(k_B*Temp)))
            BoltzmannWeights = np.exp(-evals/(k_B*Temp))/Zz
            return np.dot(BoltzmannWeights,Mderivs)/muB    #divide by muB to convert from meV/T
            
        else:
            expvals, temps = np.meshgrid(evals, Temp)
            ZZ = np.sum(np.exp(-expvals/temps/k_B), axis=1)
            MagList = np.repeat(Mderivs.reshape((1,)+Mderivs.shape), len(Temp), axis=0)
            MagList = np.sum(np.exp(-expvals/temps/k_B)*\
                                np.transpose(MagList, axes=[2,0,1]), axis=2) / ZZ
            # if np.isnan(JexpValList).any():
            #     print -expvals[0]/temps[0]/k_B
            #     print np.exp(-expvals/temps/k_B)[0]
            #     raise ValueError('Nan in result!')
            return np.nan_to_num(MagList.T) / muB




    def LplusS_expval(self, Temp, Field, deltaField):
        '''computes average of L + S. Used for g tensor calculation. This is 
        the same as magnetization but with L+S, not L+'''
        if len(Field) != 3: 
            raise TypeError("Field needs to be 3-component vector")

        # A) Define magnetic Hamiltonian
        muB = 5.7883818012e-2  # meV/T
        #mu0 = np.pi*4e-7       # T*m/A
        JdotB_0 = muB*(Field[0]*self.Jxg0 + Field[1]*self.Jyg0 + Field[2]*self.Jzg0)
        # print(JdotB_0)

        Delta = deltaField*np.array([1,0,0])
        FieldPD = Field + Delta
        JdotB_p1x = muB*(FieldPD[0]*self.Jxg0 + FieldPD[1]*self.Jyg0 + FieldPD[2]*self.Jzg0)
        FieldPD = Field - Delta
        JdotB_m1x = muB*(FieldPD[0]*self.Jxg0 + FieldPD[1]*self.Jyg0 + FieldPD[2]*self.Jzg0)

        Delta = deltaField*np.array([0,1,0])
        FieldPD = Field + Delta
        JdotB_p1y = muB*(FieldPD[0]*self.Jxg0 + FieldPD[1]*self.Jyg0 + FieldPD[2]*self.Jzg0)
        FieldPD = Field - Delta
        JdotB_m1y = muB*(FieldPD[0]*self.Jxg0 + FieldPD[1]*self.Jyg0 + FieldPD[2]*self.Jzg0)

        Delta = deltaField*np.array([0,0,1])
        FieldPD = Field + Delta
        JdotB_p1z = muB*(FieldPD[0]*self.Jxg0 + FieldPD[1]*self.Jyg0 + FieldPD[2]*self.Jzg0)
        FieldPD = Field - Delta
        JdotB_m1z = muB*(FieldPD[0]*self.Jxg0 + FieldPD[1]*self.Jyg0 + FieldPD[2]*self.Jzg0)

        # B) Diagonalize full Hamiltonian
        # first do the Delta=0 field:
        FieldHam = self.H_CEF.O + self.H_SOC.O + JdotB_0.O
        diagonalH = LA.eigh(FieldHam)

        minE = np.amin(diagonalH[0])
        evals = diagonalH[0] - minE

        # Compute LdotS for every eigenvector
        LdotS_vals = []
        for ev in diagonalH[1].T:
            jjxx = self._Re(np.dot(np.conjugate(ev),np.dot(self.Jx.O,ev)))
            jjyy = self._Re(np.dot(np.conjugate(ev),np.dot(self.Jy.O,ev)))
            jjzz = self._Re(np.dot(np.conjugate(ev),np.dot(self.Jz.O,ev)))

            LdotS_vals.append([jjxx, jjyy, jjzz])
        LdotS_vals = np.array(LdotS_vals)

        # Now do the Delta =/= 0:
        Evals_pm = []
        
        for JdotB in [JdotB_p1x, JdotB_m1x, JdotB_p1y, JdotB_m1y, JdotB_p1z, JdotB_m1z]:   
            FieldHam = self.H_CEF.O + self.H_SOC.O + JdotB.O
            diagonalH = LA.eigh(FieldHam)
            
            Evals_pm.append(diagonalH[0])
        minE = np.amin(Evals_pm)
        Evals_pm -= minE

        Evals_pm = np.array(Evals_pm).T

        # C) Compute derivative of energy w.r.t. field:
        Mderivs = np.zeros((len(Evals_pm),3))
        for i, ev in enumerate(Evals_pm):
            Mderivs[i]=[(ev[0] - ev[1])/(2*deltaField),  
                        (ev[2] - ev[3])/(2*deltaField),  
                        (ev[4] - ev[5])/(2*deltaField)]
        k_B = 8.6173303e-2  # meV/K
        #print(Mderivs)

        if (isinstance(Temp, int) or isinstance(Temp, float)):
            Zz = np.sum(np.exp(-evals/(k_B*Temp)))
            BoltzmannWeights = np.exp(-evals/(k_B*Temp))/Zz
            Magnetization = np.dot(BoltzmannWeights,Mderivs)/muB  
                            #divide by muB to convert from meV/T  
            WeightedLdotS = np.dot(BoltzmannWeights, LdotS_vals)

            return Magnetization, WeightedLdotS
                        
        else: print('Temp must be float.')



    def gtensor(self):
        '''Returns g tensor computed numerically'''
        def eliminateimag(number):
            num = np.around(number, 10)
            if num.imag == 0:
                return (num.real).astype(float)
            else:
                return number

        zeroinds = np.where(np.around(self.eigenvalues,4)==0)
        gsEVec = self.eigenvectors[zeroinds]
        vv1 = gsEVec[0]
        vv2 = gsEVec[1]
        Jx, Jy, Jz = self.Jxg0.O, self.Jyg0.O, self.Jzg0.O
        # jz01 = eliminateimag( np.dot(vv1,np.dot(Jz,vv2)) )
        # jz10 = eliminateimag( np.dot(vv2,np.dot(Jz,vv1)) )
        # jz00 = eliminateimag( np.dot(vv1,np.dot(Jz,vv1)) )
        # jz11 = eliminateimag( np.dot(vv2,np.dot(Jz,vv2)) )
        
        
        # jx01 = eliminateimag( np.dot(vv1,np.dot(Jx,vv2)) )
        # jx10 = eliminateimag( np.dot(vv2,np.dot(Jx,vv1)) )
        # jx00 = eliminateimag( np.dot(vv1,np.dot(Jx,vv1)) )
        # jx11 = eliminateimag( np.dot(vv2,np.dot(Jx,vv2)) )
        
        # jy01 = eliminateimag( np.dot(vv1,np.dot(Jy,vv2)) )
        # jy10 = eliminateimag( np.dot(vv2,np.dot(Jy,vv1)) )
        # jy00 = eliminateimag( np.dot(vv1,np.dot(Jy,vv1)) )
        # jy11 = eliminateimag( np.dot(vv2,np.dot(Jy,vv2)) )
        
        # gg = 2*np.array([[np.abs(np.real(jx01)), np.imag(jx01), jx00],
        #                  [np.real(jy01), np.imag(jy01), jy00],
        #                  [np.real(jz01), np.imag(jz01), np.abs(jz00)]])

        jz01 = np.dot(vv1,np.dot(Jz,vv2)) 
        jz10 = np.dot(vv2,np.dot(Jz,vv1))
        jz00 = np.dot(vv1,np.dot(Jz,vv1))
        jz11 = np.dot(vv2,np.dot(Jz,vv2))
        
        
        jx01 = np.dot(vv1,np.dot(Jx,vv2))
        jx10 = np.dot(vv2,np.dot(Jx,vv1))
        jx00 = np.dot(vv1,np.dot(Jx,vv1))
        jx11 = np.dot(vv2,np.dot(Jx,vv2))
        
        jy01 = np.dot(vv1,np.dot(Jy,vv2))
        jy10 = np.dot(vv2,np.dot(Jy,vv1))
        jy00 = np.dot(vv1,np.dot(Jy,vv1))
        jy11 = np.dot(vv2,np.dot(Jy,vv2))
        
        gg = 2*np.array([[np.abs(np.real(jx01)), np.imag(jx01), jx00],
                         [np.real(jy01), np.imag(jy01), jy00],
                         [np.real(jz01), np.imag(jz01), np.abs(jz00)]])

        return gg

    # def gtensor(self, spinorbitcoupling, halffilled=True):
    #     '''Returns g tensor computed numerically via perturbation theory'''

    #     g0 = 2.002319
    #     gtens = np.zeros((3,3)) + np.identity(3)*g0

    #     if halffilled: hff = -1
    #     else:  hff = 1
    #     zeta = spinorbitcoupling*2*self.S*hff

    #     Lx = LSOperator.Lx(self.L, self.S)
    #     Ly = LSOperator.Ly(self.L, self.S)
    #     Lz = LSOperator.Lz(self.L, self.S)

    #     ev0 = self.eigenvectors[1]
    #     EE0 = self.eigenvalues[1]
    #     for i, Li in enumerate([Lx, Ly, Lz]):
    #         for j, Lj in enumerate([Lx, Ly, Lz]):
    #             for k, ev in enumerate(self.eigenvectors):
    #                 if self.eigenvalues[k] != EE0:
    #                     jj1 = np.dot(np.conjugate(ev0),np.dot(Li.O,ev))
    #                     jj2 = np.dot(np.conjugate(ev),np.dot(Lj.O,ev0))
    #                     #print(jj1*jj2)
    #                     gtens[i,j] -= 2*zeta*jj1*jj2/(self.eigenvalues[k]-EE0)
    #                     print(2*zeta*jj1*jj2/(self.eigenvalues[k]-EE0))
    #                     print(gtens[i,j])
    #                 else: pass

    #     return gtens


    def fitdata(self, chisqfunc, fitargs, method='Powell', **kwargs):
        '''fits data to CEF parameters'''

        initialChisq = chisqfunc(self, **kwargs)
        print('Initial err=', initialChisq, '\n')

        # define parameters
        if len(self.Ci) != len(kwargs['coeff']):
            raise ValueError('coeff needs to have the same length as self.Ci')

        # Define function to be fit
        fun, p0, resfunc = makeFitFunction(chisqfunc, fitargs, **dict(kwargs, CFLevelsObject=self) )


        ############## Fit, using error function  #####################
        p_best = optimize.minimize(fun, p0, method=method)
        ###############################################################

        print(fun(p_best.x))
        print(chisqfunc(self, **kwargs))
        finalChisq = fun(p_best.x)
        print('\rInitial err =', initialChisq, '\tFinal err =', finalChisq)
        
        result = resfunc(p_best.x)
        #print '\nFinal values: ', result
        result['Chisq'] = finalChisq
        return result


    def testEigenvectors(self):
        """Tests if eigenvectors are really eigenvectors"""
        print('testing eigenvectors... (look for large values)')
        for i in range(len(self.eigenvalues)):
            print(np.around(
                np.dot(self.H_SOC.O+self.H_CEF.O,self.eigenvectors[i]) -\
                 self.eigenvectors[i]*self.eigenvaluesNoNorm[i],
                10))

        # print('\n Sum rule (two values should be equal):')
        # TotalTransition = 0
        # for i, ev in enumerate(self.eigenvectors):
        #     TotalTransition += self._transition(Ket(self.eigenvectors[1]),Ket(ev))
        # print(TotalTransition, '  ', self.S*(self.J+1))


    def printLaTexEigenvectors(self):
        '''prints eigenvectors and eigenvalues in the output that Latex can read'''
        # Define S array
        if (self.S*2) %2 ==0:
            Sarray = [str(int(kk)) for kk in 
                                    np.arange(-self.S,self.S+1)]
        else:
            Sarray = ['-\\frac{'+str(abs(kk))+'}{2}' if kk <0
                            else '$\\frac{'+str(abs(kk))+'}{2}'
                            for kk in np.arange(-self.S*2,self.S*2+2,2)]

        # Define L array
        Larray = [str(int(kk)) for kk in  np.arange(-self.L,self.L+1)]

        # Define Ket names
        KetNames = ['$|$'+LL+','+SS+'$\\rangle$' for LL in Larray  for SS in Sarray]

        # Print everything out
        print('\\begin{table*}\n\\begin{landscape}\n\\centering\n'+
            '\\caption{Eigenvectors and Eigenvalues... $|L,S\\rangle$}')
        print('\\begin{ruledtabular}')
        numev = len(self.eigenvalues)
        print('\\begin{tabular}{c|'+'c'*numev+'}')
        print('E (meV) &'+' & '.join(KetNames)
                +' \\tabularnewline\n \\hline ')
        sortinds = self.eigenvalues.argsort()
        sortEVal= np.around(self.eigenvalues[sortinds],2)
        sortEVec= np.around(self.eigenvectors[sortinds],3)
        for i in range(len(sortinds)):
            print(format(self._Re(sortEVal[i]), '.3f'),'&', 
                ' & '.join([str(eevv) for eevv in self._Re(sortEVec[i])]), '\\tabularnewline')
        print('\\end{tabular}\\end{ruledtabular}')
        print('\\label{flo:Eigenvectors}\n\\end{landscape}\n\\end{table*}')











def backgroundfunction(xdata, bgpoints):
    """Linear interpolation"""
    bgp=np.array(bgpoints)
    if len(bgp)==2:
        P1x, P1y, P2x, P2y = bgp[0,0], bgp[0,1], bgp[1,0], bgp[1,1], 
        a = (P2y - P1y)/(P2x - P1x)
        b = P1y - a*P1x
        return a*xdata + b
    elif len(bgp)>2:
        # partition xdata
        xdat = []
        firstdat = True
        for i in range(len(bgp)-1):
            a= (bgp[i+1,1] - bgp[i,1])/(bgp[i+1,0] - bgp[i,0])
            b = bgp[i,1] - a*bgp[i,0]
            if firstdat == True:
                firstdat = False
                xdat.append(a*xdata[xdata <= bgp[i+1,0]] + b)
            else:
                xdat.append(a*xdata[((xdata <= bgp[i+1,0]) & (xdata > bgp[i,0]))] + b)
        xdat.append(a*xdata[xdata > bgp[-1,0]] + b)
        return np.hstack(xdat)



#### Import file function
def importfile(fil, lowcutoff = 0):
    dataa = []
    for line in open(fil):
        if (not(line.startswith('#') or ('nan' in line) ) and float(line.split(' ')[-1])> lowcutoff):
            dataa.append([float(item) for item in line.split(' ')])
    return np.transpose(np.array(dataa))

def importGridfile(fil, lowcutoff = 0):
    dataa = []
    for line in open(fil):
        if 'shape:' in line:
            shape = [int(i) for i in line.split(':')[1].split('x')]
        elif (not(line.startswith('#') ) and float(line.split(' ')[-1])> lowcutoff):
            dataa.append([float(item) for item in line.split(' ')])
    data = np.transpose(np.array(dataa))

    intensity = data[0].reshape(tuple(shape)).T
    ierror = data[1].reshape(tuple(shape)).T
    Qarray = data[2].reshape(tuple(shape))[:,0]
    Earray = data[3].reshape(tuple(shape))[0]
    return {'I':intensity, 'dI':ierror, 'Q':Qarray, 'E':Earray}





def printLaTexCEFparams(Bs):
    precision = 5
    '''prints CEF parameters in the output that Latex can read'''
    print('\\begin{table}\n\\caption{Fitted vs. Calculated CEF parameters for ?}')
    print('\\begin{ruledtabular}')
    print('\\begin{tabular}{c|'+'c'*len(Bs)+'}')
    # Create header
    print('$B_n^m$ (meV)' +' & Label'*len(Bs)
        +' \\tabularnewline\n \\hline ')
    for i, (n,m) in enumerate([[n,m] for n in range(2,8,2) for m in range(0,n+1, 3)]):
        print('$ B_'+str(n)+'^'+str(m)+'$ &', 
              ' & '.join([str(np.around(bb[i],decimals=precision)) for bb in Bs]),
              '\\tabularnewline')
    print('\\end{tabular}\\end{ruledtabular}')
    print('\\label{flo:CEF_params}\n\\end{table}')

#####################################################################################
#####################################################################################

### Import cif file (only works for rare earths now)

from pcf_lib.cifsymmetryimport import FindPointGroupSymOps
from pcf_lib.cif_import import CifFile

def importCIF(ciffile, mag_ion):
    '''Call this function to generate a PyCrystalField point charge model
    from a cif file'''
    cif = CifFile(ciffile)
    for ii, at in enumerate(cif.unitcell):
        if at[4] < 0: print('negative atom!',ii, at)
    centralIon, ligandPositions, ligandCharge = FindPointGroupSymOps(cif, mag_ion)

    Lig = Ligands(ion=centralIon, ionPos = [0,0,0], ligandPos = ligandPositions)
    # Create a point charge model, assuming that a mirror plane has been found.
    print('   Creating a point charge model...')
    PCM = Lig.PointChargeModel(printB = True, LigandCharge=ligandCharge[0], suppressminusm = True)

    return Lig, PCM

#####################################################################################
#####################################################################################
