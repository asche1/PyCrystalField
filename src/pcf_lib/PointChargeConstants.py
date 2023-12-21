import numpy as np
import os

#### Point Charge Approximation stuff


#Import prefactors
directory = os.path.dirname(os.path.realpath(__file__))+'/'
#Import prefactors
coef = np.genfromtxt(directory+'TessHarmConsts.txt',delimiter = ',')
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
for line in open(directory+'RadialIntegrals.txt'):
    if not line.startswith('#'):
        l = line.split(',')
        radialI[l[0]] = [float(v) for v in l[1:]]


TMradialI = {}
for line in open(directory+'RadialIntegrals_TM.txt'):
    if not line.startswith('#'):
        l = line.split(',')
        TMradialI[l[0]] = [float(v) for v in l[1:]]

def RadialIntegral(ion,n):
    """Returns the radial integral of a rare earth ion plus self-shielding.
    Comes out in units of Bohr radius"""
    if ion == 'U4+':
        U4r = {2:2.042, 4:7.632, 6:47.774}  # from Freeman, Desclaux, Lander, and Faber, PRB (1976), Table I
        return U4r[n]
    elif ion == 'U3+':
        U3r = {2:2.346, 4:10.906, 6:90.544}  # from Freeman, Desclaux, Lander, and Faber, PRB (1976), Table I
        return U3r[n]
    else:
        shielding = 1- radialI[ion][int(n/2-1)]
        return radialI[ion][int(n/2-1) + 3] * shielding

BohrRadius= 0.5291772109 # Official NIST value in units of /AA

def RadialIntegral_TM(ion,n):
    """Returns the radial integral of a transition ion.
    The listed constants are in AA, so we convert to Bohr Radii"""
    return TMradialI[ion][int(n/2-1)]/(BohrRadius**n)


def IsHalfFilled(ion):
    '''determine if the ion has a half-filled shell or not.'''
    if ion in ['Cu2+', 'Ni2+', 'Ni3+', 'Co2+', 'Co3+', 'Fe2+', 'Fe3+', 'Mn2+', 'Rh3+','Pd2+', 'Pd3+']:
        return True
    elif ion in ['Mn3+','Mn4+','Cr2+','Cr3+','V2+','V3+','Ti2+','Ti3+','Ru3+','Tc4+','Nb3+']:
        return False
    else:
        raise ValueError('{} is not a known ion for PyCrystalField.'.format(ion))


########### Spin orbit values
hc = 1.23984193e-1 #meV*cm
SpOrbCoup = {}
for line in open(directory+'SpinOrbitCouplingConstants_TM.txt'):
    if not line.startswith('#'):
        l = line.split(',')
        SpOrbCoup[l[0]] = [float(v)*hc for v in l[1:]]



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
LSThet['U4+'] = [-0.0148148148148, -0.0003848003848, 2.46666913334e-05] #same as Pr3+
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
Thet['U4+'] = [-2.*2*13/(3*3*5*5*11), -2.*2/(3*3*5*11*11), 2.**4*17/(3**4*5*7*11**2*13)] #same as Pr3+
Thet['Nd3+'] = [-7./(3**2*11**2) , -2.**3*17/(3**3*11**3*13), -5.*17*19/(3**3*7*11**3*13**2)]
Thet['U3+'] = [-7./(3**2*11**2) , -2.**3*17/(3**3*11**3*13), -5.*17*19/(3**3*7*11**3*13**2)] #same as Nd3+
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

