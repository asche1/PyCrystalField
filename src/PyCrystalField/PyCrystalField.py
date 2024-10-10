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
from pcf_lib.plotLigands import exportLigandCif
from pcf_lib.Operators import Ket, Operator, LSOperator
from pcf_lib.StevensOperators import StevensOp, LS_StevensOp
from pcf_lib.PointChargeConstants import *
from pcf_lib.PCF_misc_functions import *
from numba import njit #, jitclass
try:
    from numba.experimental import jitclass
except ModuleNotFoundError:
    from numba import jitclass
from copy import deepcopy

# abspath = os.path.abspath(__file__)
# dname = os.path.dirname(abspath)
# os.chdir(dname)


print(' '+'*'*55 + '\n'+
     ' *                PyCrystalField 2.3.10                *\n' +
    #' *  Code to calculate the crystal Field Hamiltonian    *\n' +
    #' *   of magentic ions.                                 *\n' +
    ' *  Please cite  J. Appl. Cryst. (2021). 54, 356-362   * \n' +
    ' *    <https://doi.org/10.1107/S160057672001554X>      *\n ' + '*'*55+'\n')




JionTM = {}   # [S, L, J]

### These values are taken from the free ion states at 
### https://physics.nist.gov/PhysRefData/Elements/per_noframes.html
JionTM['Cu2+'] = [1/2, 2.]
JionTM['Ni2+'] = [1., 3.]
JionTM['Ni3+'] = [3/2, 3.]
JionTM['Co2+'] = [3/2, 3]
JionTM['Co3+'] = [2,   2]
JionTM['Fe2+'] = [2,   2]
JionTM['Fe3+'] = [5/2, 0]
JionTM['Mn2+'] = [5/2, 0]
JionTM['Mn3+'] = [2, 2]
JionTM['Mn4+'] = [3/2, 3]
JionTM['Cr2+'] = [2, 2]
JionTM['Cr3+'] = [3/2, 3]
JionTM['V2+']  = [3/2, 3]
JionTM['V3+']  = [1, 3]
JionTM['Ti2+']  = [1, 3]
JionTM['Ti3+']  = [1/2, 2]

JionTM['Nb3+'] = [1, 3]
JionTM['Tc4+'] = [3/2, 3]
JionTM['Ru3+'] = [5/2, 0]
JionTM['Rh3+'] = [2, 2]
JionTM['Pd2+'] = [1, 3]
JionTM['Pd3+'] = [3/2, 3]



Jion = {}   # [S, L, J]
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
## Actinides
Jion['U4+'] = [1., 5., 4.]
Jion['U3+'] = [1.5, 6., 4.5]


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


    def exportCif(self, filename):
        exportLigandCif(self, filename)


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
            try:
                if len(LigandCharge) == len(self.bonds):
                    charge = LigandCharge
                else:
                    charge = [LigandCharge]*len(self.bonds)
            except TypeError:
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
        OOO = []
        nonzeroB = []
        bnm_labels = []

        if self.suppressmm == False:  nmrange = [[n,m] for n in range(2,8,2) for m in range(-n,n+1)]
        elif self.suppressmm == True:   nmrange = [[n,m] for n in range(2,8,2) for m in range(0,n+1)]
        #for n,m in [[n,m] for n in range(2,8,2) for m in range(-n,n+1)]:
        for n,m in nmrange:
            # 1)  Compute gamma
            gamma = 0
            for i in range(len(self.bonds)):
                #print(np.squeeze(charge[i]))
                gamma += 4*np.pi/(2*n+1)*np.squeeze(charge[i]) *\
                            TessHarm(n,m, self.bonds[i][0], self.bonds[i][1], self.bonds[i][2])/\
                            (self.bondlen[i]**(n+1))

            # 2)  Compute CEF parameter
            B = -gamma * ahc* a0**n * Constant(n,m) * RadialIntegral(ion,n) * theta(ion,n)
            if printB ==True: print('B_'+str(n),m,' = ',np.around(B,decimals=8))
            if np.around(B,decimals=7) != 0:
                OOO.append(StevensOp(ionJ,n,m))
                nonzeroB.append(B)
                bnm_labels.append('B_{}^{}'.format(n,m))
            #print cef.StevensOp(ionJ,n,m)
            #self.H += np.around(B,decimals=15)*StevensOp(ionJ,n,m)
            if np.around(B,decimals=9) != 0:
                self.H += B*StevensOp(ionJ,n,m)
            self.B.append(B)
        self.B = np.array(self.B)

        newobj = CFLevels.Hamiltonian(self.H)
        newobj.O = OOO
        newobj.B = nonzeroB
        newobj.BnmLabels = bnm_labels
        newobj.ion = self.ion
        return newobj


    def FitChargesNeutrons(self, chisqfunc, fitargs, method='Powell', **kwargs):
        '''This is the old name. I keep it around so that the original code 
        will run with it.'''
        return self.FitCharges(chisqfunc, fitargs, method='Powell', **kwargs)

    def FitCharges(self, chisqfunc, fitargs, method='Powell', **kwargs):
        '''fits data'''

        # Define function to be fit
        fun, p0, resfunc = makeFitFunction(chisqfunc, fitargs, **dict(kwargs, LigandsObject=self) )

        print('\tFitting...')
        ############## Fit, using error function  #####################
        p_best = optimize.minimize(fun, p0, method=method)
        #p_best = optimize.minimize(fun, p0, method='Nelder-Mead')
        ###############################################################

        try:
            initialChisq, finalChisq = fun(p0), fun(p_best.x)
            finalvals = resfunc(p_best.x)
        except IndexError:
            initialChisq, finalChisq = fun(p0), fun([float(p_best.x)])
            finalvals = resfunc([float(p_best.x)])

        # split back into values
        finalCharges = finalvals['LigandCharge']

        # Print results
        print("\n#*********************************")
        print("# Final Stevens Operator Values")
        try:
            newH = self.PointChargeModel(kwargs['symequiv'], finalCharges, printB=True)
        except KeyError:
            print(float(p_best.x))
            newH = self.PointChargeModel(LigandCharge=[float(p_best.x)], printB=True)
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
        self.B = Parameters
        # self.Ci = B # Old definition of parameters
        try:
            self.J = (len(self.H) -1.)/2
            self.opttran = opttransition(Operator.Jx(self.J).O, Operator.Jy(self.J).O.imag, Operator.Jz(self.J).O)
        except TypeError: pass

    @classmethod
    def Bdict(cls, ion, Bdict):
        ionJ = Jion[ion][-1]
        Stev_O = []
        Parameters = []
        for Bnm in Bdict:
            Parameters.append(Bdict[Bnm])
            n = int(Bnm[1])
            m = int(Bnm[2:])    
            Stev_O.append(  StevensOp(ionJ,n,m)  )

        newcls = cls(Stev_O, Parameters)
        newcls.BnmLabels = Bdict.keys
        newcls.ion = ion
        return newcls

    @classmethod
    def Hamiltonian(cls, Hamil):
        newcls = cls([0,0],[0,0])  # Create empty class so we can just define Hamiltonian
        newcls.H = Hamil
        newcls.J = (len(Hamil) -1.)/2
        newcls.opttran = opttransition(Operator.Jx(newcls.J).O.real, Operator.Jy(newcls.J).O.imag, Operator.Jz(newcls.J).O.real)
        return newcls


    def newCoeff(self, newcoeff):
        self.B = np.array(newcoeff)
        newH = np.sum([a*b for a,b in zip(self.O, newcoeff)], axis=0)
        self.diagonalize(newH)

    def diagonalize(self, Hamiltonian=None, old=False):
        """A Hamiltonian can be passed to the function (used for data fits)
        or the initially defined hamiltonian is used."""
        if Hamiltonian is None:
            Hamiltonian = self.H
        else:
            self.H = Hamiltonian
        if old:
            diagonalH = LA.eigh(Hamiltonian)  #This was slower and less precise
        else:
            bands = self._findbands(Hamiltonian)
            diagonalH = LA.eig_banded(bands, lower=True)

        self.eigenvaluesNoNorm = diagonalH[0]
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

        self.eigenvaluesNoNorm = diagonalH[0]
        self.eigenvalues = diagonalH[0] - np.amin(diagonalH[0])
        self.eigenvectors = diagonalH[1].T
        # set very small values to zero
        tol = 1e-15
        self.eigenvalues[abs(self.eigenvalues) < tol] = 0.0
        self.eigenvectors[abs(self.eigenvectors) < tol] = 0.0

    def _findbands(self, matrix):
        '''used in the diagonalize_banded function'''
        diags = np.zeros((len(matrix),len(matrix)), dtype=np.complex128)
        for i in range(len(matrix)):
            diag = matrix.diagonal(i)
            if i == 0:
                diags[i] = diag
            else:
                diags[i][:-i] = diag
            if np.count_nonzero(np.around(diag,10)) > 0:
                nonzerobands = i
        return diags[:nonzerobands+1]




    def transitionIntensity(self, ii, jj, Temp):
        # for population factor weights
        beta = 1/(8.61733e-2*Temp)  # Boltzmann constant is in meV/K
        Z = sum([np.exp(-beta*en) for en in self.eigenvalues])
        # compute population factor
        pn = np.exp(-beta *self.eigenvalues[ii])/Z
        
        # compute amplitude
        mJn = self.opttran.transition(self.eigenvectors.real[ii] ,  self.eigenvectors.real[jj])
        return pn*mJn


    def neutronSpectrum(self, Earray, Temp, Ei, ResFunc, gamma = 0):
        # make angular momentum ket object
        #eigenkets = [Ket(ei) for ei in self.eigenvectors]

        try:
            eigenkets = self.eigenvectors.real
            intensity = np.zeros(len(Earray))
        except AttributeError:
            self.diagonalize()
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


    def neutronSpectrum_customLineshape(self, Earray, Temp, Ei, LineshapeFunc):
        '''calculate neutron spectrum with a custom lineshape
        which is a function of energy list and energy transfer.'''
        # make angular momentum ket object
        #eigenkets = [Ket(ei) for ei in self.eigenvectors]

        try:
            eigenkets = self.eigenvectors.real
            intensity = np.zeros(len(Earray))
        except AttributeError:
            self.diagonalize()
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
                    intensity += ((pn * mJn * LineshapeFunc(Earray - deltaE,
                                                            deltaE)).real).astype('float64')
                
        kpoverk = np.sqrt((Ei - Earray)/Ei) #k'/k = sqrt(E'/E)
        return intensity * kpoverk


    def normalizedNeutronSpectrum(self, Earray, Temp, ResFunc, gamma = 0):
        '''1D neutron spectrum without the Kf/Ki correction'''
        # make angular momentum ket object
        # eigenkets = [Ket(ei) for ei in self.eigenvectors]
        try:
            eigenkets = self.eigenvectors.real
            intensity = np.zeros(len(Earray))
        except AttributeError:
            self.diagonalize()
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


    def normalizedNeutronSpectrum_customLineshape(self, Earray, Temp, LineshapeFunc):
        '''1D neutron spectrum without the Kf/Ki correction.
        LineshapeFunc must be a function with arguments of energy list and 
        central energy.'''
        try:
            eigenkets = self.eigenvectors.real
            intensity = np.zeros(len(Earray))
        except AttributeError:
            self.diagonalize()
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
                    intensity += ((pn * mJn * LineshapeFunc(Earray - deltaE,
                                                            deltaE)).real).astype('float64')
        return intensity



    def neutronSpectrum2D(self, Earray, Qarray, Temp, Ei, ResFunc, gamma, Ion, DebyeWaller=0):
        intensity1D = self.neutronSpectrum(Earray, Temp, Ei, ResFunc,  gamma)

        # Scale by Debye-Waller Factor
        DWF = np.exp(1./3. * Qarray**2 * DebyeWaller**2)
        # Scale by form factor
        FormFactor = RE_FormFactor(Qarray,Ion)
        return np.outer(intensity1D, DWF*FormFactor)

    def normalizedNeutronSpectrum2D(self, Earray, Qarray, Temp, ResFunc, gamma, Ion, DebyeWaller=0):
        intensity1D = self.normalizedNeutronSpectrum(Earray, Temp, ResFunc,  gamma)

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
            if np.abs(value.imag) <= thresh:
                return (value.real).astype(float)
            else: 
                return value
        else:
            if np.all(np.abs(value.imag) < thresh):
                return (value.real)
            else: return value

    def printEigenvectors(self):
        '''prints eigenvectors and eigenvalues in a matrix'''
        try:
            eigenkets = self.eigenvectors.real
        except AttributeError:
            self.diagonalize()

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
        try:
            eigenkets = self.eigenvectors.real
        except AttributeError:
            self.diagonalize()
        
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
        '''field should be a 3-component vector. Temps may be an array.
        Returns a three-component vector [M_x, M_y, M_z].
        Field should be in units of Tesla, and magnetization is calculated in Bohr Magnetons'''
        if len(Field) != 3: 
            raise TypeError("Field needs to be 3-component vector")

        # A) Define magnetic Hamiltonian

        #Jx = Operator.Jx(self.J)
        # Jy = Operator.Jy(self.J).O
        #Jz = Operator.Jz(self.J)
        Jx = self.opttran.Jx
        Jy = self.opttran.Jy * 1j
        Jz = self.opttran.Jz

        #print(Jx)
        #print(Jy)
        if isinstance(ion, str):
            gJ = LandeGFactor(ion)
        else: gJ = ion
        muB = 5.7883818012e-2  # meV/T
        #mu0 = np.pi*4e-7       # T*m/A
        JdotB = gJ*muB*(Field[0]*Jx + Field[1]*Jy + Field[2]*Jz)

        # B) Diagonalize full Hamiltonian
        FieldHam = self.H + JdotB
        #FieldHam = self.H + JdotB.O
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
        deltaField needs to be a scalar value.
        Returns a powder average value if Field is a scalar, and returns
        [Chi_x, Chi_y, Chi_z] if Field is a vector.
        Field should be in Tesla, and susceptibility is calculated in Bohr Magnetons
        per Tesla.'''
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

        # Jx = Operator.Jx(self.J)
        # Jy = Operator.Jy(self.J)
        # Jz = Operator.Jz(self.J)


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

        zeroinds = np.where(np.around(self.eigenvalues,5)==0)
        gsEVec = self.eigenvectors[zeroinds]

        if len(zeroinds[0]) == 1:
            print('\tWARNING: only one ground state eivenvector found.')

            zeroinds = np.where(np.around(self.eigenvalues,1)==0)
            gsEVec = self.eigenvectors[zeroinds]
            if len(zeroinds[0]) == 2:
                print('\t\t Including another at {} meV'.format(self.eigenvalues[zeroinds[1]]))
            else: raise ValueError('Non-doublet ground state!')


        vv1 = np.around(gsEVec[0],10)
        vv2 = np.around(gsEVec[1],10)
        # Jx = Operator.Jx(self.J).O
        # Jy = Operator.Jy(self.J).O
        # Jz = Operator.Jz(self.J).O
        Jx = self.opttran.Jx
        Jy = self.opttran.Jy*1j
        Jz = self.opttran.Jz
        #print(vv1,'\n',vv2)
        jz01 = eliminateimag( np.dot(vv1,np.dot(Jz,np.conj(vv2))) )
        jz10 = eliminateimag( np.dot(vv2,np.dot(Jz,np.conj(vv1))) )
        jz00 = eliminateimag( np.dot(vv1,np.dot(Jz,np.conj(vv1))) )
        jz11 = eliminateimag( np.dot(vv2,np.dot(Jz,np.conj(vv2))) )
        
        
        jx01 = eliminateimag( np.dot(vv1,np.dot(Jx,np.conj(vv2))) )
        jx10 = eliminateimag( np.dot(vv2,np.dot(Jx,np.conj(vv1))) )
        jx00 = eliminateimag( np.dot(vv1,np.dot(Jx,np.conj(vv1))) )
        jx11 = eliminateimag( np.dot(vv2,np.dot(Jx,np.conj(vv2))) )
        
        jy01 = eliminateimag( np.dot(vv1,np.dot(Jy,np.conj(vv2))) )
        jy10 = eliminateimag( np.dot(vv2,np.dot(Jy,np.conj(vv1))) )
        jy00 = eliminateimag( np.dot(vv1,np.dot(Jy,np.conj(vv1))) )
        jy11 = eliminateimag( np.dot(vv2,np.dot(Jy,np.conj(vv2))) )
        
        gg = 2*np.array([[np.real(jx01), np.imag(jx01), jx00],
                         [np.real(jy01), np.imag(jy01), jy00],
                         [np.real(jz01), np.imag(jz01), np.abs(jz00)]])
        return gg*LandeGFactor(self.ion)


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
        # if len(self.B) != len(kwargs['coeff']):
        #     raise ValueError('coeff needs to have the same length as self.B')

        # Define function to be fit
        fun, p0, resfunc = makeFitFunction(chisqfunc, fitargs, **dict(kwargs, CFLevelsObject=self) )

        ############## Fit, using error function  #####################
        p_best = optimize.minimize(fun, p0, method=method)
        ###############################################################

        #print(fun(p_best.x))
        #print(chisqfunc(self, **kwargs))
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




# from numba import njit, jitclass
from numba import float64 #, complex128

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







### Same class, but in the LS basis

class LS_Ligands:
    """For doing point-charge calculations in LS basis"""
    def __init__(self,ion, ligandPos, SpinOrbitCoupling, latticeParams=None, ionPos=[0,0,0]):
        """Creates array of ligand bonds in cartesian coordinates.
        'ion' can either be the name of the ion or a list specifying L and S.
        For example, it could be 'Ni3+', or ['Ni3+', 0.5, 1]"""
        lp = latticeParams
        if lp == None:
            self.latt = lat.lattice(1,1,1,90,90,90)
        elif len(lp) != 6:
            raise LookupError("latticeParams needs to have 6 components: a,b,c,alpha,beta,gamma")
        else:
            self.latt = lat.lattice(lp[0], lp[1], lp[2], lp[3], lp[4], lp[5])

        self.bonds = np.array([np.array(O) - np.array(ionPos) for O in ligandPos])
        self.bonds = self.latt.cartesian(self.bonds).astype('float')
        self.bondlen = np.linalg.norm(self.bonds, axis=1)

        if isinstance(ion, str):
            self.ion = ion
            self.ionS = Jion[ion][0]
            self.ionL = Jion[ion][1]
        else:
            self.ion = ion[0]
            self.ionS = ion[1]
            self.ionL = ion[2]

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

        # if symequiv == None:
        #     charge = IonCharge*[LigandCharge]*len(self.bonds)
        # else:
        #     charge = [0]*len(self.bonds)
        #     for i,se in enumerate(symequiv):
        #         charge[i] = IonCharge*LigandCharge[se]
        # self.symequiv = symequiv

        if symequiv == None:
            # charge = IonCharge*[LigandCharge]*len(self.bonds)
            try:
                if len(LigandCharge) == len(self.bonds):
                    charge = LigandCharge
                else:
                    charge = [LigandCharge]*len(self.bonds)
            except TypeError:
                charge = [LigandCharge]*len(self.bonds)

        else:
            charge = [0]*len(self.bonds)
            for i,se in enumerate(symequiv):
                #charge[i] = IonCharge*LigandCharge[se]
                charge[i] = LigandCharge[se]
        
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
        OOO = []
        nonzeroB = []

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
            if np.around(B,decimals=8) != 0:
                OOO.append(LS_StevensOp(self.ionL,self.ionS,n,m))
                nonzeroB.append(B)
            #print cef.StevensOp(ionJ,n,m)
            if np.around(B,decimals=10) != 0:
                H += B*StevensOp(self.ionL,n,m)
            self.B.append(B)
        self.B = np.array(self.B)

        # Convert Hamiltonian to full LS basis
        H_CEF_O = np.hstack(np.hstack(np.multiply.outer(H, np.identity(int(2*self.ionS+1)))))
        self.H_CEF = LSOperator(self.ionL, self.ionS)
        self.H_CEF.O = H_CEF_O

        #self.H = self.H_CEF + self.H_LS

        newobj = LS_CFLevels.Hamiltonian(self.H_CEF, self.H_SOC, self.ionL, self.ionS)
        newobj.O = OOO
        newobj.B = nonzeroB
        return newobj


    def TMPointChargeModel(self, l=2, symequiv=None, LigandCharge= -2, IonCharge=1,
                        printB = True, suppressminusm = False):
        ''' For transition metals:
        Create point charge model of the crystal fields.
        Returns a CFLevels object with the hamiltonian defined.
        Define LigandCharge in units of e.'''
        halffilled = IsHalfFilled(self.ion)

        self.IonCharge = IonCharge
        # Lock suppressmm into whatever it was when PointChargeModel was first called.
        try: self.suppressmm
        except AttributeError:
            self.suppressmm = suppressminusm


        if symequiv == None:
            # charge = IonCharge*[LigandCharge]*len(self.bonds)
            try:
                if len(LigandCharge) == len(self.bonds):
                    charge = LigandCharge
                else:
                    charge = [LigandCharge]*len(self.bonds)
            except TypeError:
                charge = [LigandCharge]*len(self.bonds)

        else:
            charge = [0]*len(self.bonds)
            for i,se in enumerate(symequiv):
                #charge[i] = IonCharge*LigandCharge[se]
                charge[i] = LigandCharge[se]



        ahc = 1.43996e4  #Constant to get the energy in units of meV = alpha*hbar*c
        a0 = 0.52917721067    #Bohr radius in \AA

        H = np.zeros((int(2*self.ionL+1), int(2*self.ionL+1)),dtype = complex)
        self.B = []
        OOO = []
        nonzeroB = []

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
            B = -gamma * ahc* a0**n * Constant(n,m) * RadialIntegral_TM(self.ion, n) * TM_LStheta[n]
            if printB ==True: print('B_'+str(n),m,' = ',np.around(B,decimals=8))
            if np.around(B,decimals=8) != 0:
                OOO.append(LS_StevensOp(self.ionL,self.ionS,n,m))
                nonzeroB.append(B)
            #print cef.StevensOp(ionJ,n,m)
            if np.around(B,decimals=10) != 0:
                H += B*StevensOp(self.ionL,n,m)
            self.B.append(B)
        self.B = np.array(self.B)

        # Convert Hamiltonian to full LS basis
        H_CEF_O = np.hstack(np.hstack(np.multiply.outer(H, np.identity(int(2*self.ionS+1)))))
        self.H_CEF = LSOperator(self.ionL, self.ionS)
        self.H_CEF.O = H_CEF_O

        #self.H = self.H_CEF + self.H_LS
        newobj = LS_CFLevels.Hamiltonian(self.H_CEF, self.H_SOC, self.ionL, self.ionS)
        newobj.O = OOO
        newobj.B = nonzeroB
        return newobj



    def UnknownTMPointChargeModel(self, radialintegrals, halffilled=True, l=2,
                        symequiv=None, LigandCharge= -2, IonCharge=1,
                        printB = True, suppressminusm = False):
        ''' For transition metals if the radial integrals are not in PyCrystalField (d5 ions)
        Create point charge model of the crystal fields.
        Returns a CFLevels object with the hamiltonian defined.
        Define LigandCharge in units of e.'''

        self.IonCharge = IonCharge
        # Lock suppressmm into whatever it was when PointChargeModel was first called.
        try: self.suppressmm
        except AttributeError:
            self.suppressmm = suppressminusm

        if symequiv == None:
            # charge = IonCharge*[LigandCharge]*len(self.bonds)
            try:
                if len(LigandCharge) == len(self.bonds):
                    charge = LigandCharge
                else:
                    charge = [LigandCharge]*len(self.bonds)
            except TypeError:
                charge = [LigandCharge]*len(self.bonds)

        else:
            charge = [0]*len(self.bonds)
            for i,se in enumerate(symequiv):
                #charge[i] = IonCharge*LigandCharge[se]
                charge[i] = LigandCharge[se]


        ahc = 1.43996e4  #Constant to get the energy in units of meV = alpha*hbar*c
        a0 = 0.52917721067    #Bohr radius in \AA

        H = np.zeros((int(2*self.ionL+1), int(2*self.ionL+1)),dtype = complex)
        self.B = []
        OOO = []
        nonzeroB = []

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
            B = -gamma * ahc* a0**n * Constant(n,m) * radialintegrals[n] * TM_LStheta[n]
            if printB ==True: print('B_'+str(n),m,' = ',np.around(B,decimals=8))
            if np.around(B,decimals=8) != 0:
                OOO.append(LS_StevensOp(self.ionL,self.ionS,n,m))
                nonzeroB.append(B)
            #print cef.StevensOp(ionJ,n,m)
            if np.around(B,decimals=10) != 0:
                H += B*StevensOp(self.ionL,n,m)
            self.B.append(B)
        self.B = np.array(self.B)

        # Convert Hamiltonian to full LS basis
        H_CEF_O = np.hstack(np.hstack(np.multiply.outer(H, np.identity(int(2*self.ionS+1)))))
        self.H_CEF = LSOperator(self.ionL, self.ionS)
        self.H_CEF.O = H_CEF_O

        #self.H = self.H_CEF + self.H_LS
        newobj = LS_CFLevels.Hamiltonian(self.H_CEF, self.H_SOC, self.ionL, self.ionS)
        newobj.O = OOO
        newobj.B = nonzeroB
        return newobj




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
        self.B = Parameters
        self.S = S
        self.L = L
        # self.Ci = B  #old definition of B

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
    def Bdict(cls,  Bdict, L, S, SpinOrbitCoupling):
        '''Bdict must be a dictionary of labels and coefficients.
        Example: {'B20':-0.340}'''
        Stev_O = []
        Parameters = []
        for Bnm in Bdict:
            Parameters.append(Bdict[Bnm])
            n = int(Bnm[1])
            m = int(Bnm[2:])
            Stev_O.append(  LS_StevensOp(L,S,n,m)  )

        newcls = cls(Stev_O, Parameters, L, S, SpinOrbitCoupling)
        return newcls

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


    def newCoeff(self, newcoeff):
        self.B = np.array(newcoeff)
        newH = np.sum([a*b for a,b in zip(self.O, newcoeff)], axis=0)
        self.diagonalize(newH)


    def diagonalize(self, CEF_Hamiltonian=None):
        '''same as above, but using the Scipy eig_banded function'''
        if CEF_Hamiltonian is None:
            CEF_Hamiltonian = self.H_CEF.O
        else:
            self.H_CEF.O = CEF_Hamiltonian

        bands = self._findbands(CEF_Hamiltonian + self.H_SOC.O)
        diagonalH = LA.eig_banded(bands, lower=True)

        self.eigenvaluesNoNorm = diagonalH[0]
        self.eigenvalues = diagonalH[0] - np.amin(diagonalH[0])
        self.eigenvectors = diagonalH[1].T
        # set very small values to zero
        tol = 1e-15
        self.eigenvalues[abs(self.eigenvalues) < tol] = 0.0
        self.eigenvectors[abs(self.eigenvectors) < tol] = 0.0

    # Shared
    def _findbands(self, matrix):
        '''used in the diagonalize_banded function'''
        diags = np.zeros((len(matrix),len(matrix)), dtype=np.complex128)
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
        try:
            eigenkets = self.eigenvectors.real
            intensity = np.zeros(len(Earray))
        except AttributeError:
            self.diagonalize()
            eigenkets = self.eigenvectors.real
            intensity = np.zeros(len(Earray))

        maxtransition = 12 # because we can't see the others

        # make angular momentum ket object
        eigenkets = [Ket(ei) for ei in self.eigenvectors[:maxtransition]]


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


        kpoverk = np.sqrt((Ei - Earray)/Ei) #k'/k = sqrt(E'/E)
        return intensity * kpoverk


    # Shared
    def neutronSpectrum2D(self, Earray, Qarray, Temp, Ei, ResFunc, gamma, DebyeWaller, Ion):
        intensity1D = self.neutronSpectrum(Earray, Temp, Ei, ResFunc,  gamma)

        # Scale by Debye-Waller Factor
        DWF = np.exp(1./3. * Qarray**2 * DebyeWaller**2)
        # Scale by form factor
        FormFactor = RE_FormFactor(Qarray,Ion)
        return np.outer(intensity1D, DWF*FormFactor)


    def normalizedNeutronSpectrum(self, Earray, Temp, ResFunc, gamma = 0):
        '''neutron spectrum without the ki/Kf correction'''
        try:
            eigenkets = self.eigenvectors.real
            intensity = np.zeros(len(Earray))
        except AttributeError:
            self.diagonalize()
            eigenkets = self.eigenvectors.real
            intensity = np.zeros(len(Earray))

        maxtransition = 12 # because we can't see the others

        # make angular momentum ket object
        eigenkets = [Ket(ei) for ei in self.eigenvectors[:maxtransition]]

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
        return intensity


    def _transition(self,ket1,ket2):
        """Computes \sum_a |<|J_a|>|^2 = \sum_a |<|L_a + S_a|>|^2"""
        # ax = np.dot(np.conjugate(ket1.ket),np.dot(self.Jx.O,ket2.ket)) *\
        #         np.dot(np.conjugate(ket2.ket),np.dot(self.Jx.O,ket1.ket))
        # ay = np.dot(np.conjugate(ket1.ket),np.dot(self.Jy.O,ket2.ket)) *\
        #         np.dot(np.conjugate(ket2.ket),np.dot(self.Jy.O,ket1.ket))
        # az = np.dot(np.conjugate(ket1.ket),np.dot(self.Jz.O,ket2.ket)) *\
        #         np.dot(np.conjugate(ket2.ket),np.dot(self.Jz.O,ket1.ket))

        """Computes \sum_a |<|J_a|>|^2 = \sum_a |<|L_a + S_a|>|^2, including the 
        appropriate g factor."""
        ax = np.dot(np.conjugate(ket1.ket),np.dot(self.Jxg0.O,ket2.ket)) *\
                np.dot(np.conjugate(ket2.ket),np.dot(self.Jxg0.O,ket1.ket))
        ay = np.dot(np.conjugate(ket1.ket),np.dot(self.Jyg0.O,ket2.ket)) *\
                np.dot(np.conjugate(ket2.ket),np.dot(self.Jyg0.O,ket1.ket))
        az = np.dot(np.conjugate(ket1.ket),np.dot(self.Jzg0.O,ket2.ket)) *\
                np.dot(np.conjugate(ket2.ket),np.dot(self.Jzg0.O,ket1.ket))

        # eliminate tiny values
        ax, ay, az = np.around(ax, 10), np.around(ay, 10), np.around(az, 10)
        if (ax + ay + az).imag == 0:
            return ((ax + ay + az).real).astype(float)
        else:
            print(ax, ay, az)
            raise ValueError("non-real amplitude. Error somewhere.")
            
    # Shared
    def _lorentzian(self, x, x0, gamma):
        return 1/np.pi * (0.5*gamma)/((x-x0)**2 + (0.5*gamma)**2)

    # Shared
    def _voigt(self, x, x0, alpha, gamma):
        """ Return the Voigt line shape at x with Lorentzian component FWHM gamma
        and Gaussian component FWHM alpha."""
        sigma = (0.5*alpha) / np.sqrt(2 * np.log(2))
        return np.real(wofz(((x-x0) + 1j*(0.5*gamma))/sigma/np.sqrt(2))) / sigma\
                                                            /np.sqrt(2*np.pi)
    # Shared
    def _Re(self,value):
        thresh = 1e-9
        if np.size(value) == 1 & isinstance(value, complex):
            if np.abs(value.imag) <= thresh:
                return (value.real).astype(float)
            else: 
                return value
        else:
            if np.all(np.abs(value.imag) < thresh):
                return (value.real)
            else: return value

    # Shared
    def printEigenvectors(self):
        '''prints eigenvectors and eigenvalues in a matrix'''
        try:
            eigenkets = self.eigenvectors.real
        except AttributeError:
            self.diagonalize()
        
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
        Jxg0, Jyg0, Jzg0 = self.Jxg0.O, self.Jyg0.O, self.Jzg0.O
        Jx, Jy, Jz = self.Jx.O, self.Jy.O, self.Jz.O


        jzg01 = np.dot(vv1,np.dot(Jzg0,np.conj(vv2))) 
        jzg10 = np.dot(vv2,np.dot(Jzg0,np.conj(vv1)))
        jzg00 = np.dot(vv1,np.dot(Jzg0,np.conj(vv1)))
        jzg11 = np.dot(vv2,np.dot(Jzg0,np.conj(vv2)))
               
        jxg01 = np.dot(vv1,np.dot(Jxg0,np.conj(vv2)))
        jxg10 = np.dot(vv2,np.dot(Jxg0,np.conj(vv1)))
        jxg00 = np.dot(vv1,np.dot(Jxg0,np.conj(vv1)))
        jxg11 = np.dot(vv2,np.dot(Jxg0,np.conj(vv2)))
        
        jyg01 = np.dot(vv1,np.dot(Jyg0,np.conj(vv2)))
        jyg10 = np.dot(vv2,np.dot(Jyg0,np.conj(vv1)))
        jyg00 = np.dot(vv1,np.dot(Jyg0,np.conj(vv1)))


        jz01 = np.dot(vv1,np.dot(Jz,np.conj(vv2)))
        jz10 = np.dot(vv2,np.dot(Jz,np.conj(vv1)))
        jz00 = np.dot(vv1,np.dot(Jz,np.conj(vv1)))
        jz11 = np.dot(vv2,np.dot(Jz,np.conj(vv2)))
        
        jx01 = np.dot(vv1,np.dot(Jx,np.conj(vv2)))
        jx10 = np.dot(vv2,np.dot(Jx,np.conj(vv1)))
        jx00 = np.dot(vv1,np.dot(Jx,np.conj(vv1)))
        jx11 = np.dot(vv2,np.dot(Jx,np.conj(vv2)))
        
        jy01 = np.dot(vv1,np.dot(Jy,np.conj(vv2)))
        jy10 = np.dot(vv2,np.dot(Jy,np.conj(vv1)))
        jy00 = np.dot(vv1,np.dot(Jy,np.conj(vv1)))
        jy11 = np.dot(vv2,np.dot(Jy,np.conj(vv2)))
        

        # JXmatrix = np.array([[jx00,jy00,jz00],[jx01,jy01,jz01],[jx10,jy10,jz10]])
        # print(JXmatrix)
        # #zmartix=[gzx,gzy,gzz],
        # zmatrix=np.matmul(np.array([jzg00,jzg01,jzg10]),np.linalg.inv(np.transpose(JXmatrix)))
        # #xmartix=[gyx,gyy,gyz]
        # xmatrix=np.matmul(np.array([jxg00,jxg01,jxg10]),np.linalg.inv(np.transpose(JXmatrix)))
        # #ymartix=[gxx,gxy,gxz]
        # ymatrix=np.matmul(np.array([jyg00,jyg01,jyg10]),np.linalg.inv(np.transpose(JXmatrix)))
        
        # gg = np.array([xmatrix,ymatrix,zmatrix])

        gg = 2*np.array([[np.abs(np.real(jxg01)), np.imag(jxg01), jxg00],
                        [np.real(jyg01), np.imag(jyg01), jyg00],
                        [np.real(jzg01), np.imag(jzg01), np.abs(jzg00)]])
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
        if len(self.B) != len(kwargs['coeff']):
            raise ValueError('coeff needs to have the same length as self.B')

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


    def printLaTexEigenvectors(self):
        '''prints eigenvectors and eigenvalues in the output that Latex can read'''
        try:
            eigenkets = self.eigenvectors.real
        except AttributeError:
            self.diagonalize()

        # Define S array
        if (self.S*2) %2 ==0:
            Sarray = [str(int(kk)) for kk in 
                                    np.arange(-self.S,self.S+1)]
        else:
            Sarray = ['-\\frac{'+str(abs(kk))+'}{2}' if kk <0
                            else '\\frac{'+str(abs(kk))+'}{2}'
                            for kk in np.arange(-int(self.S*2), int(self.S*2+2), 2)]

        # Define L array
        Larray = [str(int(kk)) for kk in  np.arange(-self.L,self.L+1)]

        # Define Ket names
        KetNames = ['$|'+LL+','+SS+'\\rangle$' for LL in Larray  for SS in Sarray]

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





# Constants for converting between Wybourne and Stevens Operators
LambdaConstants = {}
LambdaConstants[2] = {}
LambdaConstants[2][0] = 1/2
LambdaConstants[2][1] = np.sqrt(6)
LambdaConstants[2][2] = np.sqrt(6)/2
LambdaConstants[4] = {}
LambdaConstants[4][0] = 1/8
LambdaConstants[4][1] = np.sqrt(5)/2
LambdaConstants[4][2] = np.sqrt(10)/4
LambdaConstants[4][3] = np.sqrt(35)/2
LambdaConstants[4][4] = np.sqrt(70)/8
LambdaConstants[6] = {}
LambdaConstants[6][0] = 1/16
LambdaConstants[6][1] = np.sqrt(42)/8
LambdaConstants[6][2] = np.sqrt(105)/16
LambdaConstants[6][3] = np.sqrt(105)/8
LambdaConstants[6][4] = np.sqrt(14)*3/16
LambdaConstants[6][5] = np.sqrt(77)*3/8
LambdaConstants[6][6] = np.sqrt(231)/16

def WybourneToStevens(ion, Bdict, LS=False):
    StevDict = {}
    for Anm in Bdict:
        n = int(Anm[1])
        m = int(Anm[2:])
        if LS:
            StevDict['B'+Anm[1:]] = LambdaConstants[n][m]*LStheta(ion,n)*Bdict[Anm]
        else:
            StevDict['B'+Anm[1:]] = LambdaConstants[n][m]*theta(ion,n)*Bdict[Anm]
    return StevDict


def StevensToWybourne(ion, Bdict, LS=False):
    WybDict = {}
    for Anm in Bdict:
        n = int(Anm[1])
        m = int(Anm[2:])
        if LS:
            WybDict['B'+Anm[1:]] = Bdict[Anm]/(LambdaConstants[n][m]*LStheta(ion,n))
        else:
            WybDict['B'+Anm[1:]] = Bdict[Anm]/(LambdaConstants[n][m]*theta(ion,n))
    return WybDict


#####################################################################################
#####################################################################################



### Import cif file (works for rare earths, and some TM ions)

from pcf_lib.cifsymmetryimport import FindPointGroupSymOps
from pcf_lib.cif_import import CifFile

def importCIF(ciffile, mag_ion = None, Zaxis = None, Yaxis = None, LS_Coupling = None,
                crystalImage=False, NumIonNeighbors=1, ForceImaginary=False, 
                ionL = None, ionS = None, CoordinationNumber = None, MaxDistance=None,
                ):
    '''Call this function to generate a PyCrystalField point charge model
    from a cif file'''
    cif = CifFile(ciffile)
    if mag_ion == None: #take the first rare earth in the cif file as the central ion
        for asuc in cif.asymunitcell:
            if asuc[1].strip('3+') in ['Sm','Pm','Nd','Ce','Dy','Ho','Tm','Pr','Er','Tb','Yb']:
                mag_ion = asuc[0]
                print('No mag_ion ion listed, assuming', mag_ion, 'is the central ion.')
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


    #### 
    output = []

    for cf in cifs:

        ## Calculate the ligand positions
        centralIon, ligandPositions, ligandCharge, inv, ligandNames = FindPointGroupSymOps(cf, mag_ion, Zaxis, 
                                                                    Yaxis, crystalImage,NumIonNeighbors,
                                                                    CoordinationNumber, MaxDistance)
        #print(ligandNames)
        if centralIon in Jion: # It's a rare earth ion
            if LS_Coupling:
                Lig = LS_Ligands(ion=centralIon, ionPos = [0,0,0], ligandPos = ligandPositions, 
                            SpinOrbitCoupling=LS_Coupling)

            else:
                Lig = Ligands(ion=centralIon, ionPos = [0,0,0], ligandPos = ligandPositions)
            # Create a point charge model, assuming that a mirror plane has been found.
            print('   Creating a point charge model...')
            if ForceImaginary:
                PCM = Lig.PointChargeModel(printB = True, LigandCharge=ligandCharge, suppressminusm = False)
            else:
                PCM = Lig.PointChargeModel(printB = True, LigandCharge=ligandCharge, suppressminusm = inv)


        else: # It's not a rare earth!
            if (ionL == None) | (ionS == None):
                raise TypeError('\tplease specify the ionL and ionS values in the importCIF function for '+ centralIon)

            if LS_Coupling: # User-provided SOC
                Lig = LS_Ligands(ion=[centralIon, ionS, ionL], ionPos = [0,0,0], 
                        ligandPos = ligandPositions,  SpinOrbitCoupling=LS_Coupling)
            else: # Look up SOC in a table
                print('    No SOC provided, assuming SOC =', np.around(SpOrbCoup[centralIon],2), 'meV for '+
                       centralIon +"\n           (if you'd like to adjust this, use the 'LS_Coupling' command).\n")
                Lig = LS_Ligands(ion=[centralIon, ionS, ionL], ionPos = [0,0,0], 
                        ligandPos = ligandPositions,  SpinOrbitCoupling=SpOrbCoup[centralIon])

            if ForceImaginary:
                PCM = Lig.TMPointChargeModel(printB = True, LigandCharge=ligandCharge, suppressminusm = False)
            else:
                PCM = Lig.TMPointChargeModel(printB = True, LigandCharge=ligandCharge, suppressminusm = inv)

        Lig.LigandNames = ligandNames
        output.append([Lig, PCM])

    if len(output) == 1:
        return Lig, PCM
    else: 
        print('WARNING: more than one ligand position given...\n  '+
            ' outputting [[Ligands1, CFLevels1], [Ligands2, CFLevels2], [Ligands3, CFLevels3]]')
        return output


#####################################################################################
#####################################################################################

# Heat capacity from Victor Porée
def partition_func(Eis,T):
    # partition function
    k_b = 8.6173303E-2 # in meV.K-1
    return np.sum(np.exp(-Eis/(k_b*T)))

def Cp_from_CEF(Eis,T):
    def Cp1T(t):
        R = 8.31432  # in J/K per mol
        k_b = 8.6173303E-2# in meV.K-1
        beta = k_b * t
        Z = partition_func(Eis, t)
        fs = np.sum( (Eis/beta)**2 * np.exp(-Eis/beta) )
        ss = np.sum( (Eis/beta)*np.exp(-Eis/beta) )
        return ((R/Z) * (fs - ss**2/Z))
    return np.array(list(map(Cp1T, T)))
