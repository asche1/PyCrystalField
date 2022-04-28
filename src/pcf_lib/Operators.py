import numpy as np

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










