import pickle
import numpy as np
#import time
#from dataset import DataSet
import os
directory = os.path.dirname(os.path.realpath(__file__))+'/'


####################################################################################
## Kemp's function
def isotropic_ff(s,ion):
    fhd = open(directory+'3d_formfactors_j0.pck','rb')
    form_factors_3d = pickle.load(fhd)
    fhd.close()
    
    coefs = form_factors_3d[ion]
    
    j0 = coefs[0]*np.exp(-coefs[1]*s**2) + coefs[2]*np.exp(-coefs[3]*s**2)  + coefs[4]*np.exp(-coefs[5]*s**2) +coefs[6]

    fhd = open(directory+'3d_formfactors_j2.pck','rb')
    form_factors_3d = pickle.load(fhd)
    fhd.close()
    
    coefs = form_factors_3d[ion]
    
    j2 = s**2*(coefs[0]*np.exp(-coefs[1]*s**2) + coefs[2]*np.exp(-coefs[3]*s**2)  + coefs[4]*np.exp(-coefs[5]*s**2) +coefs[6])

    return [j0,j2] 


##### My own addition



######################################################################################



# My function (for now, only valid for rare earths.)

Jion = {}   # [S, L, J]
Jion['Ce3+'] = [0.5, 3., 2.5]
Jion['Pr3+'] = [1., 5., 4.]
Jion['Nd3+'] = [1.5, 6., 4.5]
Jion['Pm3+'] = [2., 6., 4.]
Jion['Sm3+'] = [2.5, 5, 2.5]
Jion['Tb3+'] = [3., 3., 6.]
Jion['Dy3+'] = [2.5, 5., 7.5]
Jion['Ho3+'] = [2., 6., 8.]
Jion['Er3+'] = [1.5, 6., 7.5]
Jion['Tm3+'] = [1., 5., 6.]
Jion['Yb3+'] = [0.5, 3., 3.5]

def LandeGFactor(ion):
    s, l, j = Jion[ion]
    return 1.5 + (s*(s+1.) - l*(l+1.))/(2.*j*(j+1.))

def importRE_FF(ion):
    coefs = [[],[]]
    j=0
    for line in open(directory+'RE_formfactors.pck'):
        if not line.startswith(('#', ' ','\n')):
            if line.split(' \t')[0] in ion:
                coefs[j] = [float(i) for i in line.split(' \t')[1:]]
                j+=1
                #print line.split(' \t')
    return coefs[0], coefs[1]

def RE_FormFactor(magQ,ion):
    """This uses the dipole approximation.
    Note that Q must be a scalar in inverseA"""
    #magQ = np.linalg.norm(lattice.inverseA(Q),axis=-1)
    s = magQ/(4.*np.pi)
    
    coefs0, coefs2 = importRE_FF(ion)
    
    j0 = coefs0[0]*np.exp(-coefs0[1]*s**2) + coefs0[2]*np.exp(-coefs0[3]*s**2)  + \
        coefs0[4]*np.exp(-coefs0[5]*s**2) +coefs0[6]

    j2 = s**2*(coefs2[0]*np.exp(-coefs2[1]*s**2) + coefs2[2]*np.exp(-coefs2[3]*s**2)+\
               coefs2[4]*np.exp(-coefs2[5]*s**2) + coefs2[6])

    S, L, J = Jion[ion]
     
    j2factor = (J*(J+1.) - S*(S+1.) + L*(L+1.))/(3.*J*(J+1.) + S*(S+1.) - L*(L+1.))
    return (j0 + j2*j2factor)**2

#print importRE_FF('Yb3+')

#print RE_FormFactor(np.arange(0,5,0.1),'Nd3+')
