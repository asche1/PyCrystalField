import numpy as np
from numba import njit

def MomIntertia(atoms):
    '''Calculate the moment of intertia tensor of a group of ions'''
    II = np.zeros((3,3))
    for at in atoms:
        x,y,z = at
        II[0,0] += y**2 + z**2
        II[1,1] += x**2 + z**2
        II[2,2] += x**2 + y**2
        II[0,1] += -x*y
        II[0,2] += -x*z
        II[1,2] += -y*z
        II[1,0] += -x*y
        II[2,0] += -x*z
        II[2,1] += -y*z
    return II

def selectZaxisMI(atoms):
    '''Calculate the moment of intertia tensor of a group of ions,
    then select the outlier axis as the z axis.'''
    II = MomIntertia(atoms)
    evals, evecs = np.linalg.eig(II)
    # Find the "outlier" eigenvalue
    evd = [np.abs(evals[0] - evals[1]) + np.abs(evals[0] - evals[2]),
           np.abs(evals[1] - evals[0]) + np.abs(evals[1] - evals[2]),
           np.abs(evals[2] - evals[0]) + np.abs(evals[2] - evals[1])]
    rotAxis = evecs.T[np.argmax(evd)]
    yaxis = evecs.T[np.argmin(evd)]
    ## Ensure that the y axis is orthogonal...
    ## yaxis -= np.dot(yaxis,rotAxis)/np.linalg.norm(rotAxis)
    return rotAxis, yaxis


#################################### Try again with Continuous Shape Measures

# @njit
def ContinuousShapeMeasure(shape1,shape2):
    CSM = 0
    for r1 in np.array(shape1):
        dist = np.sum((np.array(shape2) - r1)**2, axis=1)
        CSM += np.min(dist)
    return CSM



######

@njit
def anglesToVector(theta,phi):
    return np.sin(theta)*np.cos(phi),  np.sin(theta)*np.sin(phi),  np.cos(theta)

@njit
def rotationMatrix(theta,phi, angle):
    '''Angle is the rotation about the axis defined by theta and phi.'''
    u, v, w = anglesToVector(theta,phi)
    rotmatrix = np.zeros((3,3))
    rotmatrix[0,0] = (u**2 +(v**2 + w**2)*np.cos(angle))
    rotmatrix[0,1] = (u*v*(1- np.cos(angle)) - w*np.sin(angle))
    rotmatrix[0,2] = (u*w*(1- np.cos(angle)) + v*np.sin(angle))
    rotmatrix[1,0] = (u*v*(1- np.cos(angle)) + w*np.sin(angle))
    rotmatrix[1,1] = (v**2 +(u**2 + w**2)*np.cos(angle))
    rotmatrix[1,2] = (v*w*(1- np.cos(angle)) - u*np.sin(angle))
    rotmatrix[2,0] = (u*w*(1- np.cos(angle)) - v*np.sin(angle))
    rotmatrix[2,1] = (v*w*(1- np.cos(angle)) + u*np.sin(angle))
    rotmatrix[2,2] = (w**2 +(v**2 + u**2)*np.cos(angle))
    return rotmatrix


def rotateArbAxis(atoms, theta,phi, angle):
    rotmat = rotationMatrix(theta,phi, angle)
    newat = []
    for at in atoms:
        newat.append(np.dot(rotmat,at))
    return newat


# print(ContinuousShapeMeasure(polyhedron1, rotateArbAxis(polyhedron1, np.pi/2, 0, np.pi/2)))

from scipy.optimize import minimize

def findZaxis_SOM_rotation(atoms, angle):
    '''use a symmetry operation measure of a rotation of angle to 
    select the direction that is closest to having this symmetry'''
    def fitfun(xx):
        theta,phi = xx
        return ContinuousShapeMeasure(atoms, rotateArbAxis(atoms, theta, phi, angle))

    ## Select starting parameters
    startX0s = []
    startFF = []
    for i in range(900):
        x,y,z = np.random.uniform(-1,1,size=3)
        norm = np.sqrt((x**2 + y**2 + z**2))
        if norm <= 1:
            startX0s.append([np.arcsin(z/norm), np.arctan(y/x)])
            startFF.append(fitfun(startX0s[-1]))        
    x0 = startX0s[np.argmin(startFF)]
    #print(x0)

    res = minimize(fitfun, x0=x0, method='Powell')
    return np.array(anglesToVector(*res.x)), res.fun



def findZaxis(atoms):
    '''use Continuous Shape measures to identify a three-fold, four-fold,
    or two-fold rotation axis. If none can be found, use a moment of intertia
    tensor.'''
    RotA4, f4 = findZaxis_SOM_rotation(atoms, np.pi/2)  ## four-fold rotation
    # print('############ FOUR FOLD ROTATION CSM =',f4,'\n')
    if f4 < 1:
        yax = np.cross(RotA4, atoms[0])
        yax /= np.linalg.norm(yax)
        print('\tFound a near-4-fold axis...  CSM=', f4)
        return RotA4, yax
    else:
        RotA3, f3 = findZaxis_SOM_rotation(atoms, np.pi/3*2)  ## three-fold rotation
        if f3 < 1:
            yax = np.cross(RotA3, atoms[0])
            yax /= np.linalg.norm(yax)
            print('\tFound a near-3-fold axis...  CSM=', f3)
            return RotA3, yax
        else:
            RotA5, f5 = findZaxis_SOM_rotation(atoms, np.pi/5*2)  ## five-fold rotation
            if f5 < 1:
                yax = np.cross(RotA5, atoms[0])
                yax /= np.linalg.norm(yax)
                print('\tFound a near-5-fold axis...  CSM=', f5)
                return RotA3, yax
            else:
                RotA2, f2 = findZaxis_SOM_rotation(atoms, np.pi)  ## two-fold rotation
                if f2 < 1:
                    yax = np.cross(RotA2, atoms[0])
                    yax /= np.linalg.norm(yax)
                    print('\tFound a near-2-fold axis...  CSM=', f2)
                    return RotA2, yax
                else:
                    print('\tUsing moment of intertia tensor to estimate z axis...')
                    return selectZaxisMI(atoms) ## Select using moment of intertia



