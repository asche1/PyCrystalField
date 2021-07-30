import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.optimize import minimize

import PyCrystalField as cef

import sys
from pythonlib.functions import SimDat


ions = ['Sm3+','Pm3+','Nd3+','Ce3+','Dy3+','Ho3+','Tm3+','Pr3+','Er3+','Tb3+','Yb3+']

for ion in ions:
    print('*'*40, ion)

    filename = ion[:2]+'TO'
            
    Data = SimDat('SimulatedData/{}_simdat.txt'.format(filename))

    YTOLig, Yb = cef.importCIF('SimulatedDataCode/{}.cif'.format(filename), ion[:2]+'1')
    Yb.diagonalize()

    maxE = np.max(Yb.eigenvalues)*1.1 + 5
    RF = lambda x: 0.03*(maxE - x/1.3)

    #############
    npts = len(Data.hw)*len(Data.Temps)

    def err_global(CFLevelsObject, coeff, gammas, pref):
        """Global error to all functions passed to it, used for fitting"""
        # define new Hamiltonian
        newH = np.sum([a*b for a,b in zip(CFLevelsObject.O, coeff)], axis=0)
        CFLevelsObject.diagonalize(newH)

        erro = 0
        
        # Compute error in neutron spectrum
        for i,T in enumerate(Data.Temps):
            errspec = (pref* CFLevelsObject.normalizedNeutronSpectrum(Earray=Data.hw, Temp=T,
                                        ResFunc= RF, gamma=gammas[i]) ) - \
                      (Data.II[i])
            erro += np.nansum(errspec**2/Data.dII[i]**2)
        
        normerr = (npts - len(coeff) - len(gammas) - 1)
        # print("err = "+str(erro/normerr)+'     ', end='\r')
        return erro/normerr

    err_global(Yb, coeff=Yb.B, gammas=[0.5*maxE/100, 1.5*maxE/100], pref=1)


    ###############

    def chisquared(pars):
        return err_global(Yb, coeff = pars[3:], gammas= pars[:2], pref= pars[2])

    # initialres = minimize(chisquared, x0 = np.hstack((Yb.B, [0.5,1.5], 1)), method='Nelder-Mead')

    # bestPars = initialres.x
    bestPars = np.hstack(([0.5*maxE/100, 1.5*maxE/100], 1, Yb.B))
    bestChiSq = chisquared(bestPars)

    lowChisqP = []
    lowChisqChi = []

    #### 

    for i in range(len(Yb.B)):
        # Loop through 
        baseB20 = bestPars[i+3]
        newpars = deepcopy(bestPars)
        jj = 1
        if baseB20 > 5e-3:
            step = 0.05
        else:
            step = 0.2
        while True:
            testB20 = baseB20*jj
            def truncatedChisquared(pars):
                return chisquared(np.hstack((pars[:i+3], testB20, pars[i+3:])))
            res = minimize(truncatedChisquared, x0 = np.hstack((newpars[:i+3], newpars[i+4:])), 
                           method='Nelder-Mead')

            print(i, jj, res.fun)
            newpars = np.hstack((res.x[:i+3], testB20, res.x[i+3:]))

            if res.fun > (bestChiSq+1):
                break
            else:
                lowChisqP.append(newpars)
                lowChisqChi.append(res.fun)
                jj += step
            if np.around(np.abs(jj),2) == 5: step *= 2
            elif np.around(np.abs(jj),2) in [20,50]: step *= 5
            elif np.abs(jj) > 200: break


        if baseB20 > 5e-3:
            step = 0.05
        else:
            step = 0.2
        jj = 1-step
        newpars = deepcopy(bestPars)
        ## Do again, counting the other way.
        while True:
            testB20 = baseB20*jj
            def truncatedChisquared(pars):
                return chisquared(np.hstack((pars[:i+3], testB20, pars[i+3:])))
            res = minimize(truncatedChisquared, x0 = np.hstack((newpars[:i+3], newpars[i+1+3:])), 
                           method='Nelder-Mead')
            print(i, jj, res.fun)
            newpars = np.hstack((res.x[:i+3], testB20, res.x[i+3:]))
            if res.fun > (bestChiSq+1):
                break
            else:
                lowChisqP.append(newpars)
                lowChisqChi.append(res.fun)
                jj -= step
            if np.around(np.abs(jj),2) == 5: step *= 2
            elif np.around(np.abs(jj),2) in [20,50]: step *= 5
            elif np.abs(jj) > 200: break


    ############# Save results

    lowChisqP = np.array(lowChisqP)

    np.save('fitresults/{}2_BestFit_uncertainty_P'.format(filename), lowChisqP)
    np.save('fitresults/{}2_BestFit_uncertainty_Chi'.format(filename),lowChisqChi)

    lowChisqGtensor = []
    lowChisqEvals = []
    lowChisqEvecs = []
    if len(Yb.eigenvalues) % 2 == 0:
        print('G!')

    for i,P in enumerate(lowChisqP):
        B = P[3:]
        newH = np.sum([a*b for a,b in zip(Yb.O, B)], axis=0)
        Yb.diagonalize(newH)
        
        #if len(Yb.eigenvalues) % 2 == 0:
        try:
            lowChisqGtensor.append(Yb.gtensor())
        except ValueError: pass
        
        lowChisqEvals.append(Yb.eigenvalues)

        lowChisqEvecs.append(Yb.eigenvectors)
        
    np.save('fitresults/{}2_BestFit_uncertainty_Gtensor'.format(filename),lowChisqGtensor)
    np.save('fitresults/{}2_BestFit_uncertainty_Evals'.format(filename), lowChisqEvals)
    np.save('fitresults/{}2_BestFit_uncertainty_Evecs'.format(filename), lowChisqEvecs)
