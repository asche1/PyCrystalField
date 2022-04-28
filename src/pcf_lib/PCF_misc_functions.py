import numpy as np

'''Miscellaneous functions for PyCrystalField, 
mostly from the earliest version used to fit Nd3Sb3Mg2O14'''


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


