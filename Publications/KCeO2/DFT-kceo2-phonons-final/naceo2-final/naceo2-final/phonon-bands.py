import yaml
from pylab import *
from itertools import groupby
plt.style.use("~/.matplotlib/styles/small.mplstyle")

THz_to_cminv = 33.35641
THz_to_meV = 4.135665538536


def getbands(folder):
    yam = yaml.load(open('%s/band.yaml'%folder, 'r'),Loader=yaml.Loader)

    # first get distance along k path
    nb = len(yam['phonon'][0]['band'])  # number of bands
    nq = yam['nqpoint']                 # number of q points

    # values of q points
    qp = array([yam['phonon'][i]['distance'] for i in range(nq)])

    # values of frequencies in bands
    freqs = zeros([nb, nq])
    for j in range(yam['nqpoint']):
        freqs[:,j] = array([yam['phonon'][j]['band'][i]['frequency'] for i in range(nb)])


    # Older versions of phonopy (<2.12)
    # xt = []
    # xl = []
    # for i in range(nq):
    #     try:
    #         xl.append(yam['phonon'][i]['label'])
    #         xt.append(yam['phonon'][i]['distance'])
    #     except:
    #         continue
    

    # New versions of phonopy (>=2.12)
    l = yam['labels']
    fl = list(array(l).flat)
    xl = [i[0] for i in groupby(fl)]
    nq = array(yam['segment_nqpoint'])
    sq = [sum(nq[:i]) for i in range(len(nq))]
    sq.append(-1)
    xt = [qp[i] for i in sq]
    print('Found %i band labels'%len(xl), xl)
    print('     with positions', xt)
    freqs = transpose(freqs)#*THz_to_cminv
    return qp, freqs, xt, xl

if __name__ == '__main__':

    folders = ['.']
    labels = ['with NAC']
    
    figure()
    for i, folder in enumerate(folders):
        qp, freqs, xt, xl = getbands(folder)
        plot(qp, freqs*THz_to_meV, color='C%i'%i)
        axhline(0,color='C%i'%i, label=labels[i])

    #legend()
    axhline(0,color='k', linewidth=1)
    xlim(qp[0],qp[-1])
    ylim(-1,80)
    xticks(xt,xl)
    for x in xt:
        axvline(x, color='k', lw=1)
    ylabel(r'$\hbar \omega(n,\mathbf{k})$ (meV)')
    yticks(arange(0,81,10))
    tight_layout()
    savefig('bands.pdf')
    show()
