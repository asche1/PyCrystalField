# Import libraries
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from matplotlib import cm
from matplotlib.colors import ListedColormap

#Plot Formatting stuff
#########################################################################
#mpl.style.use('default')
#import seaborn.apionly as sns
#cpal1 = sns.choose_colorbrewer_palette('qualitative')
from cycler import cycler

cpal1 = plt.cm.Set1(np.arange(9))
cpal2 = plt.cm.tab10(np.arange(10))
cpal3 = plt.cm.Set2(np.arange(8))
cpalpaired = plt.cm.Paired(np.arange(12))

## Tone down the yellow (too intense!)
cpal1[5] *= np.array([0.77,0.77,0.77,1])

def my_formatter(x, pos):
    """Format 0.0 as 0"""
    if x == 0: return '{:g}'.format(x)
    else: return x

params = {'text.usetex': False, 'mathtext.fontset': 'stixsans',
          'xtick.direction':'in', 'ytick.direction': 'in',
          'xtick.top': True,'ytick.right': True,
          'font.size': 15, 'axes.prop_cycle': cycler('color',cpal1)}
plt.rcParams.update(params)

################# my custom cmaps

from colormaps.BGY_cmap_01 import BGY_cm_1
from colormaps.BGY_cmap_02 import BGY_cm_2
from colormaps.parula import parula

hsv = cm.get_cmap('hsv', 256)
newcolors = hsv(np.linspace(0.91, 0, 256))
highn = 65
for i in range(highn):
    black = np.array([0,0,0,1])
    r = i/highn
    newcolors[i, :] = r*newcolors[i,:] + (1-r)*black
RaduCmp = ListedColormap(newcolors)



############################################################################

def replace_zeros(only_y = False):
    """ Replaces 0.0 with 0 for all axes of figure...
    E.g.::
        f, ax = figure()
        replace_zeros()
    :param f: matplotlib figure.
    """
    f=plt.gcf()
    f.canvas.draw()
    ax = f.get_axes()
    for ii, k in enumerate(ax):
        xlabels = [item.get_text() for item in k.get_xticklabels()]
        ylabels = [item.get_text() for item in k.get_yticklabels()]
        if not k.get_xscale() == 'log':
            for i, j in enumerate(xlabels):
                if not j == '':
                    if float(j.replace(u'\u2212', '-')) == 0:
                        xlabels[i] = '0'
        if not k.get_yscale() == 'log':
            for i, j in enumerate(ylabels):
                if not j == '':
                    if float(j.replace(u'\u2212', '-')) == 0:
                        ylabels[i] = '0'
        if not only_y:
            k.set_xticklabels(xlabels) 
        if (ii== 0) or (k.get_shared_y_axes().joined(k,ax[0])==False):  
            k.set_yticklabels(ylabels)

def reset_cpal(cpal):
    cpals = [cpal1, cpal2, cpal3]
    params = {'axes.prop_cycle': cycler('color',cpals[cpal])}
    plt.rcParams.update(params)


def adjustlightness(color, value, decrease=True):
    if decrease:
        return color*np.array([value, value, value, 1])
    else:
        return (1-value)*color + value*np.array([1,1,1,1])


subplotlabels = 'abcdefghijklmnopqrstuvwxyz'
def parSubPlotLabel(i):
    return '('+subplotlabels[(i)%26]*int(1+(i)/26)+')'
def SubPlotLabel(i):
    return subplotlabels[(i)%26]*int(1+(i)/26)