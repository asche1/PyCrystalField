# Code to calculate the crystal field Hamiltonians of KErSe2 and CsErSe2

Allen Scheie

January, 2020


## Description

This code is to accompany the paper "Crystal field Hamiltonian and anisotropy in KErSe2 and CsErSe2", currently under review.
In it, we use PyCrystalField to fit the crystal field Hamiltonian of Er3+ in these two materials

### Structure

Begin with AES_CEF_FitPC.ipynb (A= K, C), which simulates a point-charge model and uses that to fit the data.
Then run AES_CEF_fit2.ipynb, which uses the output of the opposite compound to re-fit.
Then, run PlotFinalFits.ipynb, which generates plots the results used for the paper.

To calculate uncertainty, run AES_CEF_loopthgouth.ipynb, then AES_CEF_loopthrough-PlotResults.ipynb.

### Notes

In order to run these scripts, the path PyCrystalField must be properly specified in the first cell.

We also include several data folders, and "pythonlib", which has a class used to import and plot the neutron scattering data. Also note that some of the notebooks depend on plotformat.py for properly formatting the plots.
