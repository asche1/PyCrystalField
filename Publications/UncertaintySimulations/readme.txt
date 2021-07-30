This code is to explore the Chi^2 contour of crystal field fits to simulated data.
It accompanies "Quantifying uncertainties in crystal electric field Hamiltonian fits to neutron data", ArXiv 2107.14164

There are two main directories: Pyrochlore and Delafossite. For each, the following scripts should be run in this order 
(note that the path to pythonlib may have to be adjusted):

1. SimulatedDataCode/YTO.py generates a list of .cif files based off the original crystal structure.
	This data is found in SimulatedData directory.

2. DetermineUncertainty_AllIons.py loops through each fitted parameter and tries to find the maximum and minimum values 
	based on +1 Chi^2_red from the optimum.

3. MCMC_uncertainty-step2.py performs a Monte Carlo Markov Chain for each solution found above, saving the list to 
	the \fitresults directory.

4. ProcessOutput.py generates human readable files in the /ProcessedOutput directory.

## To generate the plots found in the paper, run the following IPython notebooks:

ChiSquaredContour.ipynb    -- generates the countour plots of chi^2
GenerateTables.ipynb     -- Generates LaTex tables of the ground states for various ions
PlotResults.ipynb     -- Plots the simulated neutron spectra along with the fits with maximum, minimum, and optimum g_zz.
