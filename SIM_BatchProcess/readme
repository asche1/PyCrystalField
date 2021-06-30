# Batch processing multiple single ion magnet cif files and simulating the ground state anisotropy using PyCrystalField.
## Allen Scheie, June 2021

To process the cif files, run "PCF_BatchProcessCIF.py". 
It generates the output in "calculatedOutput.tsv", a set of shape files in the /shapefiles subdirectory, 
and a set of truncated .cif files (also in the /shapefiles subdirectory) giving the ligand environment 
and orientation used for the calculation.

To add .cif files to the list, add them to the CIFs-initset directory, also adding the necessary info
to "Data_Selected_SIM_samples.tsv".

Note that this was coded on a Linux machine, so the function call to SHAPE (line 128 in 
lib/BatchProcessFunctions.py) may need appropriate editing so it will run on a different operating system.
