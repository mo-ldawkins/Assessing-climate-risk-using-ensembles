# Note you will need to acivate the shared conda environment (climada_env_2022.yaml) and save the version of CLIMADA used in this code, available here: https://www.dropbox.com/scl/fo/inzd4vkpjrbz7sdl6c2yn/h?dl=0&rlkey=ut1qim6bh572htz0neo8lyavs 

# The data used in this code can be downloaded from here: https://www.dropbox.com/scl/fo/jyeoid411e90vwmomdfgh/h?dl=0&rlkey=7ds4dp42upwfsyytcajjldqxz 


############################################################################################################################
Carry out end-to-end risk assessment on processed data 
############################################################################################################################

1.CodeForPaperA_RiskAssessment.py

For a given data source (e.g. UKCP bias corrected), warming level (historical/current, 2deg, 4deg), SSP, SSP year associated with the warming level, vulnerability function parameters, and ensemble member, applies the CLIMADA risk quantification framework (e.g. see https://climada-python.readthedocs.io/en/stable/tutorial/1_main_climada.html). Firstly, the hazard event set (here each day is an event) is opened, and the number of years in the warming level time slice is identified. If this is less than 15, that ensemble member is dropped (the code stops). If there are 15 or more years, the hazard data is read into the CLIMADA format, exposure (UK SSP) and vulnerability information are read in/defined, and the impact of each event is calculated and summarised. Finally, the Expected Annual Impact (EAI) is saved out as a netcdf (there is also the option to save out the full impact data for all events). 

2.CodeForPaperA_GAM.py

For a given data source, associated ensemble members, warming level, SSP, SSP year associated with the warming level and vulnerability function parameters, reads in Expected Annual Impact (as output from the previous script) from all ensemble members and creates a data frame of this combined with long, lat, orography and exposure (number of jobs as used in risk calculation). The code then imports the R ‘mgcv’ package and uses this to fit a spatially varying generalised additive model for EAI with the superior functionality of this package (better than gam packages in Python) but within Python. Samples are then taken from the posterior predictive distribution of EAI based on this model. These are then saved out as a netcdf file. 

3.CodeForPaperA_Plots.py

Code to generate python plots for the paper.

Some of the code may require running externally from local PC due to memory. This is left to the user to configure.
