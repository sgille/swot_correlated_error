# swot_correlated_error
Code to test SWOT correlated error removal in an environment of westward propagating Rossby waves.

This code is designed to achieve 3 goals.
1.  Read Copernicus (aka AVISO) sea surface height (SSH) data and project it onto a set of propagating Rossby waves.  Examine the amount of SSH variance represented by the Rossby waves, and use the projections to produce stripped down datasets that contain only propagating waves with known properties. To run this you will also need the AVISO data for the California Current region (aviso_msla_ccs_1d.nc) and two sample SWOT data files (SWOT_L2_LR_SSH_Expert_474_013_20230329T081622_20230329T090516_PIB0_01.nc and SWOT_L2_LR_SSH_Expert_474_026_20230329T191926_20230329T201032_PIB0_01.nc), all of which are too big for Github.  These files are available via Zenodo:  [https://zenodo.org/records/10963448](https://doi.org/10.5281/zenodo.10963448.)
2.  Project the stripped down data onto SWOT ground tracks and add noise characteristic of SWOT correlated roll error (following the noise model outlined by Metref et al, 2019).
3.  Use a regularized least squares approach to solve for the Rossby wave signals and the noise either as a two-stage approach (solve for noise, remove noise, then solve for Rossby waves) or in a one stage approach (solve for noise and Rossby waves at the same time).  Do this for an ensemble of data.
4.  The final step of the code makes use of the SWOT simulator, which is available here:  https://swot-simulator.readthedocs.io/en/latest/ .  Note that the SWOT simulator is no longer supported.  For the wet troposphere correction, it uses an out-of-date python function, interpolate.interp2d.  To run the swot_simulator, after installing, use the conf.py file included here:  swot_simulator conf.py --first-date 20160101  --last-date 20160210
 

The repository consists of the following routines:
STEP0_debug.ipynb:  sample code for testing and debugging.
STEP1_case_study_figures.ipynb:  Carry out goal 1 (read AVISO data and project into Rossby wave) for one case study date, and produce figures.
STEP1_extract_filtered_ssh.ipynb: Carry out goal 1 for an ensemble of monthly data files
STEP2_case_study_figures.ipynb:  Carry out goal 2 and goal 3 for one case study date and produce figures
STEP2_run_ensemble.ipynb:  Carry out goals 2 and 3 for the ensemble of monthly data files and produce ensemble statistics and figures
STEP2_simulator_run_ensemble.ipynb:  Use output from the SWOT simulator to carry out the analysis for a set of monthly input files.  

The details of the method are discussed in a manuscript submitted to J. Tech.
