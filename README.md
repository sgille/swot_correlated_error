# swot_correlated_error
Code to test SWOT correlated error removal in an environment of westward propagating Rossby waves.

This code is designed to do 3 things.
1.  Read Copernicus (aka AVISO) sea surface height (SSH) data and project it onto a set of propagating Rossby waves.  Examine the amount of SSH variance represented by the Rossby waves, and use the projections to produce stripped down datasets that contain only propagating waves with known properties.
2.  Project the stripped down data onto SWOT ground tracks and add noise characteristic of SWOT correlated roll error (following the noise model outlined by Metref et al, 2019).
3.  Use a regularized least squares approach to solve for the Rossby wave signals and the noise either as a two-stage approach (solve for noise, remove noise, then solve for Rossby waves) or in a one stage approach (solve for noise and Rossby waves at the same time).  Do this for an ensemble of data

The details of the method are discussed in a manuscript to be submitted to J. Tech.
