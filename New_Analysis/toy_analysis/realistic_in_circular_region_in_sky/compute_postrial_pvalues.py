import numpy as np
import numpy.ma as ma
import pandas as pd

#from random import seed
from datetime import datetime

from scipy.stats import poisson
from scipy.interpolate import Akima1DInterpolator as akima_spline

#for plotting
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

from astropy.time import Time

import sys
import os
import pickle

sys.path.append('../src/')

#define a function to get a list of files from a path and a pattern
def get_filelist(input_path, pattern):

    filelist = []

    for file in os.listdir(input_path):

        filename = os.path.join(input_path, file)

        if os.path.isfile(filename) and pattern in filename:

            filelist.append(filename)

    return filelist

#function to fetch to sum the lattices of either estimators or pvalues
def merge_samples(filelist):

    #initialize list
    all_samples = []

    for file in filelist:

        #save the array with values to be summed
        with open(file, 'rb') as f:
            value_grid_per_sample = pickle.load(f)

        all_samples.append(value_grid_per_sample)

    result = np.concatenate(all_samples, axis = 3)

    return result

#get pos-trial p-values
def compute_postrial_pvalues(pvalues_flares, pvalues_iso, estimator_type):

    start = datetime.now()

    #set up dictionary to convert estimator name into an index
    estimator_dic = {'poisson' : 0, 'lambda' : 1, 'corrected_lambda' : 2}

    try:
        index = estimator_dic[estimator_type]
    except KeyError:
        print('Requested estimator name: (%s) does not exist in available keys' % estimator_type, list(estimator_dic.keys()))
        exit()

    #save the values of pvalues for easier manipulation
    pvalues_flare = pvalues_flares[:,:,index,:]
    pvalues_iso = pvalues_iso[:,:,index,:]

    postrial_pvalues = np.array([ np.sum(pvalues_iso[i, j] < pvalues_flare[i, j, :, np.newaxis], axis = 1) for i in range(pvalues_flare.shape[0]) for j in range(pvalues_flare.shape[1]) ])
    postrial_pvalues = np.reshape(postrial_pvalues, pvalues_iso.shape) / pvalues_iso.shape[2]

    #set to 1/n_samples null postrial pvalues
    min_pvalue = 1 / postrial_pvalues.shape[2]

    postrial_pvalues = ma.masked_array(postrial_pvalues, mask = (postrial_pvalues == 0)).filled(fill_value = min_pvalue)

    print('Computing postrial pvalues took', datetime.now() - start, 's')

    return postrial_pvalues

if __name__ == '__main__':

    #define the input directory
    input_path = './datasets/flare_lattice_study'

    #get lists of files with iso and (iso + flare) samples
    filelist_iso_pvalues = 'PValues_IsoDist'
    filelist_flare_pvalues = 'PValues_FlareLattice'

    filelist_iso_pvalues = get_filelist(input_path, filelist_iso_pvalues)
    filelist_flare_pvalues = get_filelist(input_path, filelist_flare_pvalues)

    #merge all samples to compute postrial pvalues
    pvalues_flares = merge_samples(filelist_flare_pvalues)
    pvalues_iso = merge_samples(filelist_iso_pvalues)

    #compute postrial pvalues
    postrial_pvalues_poisson = compute_postrial_pvalues(pvalues_flares, pvalues_iso, 'poisson')
    postrial_pvalues_lambda = compute_postrial_pvalues(pvalues_flares, pvalues_iso, 'lambda')

    #concatenate distributions of postrial pvalues
    postrial_pvalues = np.stack([postrial_pvalues_poisson, postrial_pvalues_lambda])
    postrial_pvalues = np.transpose(postrial_pvalues, axes=(1, 2, 0, 3))

    #define the name of the output files
    output_postrial_flare_pvs = os.path.basename(filelist_flare_pvalues[0])
    output_postrial_flare_pvs = 'Postrial_' + '_'.join(output_postrial_flare_pvs.split('_')[:6]) + '.pkl'

    #save the postrial distribution of files
    with open(os.path.join(input_path, output_postrial_flare_pvs), 'wb') as file:
        pickle.dump(postrial_pvalues, file)
