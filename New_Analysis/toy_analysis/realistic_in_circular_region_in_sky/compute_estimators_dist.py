import pandas as pd
import numpy as np

import os
import sys

sys.path.append('../src/')

#compute the average lambda distribution
def get_lambda_dist_per_rate(filelist):

    #since all the files have the same binning, compute the min and max expected number of events
    mu_per_target = np.unique(pd.read_parquet(filelist[0], engine = 'fastparquet')['mu_in_target'])
    mu_min = np.floor(np.min(mu_per_target)).astype('int')
    mu_max = np.ceil(np.max(mu_per_target)).astype('int')

    #define the binning in the number of expected events
    mu_bins = np.append(np.arange(mu_min, mu_max, .5), mu_max)

    print(mu_bins)

    #define the binning in lambda
    lambda_bins = np.append(np.arange(-15, 100, 1), 100)
    lambda_corrected_bins = np.append(np.arange(-30, 80, 1), 80)

    #save the contents of the lambda distributions
    lambda_content_per_mu_list = []
    corrected_lambda_content_per_mu_list = []

    for file in filelist:

        #save relevant columns of dataframe
        data = pd.read_parquet(file, engine = 'fastparquet')

        #save the relevant columns from the dataframe
        mu_per_target = data['mu_in_target'].to_numpy()
        lambda_per_target = data['lambda'].to_numpy()
        lambda_corrected_per_target = data['lambda_corrected'].to_numpy()

        #sort arrays according to mu
        sorted_indices = mu_per_target.argsort()
        mu_per_target, lambda_per_target, lambda_corrected_per_target = mu_per_target[sorted_indices], lambda_per_target[sorted_indices], lambda_corrected_per_target[sorted_indices]

        #split arrays of estimators according to expected number of events in target
        split_indices = np.searchsorted(mu_per_target, mu_bins)[1:-1]
        lambda_per_mu = np.split(lambda_per_target, split_indices)
        corrected_lambda_per_mu = np.split(lambda_corrected_per_target, split_indices)

        mean_lambda = [len(array) for array in lambda_per_mu]

        #print(mu_per_target)
        #print(lambda_per_mu[0])

        #compute the bin contents of the lambda distribution
        lambda_bin_content_per_mu = np.array([np.histogram(array, bins = lambda_bins)[0] for array in lambda_per_mu])
        lambda_corrected_bin_content_per_mu = np.array([np.histogram(array, bins = lambda_corrected_bins)[0] for array in corrected_lambda_per_mu])

        #save lambda bin contents
        lambda_content_per_mu_list.append(lambda_bin_content_per_mu)
        corrected_lambda_content_per_mu_list.append(lambda_corrected_bin_content_per_mu)

    #compute average lambda distribution
    lambda_content_per_mu_list = np.array(lambda_content_per_mu_list)
    corrected_lambda_content_per_mu_list = np.array(corrected_lambda_content_per_mu_list)

    lambda_content_per_mu = np.sum(lambda_content_per_mu_list, axis = 0)
    corrected_lambda_content_per_mu = np.sum(corrected_lambda_content_per_mu_list, axis = 0)

    #build dataframes with lambda distributions
    column_names = ['mu_low_edges', 'mu_upper_edges', 'lambda_bin_edges', 'lambda_bin_content']
    lambda_dist_data = pd.DataFrame(zip(mu_bins[:-1], mu_bins[1:], np.tile(lambda_bins, (len(mu_bins[1:]), 1)), lambda_content_per_mu), columns= column_names)
    corrected_lambda_dist_data = pd.DataFrame(zip(mu_bins[:-1], mu_bins[1:], np.tile(lambda_corrected_bins, (len(mu_bins[1:]), 1)), corrected_lambda_content_per_mu), columns= column_names)

    return lambda_dist_data, corrected_lambda_dist_data

#define the main function
if __name__ == '__main__':

    #define important quantities
    dec_center = np.radians(-30)
    ra_center = np.radians(0)
    patch_radius = np.radians(25)
    target_radius = np.radians(1.5)

    #define the input path
    input_path = './datasets/iso_estimators/decCenter_%.0f' % np.degrees(dec_center)
    file_substring = 'targetRadius_%.1f_IsoDist_decCenter_%.0f_raCenter_%.0f_patchRadius_%.0f' % (np.degrees(target_radius), np.degrees(dec_center), np.degrees(ra_center), np.degrees(patch_radius))

    #initialize list to hold files with lambda for each target
    filelist = []

    # loop over files in the directory
    for file in os.listdir(input_path):

        filename = os.path.join(input_path, file)

        if os.path.isfile(filename) and file_substring in filename: # and 'Scrambled' not in f:

            filelist.append(filename)

    #print warning if no files found
    if len(filelist) == 0:
        print('No files found!')
        exit()

    #compute the average lambda distribution for two lambda definitions and per bin in expected number of events
    lambda_dist_data, corrected_lambda_dist_data = get_lambda_dist_per_rate(filelist)

    #print both dataframes with lambda distributions
    print(lambda_dist_data)
    print(corrected_lambda_dist_data)

    #define the output name of the files
    output_path = './datasets/lambda_dist'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_lambda_dist = 'Lambda_dist_patchRadius_%.0f_targetRadius_%.1f.json' % (np.degrees(patch_radius), np.degrees(target_radius))
    output_corrected_lambda_dist = 'Corrected_Lambda_dist_patchRadius_%.0f_targetRadius_%.1f.json' % (np.degrees(patch_radius), np.degrees(target_radius))

    lambda_dist_data.to_json(os.path.join(output_path, output_lambda_dist), index = True)
    corrected_lambda_dist_data.to_json(os.path.join(output_path, output_corrected_lambda_dist), index = True)
