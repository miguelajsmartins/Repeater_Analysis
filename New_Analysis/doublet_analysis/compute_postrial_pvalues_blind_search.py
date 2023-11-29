import pandas as pd
import numpy as np

from datetime import datetime

import os
import sys

sys.path.append('./src/')

#compute probability
def get_postrial_p_value(p_value, array_p_value):

    if np.isnan(p_value):
        return np.nan

    else:
        #define region below p_value
        below_p_value = array_p_value <= p_value

        #compute postrial pvalue
        postrial_p_value = len(array_p_value[below_p_value])/ len(array_p_value)

        return postrial_p_value

#compute the distribution of p-values for each target
def compute_p_value_dist_per_target(filelist, flare_file):

    #since all skies are binned in the same way, save 1 dataframe
    flare_data = pd.read_parquet(flare_file, engine='fastparquet')

    #loop over all files and save distributions of p_values
    poisson_p_value_list = []
    lambda_p_value_list = []

    for file in filelist:

        #save data
        data = pd.read_parquet(file, engine='fastparquet')

        #save columns with poisson and lambda_p_values for each target
        poisson_p_value_list.append(data['poisson_p_value'].to_numpy())
        lambda_p_value_list.append(data['lambda_p_value'].to_numpy())

    #convert lists into arrays
    poisson_p_value_array = np.array(poisson_p_value_list)
    lambda_p_value_array = np.array(lambda_p_value_list)

    #transpose arrays
    poisson_p_value_dist_per_target = np.transpose(poisson_p_value_array)
    lambda_p_value_dist_per_target = np.transpose(lambda_p_value_array)

    #save this arrays in data from flare
    flare_data['poisson_postrial_p_value'] = flare_data.apply(lambda x: get_postrial_p_value(x['poisson_p_value'], poisson_p_value_dist_per_target[x.name]), axis = 1)
    flare_data['lambda_postrial_p_value'] = flare_data.apply(lambda x: get_postrial_p_value(x['lambda_p_value'], lambda_p_value_dist_per_target[x.name]), axis = 1)

    return flare_data

#read file list of all events with isotropic distributions
input_path = './datasets/estimators_binned_sky'
filelist = []

for file in os.listdir(input_path):

    file = os.path.join(input_path, file)

    if os.path.isfile(file):
        filelist.append(file)

#-------------------
# for flare
#-------------------
#checks if datafile was provided
if len(sys.argv) < 2:
    print('No datafile was provided!')
    exit()

#save file with flare data
flare_file = sys.argv[1]

#define name of output file
output_path = os.path.dirname(flare_file)
basename = os.path.basename(flare_file)
output_name = os.path.join(output_path, 'BlindSearch_PosTrial_' + basename)

start_time = datetime.now()

flare_data = compute_p_value_dist_per_target(filelist, flare_file)

print('Computing post trial p-values took', datetime.now() - start_time, 's')

print(flare_data.describe())
print(flare_data['poisson_postrial_p_value'].value_counts())
print(flare_data['lambda_postrial_p_value'].value_counts())

#save file with the new data
flare_data.to_parquet(output_name, index=True)

#-------------------------
# for isotropic distributions
#-------------------------
# output_path = os.path.join(input_path, 'BlindSearch_PosTrial')
#
# for i, file in enumerate(filelist):
#
#     #define the output file name
#     basename = os.path.basename(file)
#     output_name = os.path.join(output_path, 'BlindSearch_PosTrial_' + basename)
#
#     #compute the postrial p-values
#     start_time = datetime.now()
#
#     data = compute_p_value_dist_per_target(filelist, file)
#
#     print('Computing post trial p-values took', datetime.now() - start_time, 's for event', i, '/', len(filelist))
#
#     #save file with the new data
#     data.to_parquet(output_name, index=True)
