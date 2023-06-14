import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import sys

sys.path.append('./src/')

from hist_manip import data_2_binned_errorbar

#enable latex rendering of formulas
plt.rcParams.update({
    'text.usetex' : 'True',
    'font.family' : 'serif'
})

def directional_exposure_dec_func(list_of_files):

    bin_contents_list = []
    lower_error_band = []
    upper_error_band = []

    #loop over files
    for i, file in enumerate(list_of_files):
        data = pd.read_parquet(file, engine='fastparquet')

        bin_centers, bin_contents, bin_error = data_2_binned_errorbar(np.sin(np.radians(data['dec'])), 100, -1, 1, True)

        #plt.plot(bin_centers, bin_contents)
        #plt.show()

        if i == 0:
            bin_centers_fix = bin_centers

        bin_contents_list.append(bin_contents)

    #compute average directional exp as a function of sin(dec)
    average_bin_content = np.mean(bin_contents_list, axis = 0)

    #compute bands corresponding to 1 sigma
    fluctuations = np.std(bin_contents_list, axis = 0)

    lower_error_band = average_bin_content - fluctuations
    upper_error_band = average_bin_content + fluctuations

    return bin_centers_fix, average_bin_content, lower_error_band, upper_error_band


#save names of files containing events
path_to_files = './datasets/'
file_list = []

# Loop over files in the directory
for filename in os.listdir(path_to_files):

    f = os.path.join(path_to_files, filename)

    if os.path.isfile(f) and 'UniformDist' in f: # and 'Scrambled' not in f:

        file_list.append(f)

#save the pdf of the directional exposure for each bin in sin(dec)
bin_centers, bin_content, lower_band, upper_band = directional_exposure_dec_func(file_list)

plt.plot(bin_centers, bin_content)
plt.fill_between(bin_centers, lower_band, upper_band, alpha = .5)

plt.show()
