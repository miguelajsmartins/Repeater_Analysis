import numpy as np

#compute bin centers from bin edges
def get_bin_centers(bin_edges):
    return np.array([(bin_edges[i-1] + bin_edges[i])/ 2 for i in range(1, len(bin_edges))])

#compute bin centers from bin edges
def get_bin_width(bin_edges):

    bin_edges = np.array(bin_edges)

    bin_widths = np.diff(bin_edges)

    if len(np.unique(bin_widths)) == 1:
        return bin_widths[0]
    else:
        return bin_widths

#WARNING: the error is NOOOTTTT being well computed in case of weighted bins
#plot error bar from 1d numpy array or series
def data_2_binned_errorbar(data, nbins, bin_lower, bin_upper, weights, is_density):

    #convert data into numpy array
    bin_content, bin_edges = np.histogram(data, bins = nbins, range = [bin_lower, bin_upper], density=is_density, weights=weights)

    #compute bin centers
    bin_centers = [(bin_edges[i] + bin_edges[i-1]) / 2 for i in range(1, len(bin_edges))]

    #compute error whether density is being used or not
    if(is_density):
        bin_error = np.sqrt(bin_content) / np.sqrt(len(data))
    else:
        bin_error = np.sqrt(bin_content)

    return bin_centers, bin_content, bin_error

#plot error bar from 1d numpy array or series
def data_2_binned_content(data, nbins, bin_lower, bin_upper, weights, is_density):

    #convert data into numpy array
    bin_content, bin_edges = np.histogram(data, bins = nbins, range = [bin_lower, bin_upper], density=is_density, weights=weights)

    #compute bin centers
    bin_centers = [(bin_edges[i] + bin_edges[i-1]) / 2 for i in range(1, len(bin_edges))]

    #compute error whether density is being used or not
    if(is_density):
        bin_error = np.sqrt(bin_content) / np.sqrt(len(data))
    else:
        bin_error = np.sqrt(bin_content)

    return bin_content

#plot plot bar from 1d numpy array or series
def hist_2_plot(data, nbins, bin_lower, bin_upper, is_density):

    #convert data into numpy array
    bin_content, bin_edges = np.histogram(data, bins = nbins, range = [bin_lower, bin_upper], density=is_density)

    #compute bin centers
    bin_centers = [(bin_edges[i] + bin_edges[i-1]) / 2 for i in range(1, len(bin_edges))]

    return bin_centers, bin_content
