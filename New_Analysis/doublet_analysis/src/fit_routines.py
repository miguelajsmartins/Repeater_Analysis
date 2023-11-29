import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.special import erfc

#defines the fit function
def fit_expGauss(x, norm, mean, sigma, slope):

    return norm*np.exp((x - mean)/slope + 0.5*(sigma / slope)**2)*erfc((x - mean)/(np.sqrt(2)*sigma))

#defines the gumble distribution
def gumble_fit(x, norm, mean, scale, alpha):

    z = (mean - x) / scale

    return norm*np.exp(z - alpha*np.exp(z))

#defines exponential function
def exp_fit(x, scale, slope):
    return scale*np.exp(-slope*x)

#define the most suitible fit function based on pdf of delta t
def fit_function(x, norm, mean, sigma, gamma):

    z = (mean - x) / sigma

    return norm*np.exp(z - gamma*np.exp(z))

#performs the fit and outputs the graph of the fit function
def perform_fit_expGauss(bin_centers, bin_content, bin_error, params_init, lower_bounds, upper_bounds):

    #restrict bin contents to non-zero values
    bin_centers = np.array([bin_centers[i] for i in range(len(bin_content)) if bin_content[i] > 0])
    bin_error = np.array([bin_error[i] for i in range(len(bin_content)) if bin_content[i] > 0])
    bin_content = np.array([bin_content[i] for i in range(len(bin_content)) if bin_content[i] > 0])

    #perform fit
    popt, pcov = curve_fit(fit_expGauss, bin_centers, bin_content, p0 = params_init, bounds=(lower_bounds, upper_bounds), sigma=bin_error)

    #errors of parameters
    perr = np.sqrt(np.diag(pcov))

    #produce arrays to plot the fit function
    x = np.linspace(min(bin_centers), max(bin_centers), 5000)
    y = fit_expGauss(x, *popt)

    #compute chi2
    ndf = len(bin_content) - len(popt)
    y_exp = np.array(fit_expGauss(bin_centers, *popt))
    chi2 = sum(np.power(y_exp - bin_content, 2) / np.power(bin_error, 2)) / ndf

    return popt, perr, x, y, chi2

#performs the fit and outputs the graph of the fit function
def perform_fit_gumble(bin_centers, bin_content, bin_error, mean, sigma):

    #convert lists to arrays
    bin_centers = np.array(bin_centers)
    bin_content = np.array(bin_content)
    bin_error = np.array(bin_error)

    #restrict bin contents to non-zero values
    bin_centers = bin_centers[np.where((bin_content > 0))[0]] #[bin_centers[i] for i in range(len(bin_content)) if bin_content[i] > 0]
    bin_error = bin_error[np.where((bin_content > 0))[0]] #[bin_error[i] for i in range(len(bin_content)) if bin_content[i] > 0]
    bin_content = bin_content[np.where((bin_content > 0))[0]] #[bin_content[i] for i in range(len(bin_content)) if bin_content[i] > 0]

    #defines the fit initial values
    scale = ( np.sqrt(6) / np.pi )*sigma
    loc = mean - .51*scale

    params_init = [max(bin_content), loc - .5, scale, 1] #loc, scale]

    #defines the lower and upper bounds
    lower_bounds = [.01*max(bin_content), 1e-2, 1, 0]
    upper_bounds = [10*max(bin_content), 1, 10*scale, 20]

    #perform fit
    popt, pcov = curve_fit(gumble_fit, bin_centers, bin_content, p0 = params_init, bounds=(lower_bounds, upper_bounds), sigma=bin_error)

    #errors of parameters
    perr = np.sqrt(np.diag(pcov))

    #produce arrays to plot the fit function
    x = np.linspace(min(bin_centers), max(bin_centers), 5000)
    y = gumble_fit(x, *popt)

    #compute chi2
    ndf = len(bin_content) - len(popt)
    y_exp = np.array(gumble_fit(bin_centers, *popt))
    chi2 = sum(np.power(y_exp - bin_content, 2) / np.power(bin_error, 2)) / ndf

    return popt, perr, x, y, chi2

#performs the fit and outputs the graph of the fit function
def perform_fit_exp(bin_centers, bin_content, bin_error, initial_point):

    #convert lists to arrays
    bin_centers = np.array(bin_centers)
    bin_content = np.array(bin_content)
    bin_error = np.array(bin_error)

    #if all bin contents are 0, then skip
    if np.all(bin_content == 0):

        print('Empty bins!.')
        dummy_array = np.empty(10)
        dummy_array.fill(np.nan)

        return [dummy_array, dummy_array, dummy_array, dummy_array, dummy_array[0]]

    else:

        bin_content = bin_content[np.where(bin_centers > initial_point)[0]]
        bin_error = bin_error[np.where(bin_centers > initial_point)[0]]
        bin_centers = bin_centers[np.where(bin_centers > initial_point)[0]]

        #restrict bin contents to non-zero values
        bin_centers = bin_centers[np.where(bin_content > 0)[0]]
        bin_error = bin_error[np.where(bin_content > 0)[0]]
        bin_content = bin_content[np.where(bin_content > 0)[0]]

        #get bin width
        bin_width = bin_centers[1] - bin_centers[0]

        for lower_limit in bin_centers:

            #above lower_limit
            above_lower_lim = bin_centers > lower_limit

            fit_bin_centers = bin_centers[above_lower_lim]
            fit_bin_content = bin_content[above_lower_lim]
            fit_bin_error = bin_error[above_lower_lim]

            #only consider bins such that there are no gaps in the bin centers
            #no_gaps = np.diff(fit_bin_centers) == bin_width

            #fit_bin_centers = bin_centers[no_gaps]
            #fit_bin_content = bin_content[no_gaps]
            #fit_bin_error = bin_error[no_gaps]

            #remove outliers
            #if np.any(np.diff(fit_bin_centers)) > bin_width):

                #fit_bin_centers = fit_bin_centers[:-1]
                #fit_bin_content = fit_bin_content[:-1]
                #fit_bin_error = fit_bin_error[:-1]

                #print('removed outlier bin_centers', fit_bin_centers)
                #print('removed outlier bin_content', fit_bin_content)

            #initial guess for parameters
            slope = np.log(max(fit_bin_content) / fit_bin_content[-1]) / (fit_bin_centers[-1] - fit_bin_centers[bin_content.argmax()])
            norm = fit_bin_content[0]*np.exp(slope*fit_bin_centers[1])

            params_init = [norm, slope]

            #bounds for parameters
            lower_bounds = [0, 0]
            upper_bounds = [10*norm, 2*slope]

            #print('removed outlier bin_centers', fit_bin_centers[-10:])
            #print('removed outlier bin_content', fit_bin_content[-10:])

            #perform fit
            popt, pcov = curve_fit(exp_fit, fit_bin_centers, fit_bin_content, p0=params_init, bounds=(lower_bounds, upper_bounds), sigma=fit_bin_error)

            #errors of parameters
            perr = np.sqrt(np.diag(pcov))

            #produce arrays to plot the fit function
            x = np.linspace(min(fit_bin_centers), max(fit_bin_centers), 5000)
            y = exp_fit(x, *popt)

            #compute chi2
            ndf = len(fit_bin_content) - len(popt)
            y_exp = np.array(exp_fit(fit_bin_centers, *popt))
            chi2 = sum(np.power(y_exp - fit_bin_content, 2) / np.power(fit_bin_error, 2)) / ndf

            if chi2 < 2:
                print('fit converged!')
                break

        return [popt, perr, x, y, chi2]

#performs the fit to the lambda distribution
def perform_fit_modified_gumble(bin_centers, bin_content, bin_error, params_init): #, lower_bounds, upper_bounds):

    #convert lists to arrays
    bin_centers = np.array(bin_centers)
    bin_content = np.array(bin_content)
    bin_error = np.array(bin_error)

    #restrict bin contents to non-zero values
    bin_centers = bin_centers[np.where((bin_content > 0))[0]] #[bin_centers[i] for i in range(len(bin_content)) if bin_content[i] > 0]
    bin_error = bin_error[np.where((bin_content > 0))[0]] #[bin_error[i] for i in range(len(bin_content)) if bin_content[i] > 0]
    bin_content = bin_content[np.where((bin_content > 0))[0]] #[bin_content[i] for i in range(len(bin_content)) if bin_content[i] > 0]

    #perform fit
    popt, pcov = curve_fit(fit_function, bin_centers, bin_content, p0 = params_init, sigma=bin_error)

    #errors of parameters
    perr = np.sqrt(np.diag(pcov))

    #produce arrays to plot the fit function
    x = np.linspace(min(bin_centers), max(bin_centers), 5000)
    y = gumble_fit(x, *popt)

    #compute chi2
    ndf = len(bin_content) - len(popt)
    y_exp = np.array(fit_function(bin_centers, *popt))
    chi2 = sum(np.power(y_exp - bin_content, 2) / np.power(bin_error, 2)) / ndf

    return popt, perr, x, y, chi2
