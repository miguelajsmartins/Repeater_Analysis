import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.special import erfc

from scipy.stats import poisson

from datetime import datetime

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
def perform_chi2_fit_exp(bin_centers, bin_content, bin_error, initial_point):

    start = datetime.now()

    #save the lowest and highest initial points
    lower_initial_point = initial_point[0]
    upper_initial_point = initial_point[1]

    #convert lists to arrays
    bin_centers = np.array(bin_centers)
    bin_content = np.array(bin_content)
    bin_error = np.array(bin_error)

    #save array with initial points for fit
    fit_initial_points = bin_centers[np.logical_and(bin_centers >= lower_initial_point, bin_centers <= upper_initial_point)]

    #compute the fit initial values for the parameters
    pdf_at_lower_initial_point = bin_content[bin_centers == lower_initial_point]
    pdf_at_upper_initial_point = bin_content[bin_centers == upper_initial_point]

    slope = np.log(pdf_at_lower_initial_point / pdf_at_upper_initial_point) / (upper_initial_point - lower_initial_point)
    slope = slope[0]
    norm = np.sum(bin_content)

    params_init = [norm, slope]

    #print(params_init)

    lower_bounds = [1, .2*slope]
    upper_bounds = [10*norm, 2*slope]

    #save arrays of fit parameters and initial point
    fit_parameters = []
    chi2_list = []

    #if all bin contents are 0, then skip
    if np.all(bin_content == 0):

        print('Fit cannot be performed. All bins are empty!')
        return np.full(5, np.nan)

    else:

        for initial_point in fit_initial_points:

            #restrict bins to be above the initial point
            above_initial_point = bin_centers >= initial_point

            fit_bin_centers = bin_centers[above_initial_point]
            fit_bin_content = bin_content[above_initial_point]
            fit_bin_error = bin_error[above_initial_point]

            #eliminate entries such that the bin content is null, since curve fit cannot compute residuals in that case
            no_null_content = fit_bin_content > 0

            fit_bin_centers = fit_bin_centers[no_null_content]
            fit_bin_content = fit_bin_content[no_null_content]
            fit_bin_error = fit_bin_error[no_null_content]

            #perform fit
            popt, pcov = curve_fit(exp_fit, fit_bin_centers, fit_bin_content, p0=params_init, bounds=(lower_bounds, upper_bounds), sigma=fit_bin_error)

            #errors of parameters
            perr = np.sqrt(np.diag(pcov))

            #compute chi2
            ndf = len(fit_bin_content) - len(popt)
            y_exp = np.array(exp_fit(fit_bin_centers, *popt))
            chi2 = sum(np.power(y_exp - fit_bin_content, 2) / np.power(fit_bin_error, 2)) / ndf

            #save the values of fit parameters and corresponding normalized likelihood
            fit_parameters.append(popt)
            chi2_list.append(chi2)

        #transform lists into arrays
        fit_parameters = np.array(fit_parameters)
        chi2_array = np.array(chi2_list)

        print('Chi2 fitting procedure took', datetime.now() - start, 's')

        return fit_parameters, chi2_array

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

#define the poisson likelihood for a likelihood fit
def poisson_log_likelihood(fit_parameters, bin_centers, bin_content):

    #save the fit parameters
    norm = fit_parameters[0]
    slope = fit_parameters[1]

    #compute the predicted bin content
    prediction = exp_fit(bin_centers, norm, slope)

    #compute the log of the p.m.f
    log_prob = poisson.logpmf(bin_content, prediction)
    log_perfect_prob = poisson.logpmf(bin_content, bin_content)

    #computes the log - likelihood
    loglike = - (np.sum(log_prob) - np.sum(log_perfect_prob))

    return loglike

#define the likelihood fit procedure
def perform_likelihood_fit_exp(bin_centers, bin_content, bin_error, initial_point):

    start = datetime.now()

    #convert lists to arrays
    bin_centers = np.array(bin_centers)
    bin_content = np.array(bin_content)
    bin_error = np.array(bin_error)

    #compute the fit initial values for the parameters
    is_positive = bin_content > 0

    pdf_at_initial_point = bin_content[bin_centers == initial_point]
    pdf_at_end_point = bin_content[is_positive][-1]

    slope = np.log(pdf_at_initial_point / pdf_at_end_point) / (initial_point - bin_centers[is_positive][-1])
    norm = np.sum(bin_content)

    params_init = [norm, slope[0]]
    params_bounds = [(1, 10*norm), (0, 1)]

    #if all bin contents are 0, then skip
    if np.all(bin_content == 0):

        print('Fit cannot be performed. All bins are empty!')
        return np.full(5, np.nan)

    else:

        for fit_initial_point in bin_centers[bin_centers >= initial_point]:

            #print(fit_initial_point)

            #restrict bins to be above the initial point
            above_initial_point = bin_centers >= fit_initial_point

            fit_bin_centers = bin_centers[above_initial_point]
            fit_bin_content = bin_content[above_initial_point]
            fit_bin_error = bin_error[above_initial_point]

            #minimize the loglikelihood
            minimized_likelihood = minimize(poisson_log_likelihood, args=(fit_bin_centers, fit_bin_content), x0 = params_init, bounds = params_bounds)

            #save the fit parameters
            popt = minimized_likelihood.x
            pcov = minimized_likelihood.hess_inv.matmat(np.identity(2))
            perr = np.sqrt(np.diag(pcov))
            loglike = minimized_likelihood.fun

            #compute number of degrees of freedom and normalize the value of the loglikelihood
            ndf = len(fit_bin_content) - len(popt)
            loglike = loglike / ndf

            #produce arrays to plot the fit function
            x = np.linspace(min(fit_bin_centers), max(fit_bin_centers), 5000)
            y = exp_fit(x, *popt)

            #compute chi2 for the bins that are non-zero
            is_positive = fit_bin_content > 0

            y_exp = np.array(exp_fit(fit_bin_centers, *popt))
            chi2 = np.sum(np.power(y_exp[is_positive] - fit_bin_content[is_positive], 2) / np.power(fit_bin_error[is_positive], 2)) / ndf

            if minimized_likelihood.success:
                break

        print('Fit successful! Fitting procedure took', datetime.now() - start, 's')

        return [popt, perr, x, y, chi2]
