import numpy as np

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

    print(scale)
    print(loc)

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
def perform_fit_exp(bin_centers, bin_content, bin_error, median, sigma):

    #convert lists to arrays
    bin_centers = np.array(bin_centers)
    bin_content = np.array(bin_content)
    bin_error = np.array(bin_error)

    bin_content = bin_content[np.where(bin_centers > median + sigma)[0]]
    bin_error = bin_error[np.where(bin_centers > median + sigma)[0]]
    bin_centers = bin_centers[np.where(bin_centers > median + sigma)[0]]

    #reverse arrays
    #bin_centers = bin_centers[::-1]
    #bin_content = bin_content[::-1]
    #bin_error = bin_error[::-1]

    #upper_edge = bin_centers[0]

    #for upper_edge in bin_centers:

    #restrict to values below upper_edge
    #bin_content = bin_content[np.where(bin_centers < upper_edge)[0]]
    #bin_error = bin_error[np.where(bin_centers < upper_edge)[0]]
    #bin_centers = bin_centers[np.where(bin_centers < upper_edge)[0]]

    #print(bin_centers)

    #restrict bin contents to non-zero values
    bin_centers = bin_centers[np.where(bin_content > 0)[0]] #[bin_centers[i] for i in range(len(bin_content)) if bin_content[i] > 0]
    bin_error = bin_error[np.where(bin_content > 0)[0]] #[bin_error[i] for i in range(len(bin_content)) if bin_content[i] > 0]
    bin_content = bin_content[np.where(bin_content > 0)[0]] #[bin_content[i] for i in range(len(bin_content)) if bin_content[i] > 0]

    #initial guess for parameters
    params_init = [max(bin_content), .5]

    #bounds for parameters
    lower_bounds = [0, 0]
    upper_bounds = [10*max(bin_content), 1]

    #perform fit
    popt, pcov = curve_fit(exp_fit, bin_centers, bin_content, p0=params_init, bounds=(lower_bounds, upper_bounds), sigma=bin_error)

    #errors of parameters
    perr = np.sqrt(np.diag(pcov))

    #produce arrays to plot the fit function
    x = np.linspace(min(bin_centers), max(bin_centers), 5000)
    y = exp_fit(x, *popt)

    #compute chi2
    ndf = len(bin_content) - len(popt)
    y_exp = np.array(exp_fit(bin_centers, *popt))
    chi2 = sum(np.power(y_exp - bin_content, 2) / np.power(bin_error, 2)) / ndf

    #print(chi2)

        #if chi2 < 2:
            #break

    return [popt, perr, x, y, chi2]
