import math
import numpy as np
import healpy as hp
from healpy.newvisufunc import projview

#for operating system
import os

#for plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc

#to import and work with data frames
import pandas as pd

#to work with time and angular coordinates in the celeastial sphere
from astropy.time import Time

#for statistics
from scipy import stats
from scipy.optimize import curve_fit
from scipy.stats import chisquare
from scipy.stats import norm
from scipy.special import erf

rc('font',**{'family':'serif','serif':['DejaVu Sans']})
rc('text', usetex=True)

#define the cumulative function of the normal distribution
def NormalComulative(x, mu, sigma, norm, shift):

    return shift + norm*( 1 + erf( (x - mu)/(np.sqrt(2)*sigma)))
    
#convert a set of pandas columns into a list of event
def pandas_to_event(df, str1 , str2, str3, str4):

    alpha = np.radians(df[str1].to_numpy())
    delta = np.radians(df[str2].to_numpy())
    time = df[str3].to_numpy()
    energy = df[str4].to_numpy()

    evt_list = []

    for i in range(len(alpha)):
        evt_list.append([alpha[i],delta[i],time[i],energy[i]])

    return evt_list

#convert equatorial coordinates of event into healpy coordinates
def Healpy_Event(evt_list):

    new_evt_list = []

    for evt in evt_list:
        evt[1] = math.pi/2 - evt[1]
        new_evt_list.append(evt)


    return new_evt_list

#convert equatorial coordinates into healpy coordinates
def ToHealpyCoordinates(alpha, dec):

    return alpha, np.pi/2 - dec

#convert healpy coordinates into Equatorial Coordinates
def ToEquatorialCoordinates(phi, theta):

    return phi, np.pi/2 - theta

#convert parquet to healpy map
def FileToHealpyMap(filename, df_columns, NSIDE):

    data = pd.read_parquet(filename, engine = 'fastparquet')

    #save right ascencion and declination
    ra_vec = data[df_columns[0]].to_numpy()
    dec_vec = np.pi/2 - data[df_columns[1]].to_numpy()

    #vector to save sky indexes
    hp_indexes = []

    #for pos in coordinates:
    hp_indexes.append(hp.ang2pix(NSIDE, dec_vec, ra_vec))

    #create vectors with zeros and length equal to number of pixels
    skymap = np.zeros(hp.nside2npix(NSIDE))

    #at the indexes specified earlier, place the corresponding number of events
    np.add.at(skymap, tuple(hp_indexes), 1)

    return skymap

#defines the top-hat function for the computation of the LiMa significance
def top_hat_beam(radius, NSIDE):

    b = np.linspace(0.0, np.pi, 10000)
    bw = np.where(abs(b) <= radius, 1, 0)
    return hp.sphtfunc.beam2bl(bw, b, lmax=NSIDE*3) #beam in the spherical harmonics space

def TopHatSmoothedMap(hp_map, radius, NSIDE):

    solid_angle = 2.*np.pi*(1. - np.cos(radius))

    return hp.smoothing(hp_map, beam_window=top_hat_beam(radius, NSIDE)) / solid_angle

#compute the total exposure and the integrated exposure per solid angle
def integrated_exposure(theta_min, theta_max, a_cell, time_begin, time_end, time, station_array):

    #to select only the stations within the time period time_begin < time < time_end
    time_indexes = np.where(np.logical_and(time > time_begin, time < time_end))[0]

    return (math.pi/2)*(math.cos(2*theta_min) - math.cos(2*theta_max))*a_cell*sum(station_array[time_indexes])

#compute the relative exposure per solid angle
def relative_exposure(dec, lat_auger, theta_min, theta_max):

    xi_max = (math.cos(theta_max) - math.sin(lat_auger)*math.sin(dec))/(math.cos(lat_auger)*math.cos(dec))

    if ( xi_max > 1):
        h_max = 0
    elif (xi_max < -1):
        h_max = math.pi
    else:
        h_max = math.acos(xi_max)


    xi_min = (math.cos(theta_min) - math.sin(lat_auger)*math.sin(dec))/(math.cos(lat_auger)*math.cos(dec))

    if ( xi_min > 1):
        h_min = 0
    elif (xi_min < -1):
        h_min = math.pi
    else:
        h_min = math.acos(xi_min)

    return math.cos(lat_auger)*math.cos(dec)*(math.sin(h_max) - math.sin(h_min)) + (h_max - h_min)*math.sin(lat_auger)*math.sin(dec)

#produce the exposure map
def MakeExposureMap(NSIDE, lat_auger, theta_min, theta_max, total_exposure):

    #get the angles of each pixel in skymap
    exp_theta, exp_phi = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)))

    exp_alpha, exp_dec = ToEquatorialCoordinates(exp_phi, exp_theta)

    exp_map_indexes = []
    exp_map_content = []

    for i in range(len(exp_phi)):

        exp_map_indexes.append(hp.ang2pix(NSIDE, exp_theta[i], exp_phi[i]))
        exp_map_content.append(relative_exposure(exp_dec[i], lat_auger, theta_min, theta_max))

    #computes the normalization of the relative exposure
    exposure_normalization = sum(exp_map_content)

    #produce vector with 0's
    exp_skymap = np.zeros(hp.nside2npix(NSIDE))

    np.add.at(exp_skymap, exp_map_indexes, np.multiply(exp_map_content, total_exposure/exposure_normalization))

    return exp_skymap

#Compute the LI and MA significance
def LiMaSignificance(LiMa_alpha, N_on, N_off):

    log_N_on = (1. + LiMa_alpha)*N_on / (LiMa_alpha*(N_on + N_off))
    log_N_off = (1. + LiMa_alpha)*N_off / (N_on + N_off)

    #to avoid negative values, induced by smoothing, in the arg of logs
    sig_2 = np.zeros(len(N_on))

    positive_indexes = np.where(np.logical_and(N_on > 0, LiMa_alpha > 0))
    sig_2[positive_indexes] += N_on[positive_indexes]*np.log(log_N_on[positive_indexes])

    positive_indexes = np.where(N_off > 0)
    sig_2[positive_indexes] += N_off[positive_indexes]*np.log(log_N_off[positive_indexes])
    return np.sqrt(2*np.abs(sig_2))*np.sign(N_on - LiMa_alpha*N_off)

#Makes the Li and Ma significance map and distribution
def MakeLiMaSignificanceDist(topHat_smoothing_radius, count_map, topHat_smooth_count_map, exposure_map, topHat_smooth_exposure_skymap, NSIDE):

    #define the number of pixels per region
    region_solid_angle = 2*math.pi*(1 - math.cos(topHat_smoothing_radius))
    solid_angle_per_pix = 4*math.pi / hp.nside2npix(NSIDE)
    npix_per_region = region_solid_angle / solid_angle_per_pix

    #define the LiMa_alpha, N_on and N_off
    LiMa_alpha = topHat_smooth_exposure_skymap / (np.sum(exposure_skymap) / npix_per_region - topHat_smooth_exposure_skymap)
    N_on = topHat_smooth_count_map*npix_per_region
    N_off = np.sum(count_map) - N_on

    #make the significance map
    significance_map = LiMaSignificance(LiMa_alpha, N_on, N_off)

    #creates mask to avoid showing pixels in region where exposure is null
    mask = np.zeros(hp.nside2npix(NSIDE))
    mask[hp.query_strip(NSIDE, np.pi/2-(theta_max + lat_auger), 0)] = 1

    #to create the distribution of Li and Ma significance
    LiMa_significance_dist = significance_map[np.where(mask == 1)]

    return significance_map, LiMa_significance_dist

#define the average distribution of tau for many realizations
def AverageDist(list_of_histograms):

    list_of_bin_edges = []
    list_of_bin_contents = []

    for histogram in list_of_histograms:

        bin_contents, bin_edges = histogram

        list_of_bin_edges.append(bin_edges)
        list_of_bin_contents.append(bin_contents)

    avg_bin_edges = list_of_bin_edges[0][1:]
    avg_bin_content = []

    nbins = len(avg_bin_edges)

    for i in range(nbins):

        bin_contents = []

        for hist_content in list_of_bin_contents:

            bin_contents.append(hist_content[i])

        avg_bin_content.append(np.mean(bin_contents))

    return avg_bin_edges, avg_bin_content

#compute the comulative distribution function
def ComulativeDist(pdf_bin_edges, pdf_data):

    cdf = np.cumsum(pdf_data)/sum(pdf_data)

    return pdf_bin_edges, cdf

#compute the complementary comulative distribution function
def ComplementaryComulativeDist(pdf_bin_edges, pdf_data):

    cdf = 1 - np.cumsum(pdf_data)/sum(pdf_data)

    return pdf_bin_edges, cdf

#fit estimator distribution
def GaussianFitDist(bin_content, bin_edges, estimator_list):

    bin_x = []
    #centers the bin content
    for i in range(len(bin_edges)-1):
        bin_x.append((bin_edges[i+1] + bin_edges[i])/2)

    sigma_data = np.sqrt(bin_content)

    init_parameters = [np.mean(estimator_list), math.sqrt(np.var(estimator_list)), len(estimator_list), 0]

    parameters, covariance = curve_fit(Gaussian, np.array(bin_x), np.array(bin_content), p0=init_parameters) #, sigma = sigma_data, absolute_sigma = True)

    x_gauss_fit = np.arange(min(estimator_list), max(estimator_list), 0.01)
    y_gauss_fit = Gaussian(x_gauss_fit, *parameters)

    parameters_error = np.sqrt(np.diag(covariance))

    return x_gauss_fit, y_gauss_fit, parameters, parameters_error, covariance


#computes the 95% confidence level areas around the CDF
def ConfidenceLevelArea(pdf_list):

    #bin edges of distribution
    pdf_content, pdf_bin_edges = pdf_list[0]

    #list to hold values of CDFs
    cdf_list = []

    #bin edges of distribution
    for pdf in pdf_list:
        pdf_content, pdf_bin_edges = pdf

        cdf_list.append(np.cumsum(pdf_content)/sum(pdf_content))



#-------------------------------
# main
#-------------------------------
#read the relavant info from the event selection files
path_to_selection = '../../Tau_Distribution/results/'
selection_info_file = 'AugerOpenData_VerticalEvents_SelectionInfo.parquet'
selection_info = pd.read_parquet(path_to_selection + selection_info_file, engine='fastparquet')

#latitude and longitude of the Pierre Auger Observatory
lat_auger = math.radians(-35.23)

#values of the maximum and minimum values of theta. In this case we are considering vertical events
theta_min = math.radians(float(selection_info.iloc[0]['Theta_min']))
theta_max = math.radians(float(selection_info.iloc[0]['Theta_max']))

#set value of unit cell of array according to distance between stations
lattice_parameter = 1.5 #in kilometers
a_cell = .5*math.sqrt(3)*lattice_parameter ** 2

#set time interval in seconds
time_begin = Time(selection_info.iloc[0]['t_begin'], format='fits').gps
time_end = Time(selection_info.iloc[0]['t_end'], format='fits').gps

#load station data
station_data = pd.read_parquet('../../DataSets/Auger_Hexagons/Hexagons_NoBadStations.parquet', engine='fastparquet')

time = station_data['gps_time'].to_numpy()
n5T5 = station_data['n5T5'].to_numpy()
n6T5 = station_data['n6T5'].to_numpy()

#compute time integrated exposure
total_exposure = integrated_exposure(theta_min, theta_max, a_cell, time_begin, time_end, time, (2/3)*n5T5 + n6T5)/(365*24*60) #per year per sr per km^2

print(total_exposure)

#defines path to files
path2file_ud = '../../DataSets/Vertical/UD_large_stats/'
path2file_rep = '../../DataSets/Vertical/MockData_Repeaters/Repeater_RandPosAndDate_large_stats/'

#defines the file counter
file_counter = 0

#defines the NSIDE parameter (aka number of pixels <-> angular resolution)
NSIDE = 128

#creates the smooth map using the top hat proceedure
topHat_smoothing_radius = math.radians(2)

#make exposure map and smooth exposure map
exposure_skymap = MakeExposureMap(NSIDE, lat_auger, theta_min, theta_max, total_exposure)
topHat_smooth_exposure_skymap = TopHatSmoothedMap(exposure_skymap, topHat_smoothing_radius, NSIDE)

#list to hold the Li and Ma histograms
LiMa_Significance_dist_list = []
Repeater_LiMa_Significance_dist_list = []

#loops over the files and computes the corresponding Li and Ma significance
for filename in os.listdir(path2file_ud):

    f = os.path.join(path2file_ud, filename)

    if os.path.isfile(f) and 'TimeOrdered_Accepted_Events_Uniform_Dist_N_100000_3824455_' in f and file_counter < 100:

        #skymap with events
        ud_skymap = FileToHealpyMap(f, ['ud_ra', 'ud_dec'], NSIDE)

        #creates smoothed top and Gaussian smoothed count maps
        topHat_smooth_skymap_ud = TopHatSmoothedMap(ud_skymap, topHat_smoothing_radius, NSIDE)

        #computes the Li and Ma distribution and skymap
        LiMa_significance_map, LiMa_significance_dist = MakeLiMaSignificanceDist(topHat_smoothing_radius, ud_skymap, topHat_smooth_skymap_ud, exposure_skymap, topHat_smooth_exposure_skymap, NSIDE)

        #creates histogram with Li and Ma Significance dist
        LiMa_Significance_dist_list.append(np.histogram(LiMa_significance_dist, bins=100, range=[-5, 5]))

        file_counter+=1

        print(file_counter,'files read!')

#resets the number of files to 0
file_counter = 0

#loops over the files and computes the corresponding Li and Ma significance
for filename in os.listdir(path2file_rep):

    f = os.path.join(path2file_rep, filename)

    if os.path.isfile(f) and 'IsoBG_ExpRepeater_RandPosAndDate_Period_3600_TotalEvents_100000_AcceptedRepEvents_200_RepIntensity_5_3843630' in f and file_counter < 100:

        #skymap with events
        rep_skymap = FileToHealpyMap(f, ['rep_ud_ra', 'rep_ud_dec'], NSIDE)

        #creates smoothed top and Gaussian smoothed count maps
        topHat_smooth_skymap_rep = TopHatSmoothedMap(rep_skymap, topHat_smoothing_radius, NSIDE)

        #computes the Li and Ma distribution and skymap
        Rep_LiMa_significance_map, Rep_LiMa_significance_dist = MakeLiMaSignificanceDist(topHat_smoothing_radius, rep_skymap, topHat_smooth_skymap_rep, exposure_skymap, topHat_smooth_exposure_skymap, NSIDE)

        #creates histogram with Li and Ma Significance dist
        Repeater_LiMa_Significance_dist_list.append(np.histogram(Rep_LiMa_significance_dist, bins=100, range=[-5, 5]))

        file_counter+=1

        print(file_counter,'files read!')

#computes the average Li and Ma significance distribution, cumulative and complementary to comulative
Average_LiMaSigDist_bins, Average_LiMaSigDist_content = AverageDist(LiMa_Significance_dist_list)
CDF_Average_LiMaSigDist_bins, CDF_Average_LiMaSigDist_content = ComulativeDist(Average_LiMaSigDist_bins, Average_LiMaSigDist_content)
Comp_CDF_Average_LiMaSigDist_bins, Comp_CDF_Average_LiMaSigDist_content = ComplementaryComulativeDist(Average_LiMaSigDist_bins, Average_LiMaSigDist_content)

#computes the Li and Ma significance for a repeater
#defines path to UD files

#filename_rep = 'IsoBG_ExpRepeater_RandPosAndDate_Period_3600_TotalEvents_100000_AcceptedRepEvents_200_RepIntensity_5_3843630_500.parquet'
#skymap with events
#rep_skymap = FileToHealpyMap(path2file_rep + filename_rep, ['rep_ud_ra', 'rep_ud_dec'], NSIDE)

#creates smoothed top and Gaussian smoothed count maps
#topHat_smooth_skymap_rep = TopHatSmoothedMap(rep_skymap, topHat_smoothing_radius, NSIDE)

#computes the Li and Ma distribution and skymap
#Repeater_LiMa_significance_map, Repeater_LiMa_significance_dist = MakeLiMaSignificanceDist(topHat_smoothing_radius, rep_skymap, topHat_smooth_skymap_rep, exposure_skymap, topHat_smooth_exposure_skymap, NSIDE)

#computes the average Li and Ma significance distribution, cumulative and complementary to comulative
Rep_Average_LiMaSigDist_bins, Rep_Average_LiMaSigDist_content = AverageDist(Repeater_LiMa_Significance_dist_list)
Rep_CDF_Average_LiMaSigDist_bins, Rep_CDF_Average_LiMaSigDist_content = ComulativeDist(Rep_Average_LiMaSigDist_bins, Rep_Average_LiMaSigDist_content)
Rep_Comp_CDF_Average_LiMaSigDist_bins, Rep_Comp_CDF_Average_LiMaSigDist_content = ComplementaryComulativeDist(Rep_Average_LiMaSigDist_bins, Rep_Average_LiMaSigDist_content)

plt.plot(CDF_Average_LiMaSigDist_bins, CDF_Average_LiMaSigDist_content, color = 'tab:blue')
plt.plot(Comp_CDF_Average_LiMaSigDist_bins, Comp_CDF_Average_LiMaSigDist_content, color='tab:blue')

plt.plot(Rep_CDF_Average_LiMaSigDist_bins, Rep_CDF_Average_LiMaSigDist_content, color = 'tab:orange')
plt.plot(Rep_Comp_CDF_Average_LiMaSigDist_bins, Rep_Comp_CDF_Average_LiMaSigDist_content, color='tab:orange')

#plt.plot(Comp_CDF_Average_LiMaSigDist_bins, Comp_CDF_Average_LiMaSigDist_content)

plt.yscale('log')

plt.show()
