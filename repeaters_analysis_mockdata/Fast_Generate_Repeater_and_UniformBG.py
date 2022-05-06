#---------------------------------------
# Try optimize the code!!!!
#---------------------------------------
import sys
import math
import numpy as np
import healpy as hp

#for plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc

#to generate random points
from random import randint
from random import seed
from random import random
from random import sample

from datetime import datetime

#to convert to pandas format
import pandas as pd

#to handle coordinate transformations and time manipulations
from astropy.time import Time
from astropy import units as u

#import class event
from Class_Event import Event

rc('font',**{'family':'serif','serif':['DejaVu Sans']})
rc('text', usetex=True)

#round gps time in seconds, to minutes
def round_to_minute(time):
    r = time % 60
    return time - r

#to compute right ascension from a uniformly distributed random var x
def alpha(x):
    return math.pi*2*x

#to compute declination from a uniformly distributed random var x
def delta(x):
    return math.asin(2*x -1)

#convert to healpy coordinates
def ToHealpyCoordinates(alpha,delta):
    phi, theta = [], []

    for i in range(len(alpha)):
        phi.append(alpha[i])
        theta.append(math.pi/2 - delta[i])

    return phi, theta

#convert from healpy to equatorial coordinates
def ToEquatorialCoordinates(phi,theta):
    alpha, delta = [], []

    for i in range(len(phi)):
        alpha.append(phi[i])
        delta.append(math.pi/2 - theta[i])

    return alpha, delta

#compute exposure per solid angle per unit time
def InstantExposure(time, gps_time, nT5_list, nT5_max, l, delta, alpha, LST):

	rounded_time = round_to_minute(time)

	nT5 = nT5_list[np.where(gps_time == rounded_time)]

	return (nT5/nT5_max)*(math.cos(l)*math.cos(delta)*math.cos(LST - alpha) + math.sin(l)*math.sin(delta))

#accept events
def AcceptEvent(time, gps_time, nT5_list, nT5_max, lat, theta_min, theta_max, delta, alpha, LST):

    ang = math.acos(math.cos(delta)*math.cos(lat)*math.cos(alpha - LST) + math.sin(delta)*math.sin(lat))

    if ( ang > theta_max or ang < theta_min ):
        return False
    else:
        instant_exp = InstantExposure(time, gps_time, nT5_list, nT5_max, lat, delta, alpha, LST)
        x = random()

        if (x < instant_exp):
            return True
        else:
            return False

#generate, randomly, a random timestamp between two dates
def RandomEventTimeStamp(t_min, t_max, lat_pao, long_pao):

    long_pao = long_pao*u.rad
    lat_pao = lat_pao*u.rad

    t_gps = randint(t_min,t_max)

    t = Time(str(t_gps), format='gps', scale='utc',location=(long_pao,lat_pao))

    return t

#generate points (declination,RA) uniformely over a sphere
def RandomEventEquatorialCoordinates():

    u = random()
    v = random()

    return alpha(u), delta(v)

#define key to sort events by
def sort_by_event_time(evt):
    return evt[2]

#to order events according to their time stamp
def TimeOrderedEvents(event_list):

    ordered_event_list = []

    for evt in event_list:
        ordered_event_list.append([evt.alpha, evt.delta, evt.time.gps, evt.energy])

    ordered_event_list = sorted(ordered_event_list, key=sort_by_event_time)

    return ordered_event_list

#return the number of auger events in open data with E > 10^18.5 eV and theta < 60
def AugerEventsData(file_name, energy_th, theta_min, theta_max):

    data_auger = pd.read_csv(file_name)

    event_timestamp = data_auger["gpstime"].to_numpy()
    event_sd_energy = data_auger["sd_energy"].to_numpy()
    event_sd_theta = data_auger["sd_theta"].to_numpy()

    #saves timestamp of accepted events
    accepted_events_timestamp = []

    #list to hold events after events after cuts
    for i in range(len(event_timestamp)):
        if( event_sd_energy[i] > energy_th and event_sd_theta[i] < theta_max and  theta_min < event_sd_theta[i] ):
                accepted_events_timestamp.append(event_timestamp[i])

    t_min = min(accepted_events_timestamp)
    t_max = max(accepted_events_timestamp)

    return t_min, t_max, len(accepted_events_timestamp)

#generate repeater with period and lifespan in seconds
def Generate_Discrete_Repeater(delta, alpha, time_stamp, period, lifespan, lat_pao, long_pao):

    #list to hold events
    event_list = []

    #define coordinates of PAO in radians
    long_pao = long_pao*u.rad
    lat_pao = lat_pao*u.rad

    t = Time(time_stamp, format='fits',location=(long_pao,lat_pao))

    end_date = t.gps + lifespan

    while t.gps < end_date:

        event_list.append(Event(alpha,delta,t,math.nan))

        t.gps+=period

        t = Time(str(t.gps), format='gps', location=(long_pao,lat_pao))

    return event_list

#generate repeater with period and lifespan in seconds
def Generate_Exponential_Repeater(delta, alpha, time_stamp, avg_period, lat_pao, long_pao):

    #define coordinates of PAO in radians
    long_pao = long_pao*u.rad
    lat_pao = lat_pao*u.rad

    t = Time(time_stamp, format='fits',location=(long_pao,lat_pao))

    #generate random period from period for exponentially decreasing luminosity of source
    period = -avg_period*math.log(1 - random())
    time = Time(t.gps + period, format='gps', location=(long_pao,lat_pao))

    #smearing of angular positions
    alpha_gauss = np.random.normal(alpha, math.radians(1), 1)[0]
    delta_gauss = np.random.normal(delta, math.radians(1), 1)[0]

    return Event(alpha_gauss,delta_gauss,time,math.nan)

#verify in event is in spherical cap around another event
def EventInSphericalCap(alpha_center, delta_center, alpha, delta, ang_window):

    ang_sep = math.acos(math.cos(delta_center)*math.cos(delta)*math.cos(alpha_center - alpha) + math.sin(delta_center)*math.sin(delta))

    if( ang_sep > ang_window ):
        return False
    else:
        return True

#transform data from events file into a vector of events
def FromEventFile_to_EventList(filename):

    df = pd.read_parquet(filename, engine='fastparquet')

    alpha = df["ud_ra"].to_numpy()
    delta = df["ud_dec"].to_numpy()
    gpstime = df["ud_gpstime"].to_numpy()
    energy = df["ud_energy"].to_numpy()

    event_list = []

    for i in range(len(alpha)):
        event_list.append(Event(alpha[i], delta[i], Time(gpstime[i], format='gps'), energy[i]))

    return event_list

#-------------------------------
# main
#-------------------------------
#save the oldest and most recent timestamp of all public events with E > 10^18.5 eV and theta < 60
energy_th = math.sqrt(10)
theta_min = 0
theta_max = 60

t_min, t_max, N_auger_events = AugerEventsData("./input/dataSummary.csv",energy_th,theta_min, theta_max)

#print begin date, end date and number of Auger events
print('First event date:', Time(str(t_min), format='gps').fits, ' Last event date:', Time(str(t_max), format='gps').fits)

#save the gpstime, number of 6T5 and 5T5 hexagons since 2004
hexagon_data = pd.read_parquet("./input/Hexagons_NoBadStations.parquet", engine = 'fastparquet')

n6T5 = hexagon_data['n6T5'].to_numpy()
n5T5 = hexagon_data['n5T5'].to_numpy()

gps_time = hexagon_data['gps_time'].to_numpy()

nT5 = np.add(n6T5, n5T5)

nT5_max = max(nT5)

#define number of accepted events
if len(sys.argv) == 1:
    raise Exception('Choose the total number of events simulated:')

N_events = int(sys.argv[1])

#latitude and longitude of PAO
lat_pao = np.radians(-35.28)
long_pao = np.radians(-69.2)

#define seed
seed = seed(open('/dev/random','rb').read(4))

#Convert accepted zenith angles to radians
theta_max = np.radians(theta_max)
theta_min = np.radians(theta_min)

#compute the maximum and minimum values of accepted declinations
delta_max = lat_pao + theta_max
delta_min = lat_pao - theta_max

#define the characteristics of the repeater
rep_nevents = 100
rep_number_of_explosions = 20
rep_events_per_explosion = int(rep_nevents / rep_number_of_explosions)
rep_period = 3600 #one sidereal year in seconds

#begin
task1_begin = datetime.now()

#create list of accepted events uniformely distributed in time and over a sphere
accepted_ud_events = []

task1_end = datetime.now() - task1_begin

#count the number of generated events
N_gen_events = 0

#accept repeater events
while len(accepted_ud_events) < rep_nevents:

    rep_alpha, rep_delta = RandomEventEquatorialCoordinates()

    if( rep_delta > theta_max + lat_pao):
        continue

    rep_date = RandomEventTimeStamp(Time('2010-01-01T00:00:00', format='fits').gps, Time('2018-01-01T00:00:00', format='fits').gps, lat_pao, long_pao)

    for i in range(rep_events_per_explosion):

        #generate repeater event
        repeater_event = Generate_Exponential_Repeater(rep_delta, rep_alpha, rep_date, rep_period, lat_pao, long_pao)

        #sidereal time of the observatory at the time of the event
        sidereal_time = repeater_event.time.sidereal_time('apparent').rad

        #accept events according to exposure
        if AcceptEvent(repeater_event.time.gps, gps_time, nT5, nT5_max, lat_pao, theta_min, theta_max, repeater_event.delta, repeater_event.alpha, sidereal_time):
            accepted_ud_events.append(repeater_event)

    print(len(accepted_ud_events),'repeater events accepted!')

#events from repeater
N_accepted_rep_events = len(accepted_ud_events)

print('Accepted repeater events', N_accepted_rep_events)

#choose remaining events from isotropic background
ud_filename = '/home/miguelm/Documents/Anisotropies/Repeater_Analysis/DataSets/UD_large_stats/TimeOrdered_Accepted_Events_Uniform_Dist_N_100000_3824455_1.parquet'

ud_event_list = FromEventFile_to_EventList(ud_filename)

new_event_list = sample(ud_event_list, len(ud_event_list) - N_accepted_rep_events)

for evt in new_event_list:
    accepted_ud_events.append(evt)

#print time it takes to accept events
print('Accepting generated events took', datetime.now() - task1_begin,'to run')

task2_begin = datetime.now()

#time order accepted events and converts each event to array
time_ordered_accepted_events = TimeOrderedEvents(accepted_ud_events)

#print time it takes to time order events
print('Time ordering events took', datetime.now() - task2_begin,'to run')

#save data to parquet file
data = pd.DataFrame(time_ordered_accepted_events,columns=['rep_ud_ra','rep_ud_dec','rep_ud_gpstime','rep_ud_energy'])

output_path = '/home/miguelm/Documents/Anisotropies/Repeater_Analysis/DataSets/MockData_Repeaters/'
output_rep_events = 'TimeOrdered_Events_ExponentialRepeater_RandomEquatorialCoorinates_Period_' + str(rep_period)+ '_TotalEvents_' + str(N_events) + '_AcceptedRepEvents_' + str(N_accepted_rep_events) + '_MaxRepIntensity_' + str(rep_events_per_explosion) + '.parquet'
data.to_parquet(output_path + output_rep_events, index=False)

print(data)

#------------------------------------------------------
# Compute tau for this set of events
#------------------------------------------------------
print('%%%%%%%% TAU COMPUTATION %%%%%%%%')

#angular window
ang_window = np.radians(1)

#set begin time
task3_begin = datetime.now()

#vector with time interval between consecutive events
tau_rep = []

#vector to save events with respective tau
accepted_rep_events_with_tau = []
event_counter = 0

#compute the time difference between two consecutive events for repeater with uniform background
for evt1 in time_ordered_accepted_events:

    for evt2 in time_ordered_accepted_events:

        if evt2[2] > evt1[2]:

            if ( abs(evt1[1] - evt2[1]) > ang_window or abs(evt1[0] - evt2[0]) > ang_window ):
                continue

            if EventInSphericalCap(evt1[0], evt1[1], evt2[0], evt2[1], ang_window):
                tau = evt2[2] - evt1[2]
                tau_rep.append(tau)
                accepted_rep_events_with_tau.append([evt1[0], evt1[1], evt1[2], evt1[3], evt2[0], evt2[1], evt2[2], evt2[3], tau])
                break

    event_counter+=1

    print(event_counter,'events done!')

print('The computation for tau for', len(accepted_ud_events) , ' events with repeater and uniform background took', datetime.now() - task3_begin)

#print file with events and respective taus
output_rep_data = pd.DataFrame(accepted_rep_events_with_tau,columns=['evt1_ra (rad)','evt1_dec (rad)','evt1_gpstime','evt1_energy (EeV)', 'evt2_ra (rad)','evt2_dec (rad)','evt2_gpstime','evt2_energy (EeV)','tau (s)'])

print(output_rep_data)

#save parquet file with data
output_rep_tau = 'Rep_events_with_tau_Period_' + str(rep_period)+ '_TotalEvents_' + str(N_events) + '_AcceptedRepEvents_' + str(N_accepted_rep_events) + '_MaxRepIntensity_' + str(rep_events_per_explosion) + '.parquet'
output_rep_data.to_parquet(output_path  + output_rep_tau, index=False)

#export output file with relevant parameters
selection_info_ud = np.array([len(accepted_ud_events), math.nan, theta_max, ang_window, Time(str(t_min),format='gps').fits, Time(str(t_max),format='gps').fits])
output_selection_name = 'Selection_event_info_Period_' + str(rep_period)+ '_TotalEvents_' + str(N_events) + '_AcceptedRepEvents_' + str(N_accepted_rep_events) + '_MaxRepIntensity_' + str(rep_events_per_explosion) + '.parquet'

output_info = pd.DataFrame([selection_info_ud], columns=["N_events","E_th","Theta_max","Ang_window","t_begin","t_end"])
output_info.to_parquet(output_path + output_selection_name, index=False)

print(output_info.info())
