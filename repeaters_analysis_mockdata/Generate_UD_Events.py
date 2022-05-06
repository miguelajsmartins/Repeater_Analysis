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
def InstantExposure(time, gps_time, n6T5_list, n6T5_max, l, delta, alpha, LST):

	rounded_time = round_to_minute(time)

	n6T5 = n6T5_list[np.where(gps_time == rounded_time)]

	return (n6T5/n6T5_max)*math.cos(l)*math.cos(delta)*math.cos(LST - alpha) + math.sin(l)*math.sin(delta)

#accept events
def AcceptEvent(time, gps_time, n6T5_list, n6T5_max, lat, theta_min, theta_max, delta, alpha, LST):

    pos_source = hp.ang2vec(math.pi/2 - delta, alpha)
    pos_obs = hp.ang2vec(math.pi/2 - lat, LST)

    ang = math.acos(np.dot(pos_source,pos_obs))

    if ( ang > theta_max or ang < theta_min ):
        return False
    else:
        instant_exp = InstantExposure(time, gps_time, n6T5_list, n6T5_max, lat, delta, alpha, LST)
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

#-------------------------------
# main
#-------------------------------
#save the oldest and most recent timestamp of all public events with E > 10^18.5 eV and theta < 60
energy_th = math.sqrt(10)
theta_min = 0
theta_max = 60

t_min, t_max, N_auger_events = AugerEventsData("dataSummary.csv",energy_th,theta_min, theta_max)

#print begin date, end date and number of Auger events
print('First event date:', Time(str(t_min), format='gps').fits, ' Last event date:', Time(str(t_max), format='gps').fits)

#save the gpstime, number of 6T5 and 5T5 hexagons since 2004
hexagon_data = pd.read_parquet("./input/Hexagons_NoBadStations.parquet", engine = 'fastparquet')

n6T5 = hexagon_data['n6T5'].to_numpy()

gps_time = hexagon_data['gps_time'].to_numpy()

n6T5_max = max(n6T5)

#define number of accepted events
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

#begin
task1_begin = datetime.now()

#create list of accepted events uniformely distributed in time and over a sphere
accepted_ud_events = []

task1_end = datetime.now() - task1_begin

#count the number of generated events
N_gen_events = 0

#accept remaining events to form an isotropic background
while len(accepted_ud_events) < N_events:

    N_gen_events+=1

    #generate random event
    evt_time = RandomEventTimeStamp(t_min,t_max, lat_pao, long_pao)
    evt_alpha, evt_delta = RandomEventEquatorialCoordinates()

    evt = Event(evt_alpha, evt_delta, evt_time, math.nan)

    #if event is always outside the field of view of the observatory, it is skiped
    if( evt.delta > delta_max or evt.delta < delta_min):
        continue

    #sidereal time of the observatory at the time of the event
    sidereal_time = evt.time.sidereal_time('apparent').rad

    #accept events according to exposure
    if AcceptEvent(evt.time.gps, gps_time, n6T5, n6T5_max, lat_pao, theta_min, theta_max, evt.delta, evt.alpha, sidereal_time):
        accepted_ud_events.append(evt)
        print(len(accepted_ud_events),'events accepted')

#fraction of events kept
eff = 100*N_events/N_gen_events

print('Number of events =', N_events,' Number of accepted events =', len(accepted_ud_events),' Percentage of events kept',eff,'%')

#print time it takes to accept events
print('Accepting generated events took', datetime.now() - task1_begin,'to run')

task2_begin = datetime.now()

#time order accepted events and converts each event to array
time_ordered_accepted_events = TimeOrderedEvents(accepted_ud_events)

#print time it takes to time order events
print('Time ordering events took', datetime.now() - task2_begin,'to run')

print(type(time_ordered_accepted_events[0][2]))

#save data to csv file
data = pd.DataFrame(time_ordered_accepted_events,columns=['ud_ra','ud_dec','ud_gpstime','ud_energy'])

data.to_parquet('./output/TimeOrdered_UniformEvts.parquet', index=False)

print(data)
