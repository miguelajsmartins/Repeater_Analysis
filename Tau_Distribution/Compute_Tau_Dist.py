import os
import sys
import math
import numpy as np

#for plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc

#for data manipulation
import pandas as pd

#for ploting on sphere
import healpy as hp

#for time and coordinates manipulation
from astropy.time import Time

#to calculate script run time
from datetime import datetime

#import class "Event"
from Class_Event import Event

rc('font',**{'family':'serif','serif':['DejaVu Sans']})
rc('text', usetex=True)

#def AcceptAugerEvent(array):

def EventInSphericalCap(alpha_center, delta_center, alpha, delta, ang_window):

    ang_sep = math.acos(math.cos(delta_center)*math.cos(delta)*math.cos(alpha_center - alpha) + math.sin(delta_center)*math.sin(delta))

    if( ang_sep > ang_window ):
        return False
    else:
        return True

#name of file with repeater
if len(sys.argv) == 1:
    raise Exception('Please insert the name of the event file!')

filename = './output/' + sys.argv[1]

# date = '2015-01-01T00:00:00'
# period = '3600.0'
# N_total_events = '100'
# N_accepted_rep_events = '68'
#
# #verifies the existence of file
# filename = './output/TimeOrdered_Events_ExponentialRepeater_Date_' + date + '_Period_' + period + '_TotalEvents_' + N_total_events + '_AcceptedRepEvents_' + N_accepted_rep_events + '.parquet'

try:
    file = open(filename)
except FileNotFoundError:
    print('File not found')
finally:
    file.close()

#read file with shower data
data_auger = pd.read_csv('./input/dataSummary.csv')
data_rep = pd.read_parquet(filename, engine='fastparquet')

print(data_rep)

#convert relevant pandas columns into np arrays
auger_gpstime = data_auger['gpstime'].to_numpy()

rep_event_timestamp = data_rep["rep_ud_gpstime"].to_numpy()
rep_event_energy = data_rep["rep_ud_energy"].to_numpy()
rep_event_alpha = data_rep["rep_ud_ra"].to_numpy()
rep_event_delta = data_rep["rep_ud_dec"].to_numpy()

#define the minimum date, energy and zenith angle
time_min = Time(str(min(auger_gpstime)), scale='utc', format='gps').gps
time_max = Time(str(max(auger_gpstime)), scale='utc', format='gps').gps

#create list of accepted events
accepted_rep_events_list = []

#list to hold events with repeater and uniform background
for i in range(len(rep_event_timestamp)):
    #time = Time(str(rep_event_timestamp[i]),format='gps')
    accepted_rep_events_list.append(Event(rep_event_alpha[i], rep_event_delta[i],rep_event_timestamp[i], rep_event_energy[i]))

#saves number of accepted events
N_accepted_rep_events = len(accepted_rep_events_list)

print('Number of events with repeater and uniform background:', N_accepted_rep_events)

#angular window
ang_window = np.radians(1)

#set begin time
task1_begin = datetime.now()

#vector with time interval between consecutive events
tau_rep = []

#vector to save events with respective tau
accepted_rep_events_with_tau = []
event_counter = 0

#compute the time difference between two consecutive events for repeater with uniform background
for evt1 in accepted_rep_events_list:

    for evt2 in accepted_rep_events_list:

        if evt2 > evt1:

            if ( abs(evt1.delta - evt2.delta) > ang_window or abs(evt1.alpha - evt2.alpha) > ang_window ):
                continue

            if EventInSphericalCap(evt1.alpha, evt1.delta, evt2.alpha, evt2.delta, ang_window):
                tau = evt2.time - evt1.time
                tau_rep.append(tau)
                accepted_rep_events_with_tau.append([evt1.alpha, evt1.delta, evt1.time, evt1.energy, evt2.alpha, evt2.delta, evt2.time, evt2.energy, tau])
                break

    event_counter+=1

    print(event_counter,'events done!')

print('The computation for tau for', N_accepted_rep_events, ' events with repeater and uniform background took', datetime.now() - task1_begin)

#print file with taus
#output_ud_data = pd.DataFrame(tau_ud,columns=['tau (s)'])
#output_rep_data = pd.DataFrame(tau_rep,columns=['tau (s)'])

#print file with events and respective taus
output_rep_data = pd.DataFrame(accepted_rep_events_with_tau,columns=['evt1_ra (rad)','evt1_dec (rad)','evt1_gpstime','evt1_energy (EeV)', 'evt2_ra (rad)','evt2_dec (rad)','evt2_gpstime','evt2_energy (EeV)','tau (s)'])

print(output_rep_data)

#save csv file with data
output_rep_data.to_parquet('./output/Rep_events_with_tau_Date_' + date + '_Period_' + period + '_N_' + N_events + '.parquet', index=False)

#define energy and theta threshold
theta_max = 60

print(type(ang_window))

#export output file with relevant parameters
selection_info_ud = np.array([N_accepted_rep_events, math.nan, theta_max, ang_window, Time(str(time_min),format='gps').fits, Time(str(time_max),format='gps').fits])

print([selection_info_ud])

output_info = pd.DataFrame([selection_info_ud], columns=["N_events","E_th","Theta_max","Ang_window","t_begin","t_end"])
output_info.to_parquet('./output/Selection_event_info_Date_' + date + '_Period_' + period + '_N_' + N_events + '.parquet', index=False)

print(output_info.info())
