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

#Applies cuts to the auger data set and returns a list with the selected events
def SelectAugerData(filename, energy_th, theta_min, theta_max, t_min, t_max):

    #saves raw auger data from input file
    auger_data_raw = pd.read_parquet(filename, engine='fastparquet')

    #applies cuts in energy, and theta and time
    auger_data = auger_data_raw.loc[(auger_data_raw['sd_theta'] > theta_min) & (auger_data_raw['sd_theta'] < theta_max) & (auger_data_raw['sd_energy'] > energy_th) & (auger_data_raw['gpstime'] > t_min) & (auger_data_raw['gpstime'] < t_max) ]

    #saves the number of events after cuts
    N_events = len(auger_data.index)

    #saves the gpstime
    evt_gpstime = auger_data['gpstime'].to_numpy()

    #list to hold events
    event_list = []

    for i in range(N_events):
        time = auger_data.iloc[i]['gpstime']
        ra = math.radians(auger_data.iloc[i]['sd_ra'])
        dec = math.radians(auger_data.iloc[i]['sd_dec'])
        energy = auger_data.iloc[i]['sd_energy']

        event_list.append(Event(ra, dec, time, energy))

    return min(evt_gpstime), max(evt_gpstime), event_list

#checks if event is in the angular window centered at another event
def EventInSphericalCap(alpha_center, delta_center, alpha, delta, ang_window):

    ang_sep = math.acos(math.cos(delta_center)*math.cos(delta)*math.cos(alpha_center - alpha) + math.sin(delta_center)*math.sin(delta))

    if( ang_sep > ang_window ):
        return False
    else:
        return True

#name of file with
# if len(sys.argv) == 1:
#     raise Exception('Please insert the name of the event file!')
#
# filename = './output/' + sys.argv[1]
#
# # date = '2015-01-01T00:00:00'
# # period = '3600.0'
# # N_total_events = '100'
# # N_accepted_rep_events = '68'
# #
# #verifies the existence of file
# filename = './output/TimeOrdered_Events_ExponentialRepeater_Date_' + date + '_Period_' + period + '_TotalEvents_' + N_total_events + '_AcceptedRepEvents_' + N_accepted_rep_events + '.parquet'

# try:
#     file = open(filename)
# except FileNotFoundError:
#     print('File not found')
# finally:
#     file.close()

#defines the cuts to apply to auger data
energy_th = math.sqrt(10)
theta_min = 0
theta_max = 60

#saves auger events after appyling the cuts
path_to_data = '../DataSets/Auger_Pub_Data/'
data_file = 'AugerOpenData_VerticalEvents.parquet'
t_min, t_max, accepted_events_list = SelectAugerData(path_to_data + data_file, energy_th, theta_min, theta_max, Time('2004-01-01T00:00:00', format='fits').gps, Time('2021-12-31T23:59:59', format='fits').gps)

#print the number of events and the dates between which events are being considered
print('Total number of events:', len(accepted_events_list), 'between', Time(t_min, format='gps').fits, 'and', Time(t_max, format='gps').fits)

#print(data_rep)

#convert relevant pandas columns into np arrays
# auger_gpstime = data_auger['gpstime'].to_numpy()
#
# rep_event_timestamp = data_rep["rep_ud_gpstime"].to_numpy()
# rep_event_energy = data_rep["rep_ud_energy"].to_numpy()
# rep_event_alpha = data_rep["rep_ud_ra"].to_numpy()
# rep_event_delta = data_rep["rep_ud_dec"].to_numpy()
#
# #define the minimum date, energy and zenith angle
# time_min = Time(str(min(auger_gpstime)), scale='utc', format='gps').gps
# time_max = Time(str(max(auger_gpstime)), scale='utc', format='gps').gps
#
# #create list of accepted events
# accepted_rep_events_list = []
#
# #list to hold events with repeater and uniform background
# for i in range(len(rep_event_timestamp)):
#     #time = Time(str(rep_event_timestamp[i]),format='gps')
#     accepted_rep_events_list.append(Event(rep_event_alpha[i], rep_event_delta[i],rep_event_timestamp[i], rep_event_energy[i]))
#
# #saves number of accepted events
# N_accepted_rep_events = len(accepted_rep_events_list)
#
# print('Number of events with repeater and uniform background:', N_accepted_rep_events)

#angular window
ang_window = np.radians(1)

#set begin time
task1_begin = datetime.now()

#vector with time interval between consecutive events
tau_rep = []

#vector to save events with respective tau
accepted_events_with_tau = []
event_counter = 0

#compute the time difference between two consecutive events for repeater with uniform background
for evt1 in accepted_events_list:

    for evt2 in accepted_events_list:

        if evt2 > evt1:

            if ( abs(evt1.delta - evt2.delta) > ang_window or abs(evt1.alpha - evt2.alpha) > ang_window ):
                continue

            if EventInSphericalCap(evt1.alpha, evt1.delta, evt2.alpha, evt2.delta, ang_window):
                tau = evt2.time - evt1.time
                tau_rep.append(tau)
                accepted_events_with_tau.append([evt1.alpha, evt1.delta, evt1.time, evt1.energy, evt2.alpha, evt2.delta, evt2.time, evt2.energy, tau])
                break

    event_counter+=1

    print(event_counter,'events done!')


print('The computation for tau for', len(accepted_events_list), ' events took', datetime.now() - task1_begin)

#print file with taus
#output_ud_data = pd.DataFrame(tau_ud,columns=['tau (s)'])
#output_rep_data = pd.DataFrame(tau_rep,columns=['tau (s)'])

#print file with events and respective taus
output_data = pd.DataFrame(accepted_events_with_tau, columns=['evt1_ra (rad)','evt1_dec (rad)','evt1_gpstime','evt1_energy (EeV)', 'evt2_ra (rad)','evt2_dec (rad)','evt2_gpstime','evt2_energy (EeV)','tau (s)'])

print(output_data)

#save data in file
output_data.to_parquet('./results/AugerOpenData_VerticalEvents_with_tau.parquet', index=False)

#export output file with relevant parameters
selection_info = np.array([len(accepted_events_list), energy_th, theta_min, theta_max, ang_window, Time(t_min,format='gps').fits, Time(t_max,format='gps').fits])

output_info = pd.DataFrame([selection_info], columns=["N_events","E_th","Theta_min", "Theta_max","Ang_window","t_begin","t_end"])
output_info.to_parquet('./results/AugerOpenData_VerticalEvents_SelectionInfo.parquet', index=False)

print(output_info.info())
