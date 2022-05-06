import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from random import random
from random import seed
from datetime import datetime

from astropy.time import Time

def generate_repeater(date, meantime, n_events):

    times = []

    for i in range(n_events):
        v = random()
        time = -meantime*math.log(1 - v)
        times.append(time)

    return times

# data_rep = pd.read_parquet('TimeOrdered_Events_ExponentialRepeater_Date_2015-01-01T00:00:00_Period_3600.0_Nevents_100.parquet', engine='fastparquet')
#
# print(data_rep.head())
#
# evt_ra = data_rep['rep_ud_ra'].to_numpy()
# evt_dec = data_rep['rep_ud_dec'].to_numpy()
# evt_time = data_rep['rep_ud_gpstime'].to_numpy()
#
# evt_ra = np.degrees(evt_ra)
# evt_dec = np.degrees(evt_dec)
#
# rep_ra = evt_ra[np.where(abs(evt_ra - 100) < 0.001)]
# rep_dec = evt_dec[np.where(abs(evt_ra - 100) < 0.001)]
# rep_gpstime = evt_time[np.where(abs(evt_ra - 100) < 0.001)]
#
# print(rep_ra)
# print(rep_dec)
# print(Time(rep_gpstime, format='gps').fits)
#
# number_of_rep_events = len(rep_ra)
#
# print(number_of_rep_events)

#generate test repeater
n_events = 10000
date = Time('2015-01-01T00:00:00', format='fits')
meantime = 3600 #in seconds

seed(datetime.now())

list_rep_evts = generate_repeater(date, meantime, n_events)

#plt.hist(rep_gpstime - date.gps, bins=10, range=[min(rep_gpstime-date.gps),max(rep_gpstime - date.gps)])
plt.hist(np.log10(list_rep_evts), bins=100, range=[min(np.log10(list_rep_evts)),max(np.log10(list_rep_evts))])
plt.yscale('log')

plt.show()
