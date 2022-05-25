import numpy as np
import math
import pandas as pd

from astropy.time import Time

def integrated_exposure(theta_min, theta_max, a_cell, time_begin, time_end, time, station_array):

    #to select only the stations within the time period time_begin < time < time_end
    time_indexes = np.where(np.logical_and(time > time_begin, time < time_end))[0]

    return (math.pi/2)*(math.cos(2*theta_min) - math.cos(2*theta_max))*a_cell*sum(station_array[time_indexes])

#values of the maximum and minimum values of theta
theta_min = math.radians(0)
theta_max = math.radians(80)

#set value of unit cell of array according to distance between stations
d = 1.5 #in kilometers
a_cell = .5*math.sqrt(3)*d*d

#set time interval in seconds
time_begin = Time('2004-01-03T23:16:17', format='fits').gps
time_end = Time('2019-12-29T16:40:49', format='fits').gps

#load station data
station_data = pd.read_parquet('DataSets/Auger_Hexagons/Hexagons_NoBadStations.parquet', engine='fastparquet')

time = station_data['gps_time'].to_numpy()
n5T5 = station_data['n5T5'].to_numpy()
n6T5 = station_data['n6T5'].to_numpy()

#compute time integrated exposure
total_exposure = integrated_exposure(theta_min, theta_max, a_cell, time_begin, time_end, time, n6T5 + (2/3)*n5T5 )/(365*24*60) #per year per sr per km^2

print(total_exposure)
