import pandas as pd
import numpy as np
import math

from astropy.time import Time

#Applies cuts to the auger data set and returns reduced data frame
def SelectAugerData(auger_data_raw, columns, energy_th, theta_min, theta_max, t_min, t_max):

    gpstime = columns[0]
    theta = columns[3]
    energy = columns[7]

    #applies cuts in energy, and theta and time
    auger_data = auger_data_raw.loc[(auger_data_raw[theta] > theta_min) & (auger_data_raw[theta] < theta_max) & (auger_data_raw[energy] > energy_th) & (auger_data_raw[gpstime] > t_min) & (auger_data_raw[gpstime] < t_max) ]

    return auger_data[columns]

#defines the cuts and selects the relevant columns
time_min = Time('2004-01-01T00:00:00', format='fits').gps
time_max = Time('2019-01-01T00:00:00', format='fits').gps

theta_min = 0
theta_max = 60

energy_th = math.sqrt(10)

columns = ['gpstime', 'sd_ra', 'sd_dec', 'sd_theta', 'sd_dtheta', 'sd_phi', 'sd_dphi', 'sd_energy']

#saves raw data
raw_data = pd.read_csv('dataSummary.csv')

#applies cuts to data
auger_vert_data = SelectAugerData(raw_data, columns, energy_th, theta_min, theta_max, time_min, time_max)
print(auger_vert_data)

#saves data in parquet file
output_file = auger_vert_data.to_parquet('AugerOpenData_VerticalEvents.parquet', index = False)
