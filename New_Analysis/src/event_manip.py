import numpy as np

#order events by time
def time_ordered_events(time, ra, dec, theta, lst):

    #indices of ordered time array
    sorted_indices = time.argsort()

    time = time[sorted_indices]
    ra = ra[sorted_indices]
    dec = dec[sorted_indices]
    theta = theta[sorted_indices]
    lst = lst[sorted_indices]

    return time, ra, dec, theta, lst

#compute angular difference between 2 events
def ang_diff(dec_1, dec_2, ra_1, ra_2):
    return np.arccos(np.cos(dec_1)*np.cos(dec_2)*np.cos(ra_1 - ra_2) + np.sin(dec_1)*np.sin(dec_2))
