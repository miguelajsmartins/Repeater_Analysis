import numpy as np
import math

from random import random
from random import seed

from datetime import datetime

from operator import itemgetter

import matplotlib.pyplot as plt

#to generate the events with a given position on a rectange with area lenght_x * lenght_y and with time in [t_min, t_max]
def GenerateEvents(N_events, length_x, length_y, t_min, t_max):

    event_list = []

    for i in range(N_events):

        x = length_x*random()
        y = length_y*random()
        t = (t_max - t_min)*random() + t_min

        event_list.append([x,y,t])

    return event_list

#to order the list of events
def OrderEvents(event_list):

    get_time = itemgetter(2)

    event_list.sort(key=get_time)

    return event_list

#compute the time difference between consecutive events
def ComputeTau(event_list, lenght_box_x, lenght_box_y):

    tau_list = []

    for evt1 in event_list:
        for evt2 in event_list:

            if ( evt1[2] < evt2[2] and abs(evt1[0] - evt2[0]) < length_box_x and abs(evt1[1] - evt2[1]) < length_box_y ):
                tau_list.append(evt2[2] - evt1[2])
                break


    return tau_list

#compute number of events with tau < tau_max
def ComulativeBelow(tau_list, tau_max):

    tau_below = [tau for tau in tau_list if tau < tau_max]

    return len(tau_below)

#computes the number of doublets with time difference less than tau_max
def ExpectedNumberOfDoublets(N_events, tau_max, mean_rate, fraction_of_sky):

    new_rate = fraction_of_sky*tau_max*mean_rate

    return N_events*(1 - math.exp(-new_rate)*( 1 + new_rate))
#------------------------------
# main
#------------------------------

#important parameters for event generation
N_events = 10000
length_x = 10
length_y = 10
t_min = 0
t_max = 100
mean_rate = N_events/(t_max - t_min)

seed(1)

event_list = GenerateEvents(N_events, length_x, length_y, t_min, t_max)

ordered_event_list = OrderEvents(event_list)

#parameters for tau computation
length_box_x = 10
length_box_y = 10

#ratio between search region and all region
fraction_of_sky = (length_box_x * length_box_y)/(length_x * length_y)

tau_list = ComputeTau(ordered_event_list, length_box_x, length_box_y)

print("Expected value of mean tau", 1/(mean_rate))
print("Observed value of mean tau", np.mean(tau_list))

#define the tau below which the comulative is computed
tau_max = 0.01

print("Expected number of events with tau <", tau_max, "in region with", fraction_of_sky, "of the total region:", N_events*(1 - math.exp(-mean_rate*tau_max)))
print("Expected number of doublets with tau <", tau_max, "in region with", fraction_of_sky, "of the total region:", ExpectedNumberOfDoublets(N_events, tau_max, mean_rate, fraction_of_sky))
print("Observed number of events with tau <", tau_max, "in region with", fraction_of_sky, "of the total region:", ComulativeBelow(tau_list, tau_max))


plt.hist(tau_list, bins=100, range=[min(tau_list), max(tau_list)])
plt.yscale('log')
plt.show()
