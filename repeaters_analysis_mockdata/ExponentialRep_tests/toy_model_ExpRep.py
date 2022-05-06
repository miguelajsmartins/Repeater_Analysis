import numpy as np
import matplotlib.pyplot as plt
import math

from random import random
from random import seed

from datetime import datetime

seed(datetime.now())

time = []
time_diff = []
tau = 10

for i in range(100000):
    u = random()
    v = random()

    time.append(-tau*math.log(1 - u))

    time_diff.append(tau*abs(math.log(1 - u) - tau*math.log(1 - v)))

plt.hist(time, bins=200, alpha = 0.5, range=[0,1000])
plt.hist(time_diff, bins=200, alpha = 0.5, range=[0,1000])

print(np.mean(time))
print(np.mean(time_diff))

plt.show()
