import numpy as np
import numpy.ma as ma

def get_principal_argument(x):

    principal_argument = np.angle(np.exp(1j*x))

    is_positive = principal_argument >= 0

    return ma.masked_array(principal_argument, mask = np.logical_not(is_positive)).filled(fill_value = 2*np.pi + principal_argument)

array = np.array([-7, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7])

print(get_principal_argument(array))
