import numpy as np

def unsorted_search(array, values):

    sorted_indices = array.argsort()
    array = array[sorted_indices]
    unsorted_indices = sorted_indices[np.searchsorted(array, values)]

    return unsorted_indices
