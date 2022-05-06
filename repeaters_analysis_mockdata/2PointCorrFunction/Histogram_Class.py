import numpy as np

class Histogram:
    def __init__(self, array, nbins):
        self.data = array
        self.nbins = nbins

        self.bin_contents, self.bin_edges = np.histogram(array, nbins)

    def GetBins(self):
        return self.
