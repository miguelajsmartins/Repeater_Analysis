#class to summarise information about events
import numpy as np
import sys

class Event:
    def __init__(self,alpha,delta,time,energy):
        self.alpha = alpha
        self.delta = delta
        self.time = time
        self.energy = energy

    def __repr__(self):
        return '[%s, %s, %s, %s]' % (self.alpha, self.delta, self.time, self.energy)

    def __str__(self):
        return '[%s, %s, %s, %s]' % (self.alpha, self.delta, self.time, self.energy)

    def __gt__(self,other):
        if(self.time > other.time):
            return True
        else:
            return False

    def __lt__(self,other):
        if(self.time < other.time):
            return True
        else:
            return False

    def __eq__(self,other):
        if(self.alpha == other.alpha and self.delta == other.delta and self.time == other.time and self.energy == other.energy):
            return True
        else:
            return False

    def to_array(self):
        evt = [self.alpha, self.delta, self.time, self.energy]
        evt = np.array(evt)

        return evt

    def from_array(self,array):
        if ( len(array) != 4 ):
            sys.exit("Array must have 4 elements")
        else:
            self.alpha = array[0]
            self.delta = array[1]
            self.time = array[2]
            self.energy = array[3]

        return self
