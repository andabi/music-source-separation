# -*- coding: utf-8 -*-
#!/usr/bin/env python
'''
By Dabi Ahn. andabi412@gmail.com.
https://www.github.com/andabi
'''

from __future__ import division

import numpy as np

class Diff(object):
    def __init__(self, v=0.):
        self.value = v
        self.diff = 0.

    def update(self, v):
        if self.value:
            diff = (v / self.value - 1)
            self.diff = diff
        self.value = v


def shape(tensor):
    s = tensor.get_shape()
    return tuple([s[i].value for i in range(0, len(s))])


# TODO general pretty print
def pretty_list(list):
    return ', '.join(list)

def pretty_dict(dict):
    return '\n'.join('{} : {}'.format(k, v) for k, v in dict.items())

def closest_power_of_two(target):
    if target > 1:
        for i in range(1, int(target)):
            if (2 ** i >= target):
                pwr = 2 ** i
                break
        if abs(pwr - target) < abs(pwr/2 - target):
            return pwr
        else:
            return int(pwr / 2)
    else:
        return 1

# Write the nd array to txtfile
def nd_array_to_txt(filename, data):
    path = filename + '.txt'
    file = open(path, 'w')
    with file as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write('# Array shape: {0}\n'.format(data.shape))

        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        for data_slice in data:

            # The formatting string indicates that I'm writing out
            # the values in left-justified columns 7 characters in width
            # with 2 decimal places.
            np.savetxt(outfile, data_slice, fmt='%-7.2f')

            # Writing out a break to indicate different slices...
            outfile.write('# New slice\n')
