import math
import os
import sys
import numpy
import scipy
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
from mpl_toolkits.mplot3d import Axes3D
from numpy import log10
from multiprocessing import Process

plt.figure(figsize=(9,6))

data = numpy.loadtxt(open('data/FrodosChocolates.csv','rb'),delimiter=',',skiprows=0)
xValues = numpy.arange(len(data))
plt.bar(xValues, data)
plt.xlabel('rank')
plt.ylabel('chocolate-holding frequency')

filename = 'figures/FrodosChocolates.pdf'
plt.savefig(filename)
print 'saved file to %s' % filename
