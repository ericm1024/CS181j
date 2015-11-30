import math
import os
import sys
import numpy
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import csv
from mpl_toolkits.mplot3d import Axes3D
from numpy import log10

prefix = 'data/ManyMatrixMultiplications_'
suffix = '_shuffler'
outputPrefix = 'figures/ManyMatrixMultiplications_'

filename = prefix + 'times' + suffix + '.csv'
data = numpy.loadtxt(open(filename,'rb'),delimiter=',',skiprows=1)
with open(filename, 'r') as f:
  flavorNames = f.readline()
  flavorNames = flavorNames.strip().split(',')

colors      = cm.jet(numpy.linspace(1, 0, len(flavorNames)-2))

legendNames = flavorNames[3:]
plt.figure(figsize=(12, 6))
ax = plt.subplot(111)
plt.xscale('log')
#plt.yscale('log')
for i in range(len(legendNames)):
  print 'plotting \"%s\"' % legendNames[i]
  plt.plot(data[:,0], data[:,2] / data[:,i+3], color=colors[i+1], hold='on', linewidth=2)
plt.title('matrix multiplication speedup', fontsize=16)
plt.xlabel('matrixSize', fontsize=16)
plt.ylabel('speedup', fontsize=16)
plt.xlim([min(data[:, 0]), max(data[:, 0])])
plt.ylim([0, 20])
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.70, box.height])
ax.legend(legendNames, loc='upper left', bbox_to_anchor=(1.00, 0.8))
filename = outputPrefix + 'speedup' + suffix + '.pdf'
plt.savefig(filename)
print 'saved file to %s' % filename
