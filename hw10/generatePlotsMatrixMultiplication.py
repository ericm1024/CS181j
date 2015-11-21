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

prefix = 'data/MatrixMultiplication_'
suffix = '_shuffler'
outputPrefix = 'figures/MatrixMultiplication_'

data = numpy.loadtxt(open(prefix + 'results' + suffix + '.csv','rb'),delimiter=',',skiprows=0)

colors      = cm.jet(numpy.linspace(1, 0, 5))

plt.figure(figsize=(12, 8))
ax = plt.subplot(111)
plt.xscale('log')
legendNames = []
plt.plot(data[:,0], data[:,1] / data[:,5], color=colors[0], hold='on', linewidth=2)
legendNames.append('gpu cublas')
plt.plot(data[:,0], data[:,1] / data[:,2], color=colors[1], hold='on', linewidth=2)
legendNames.append('gpu naive row*row')
plt.plot(data[:,0], data[:,1] / data[:,3], color=colors[2], hold='on', linewidth=2)
legendNames.append('gpu row st tiled mult global')
plt.plot(data[:,0], data[:,1] / data[:,4], color=colors[3], hold='on', linewidth=2)
legendNames.append('gpu naive row*col')
plt.plot([min(data[:, 0]), max(data[:, 0])], [1, 1], '--', color=colors[4], hold='on', linewidth=2)
legendNames.append('unity')
plt.title('matrix multiplication speedup over serial cpu', fontsize=16)
plt.xlabel('matrixSize', fontsize=16)
plt.ylabel('speedup', fontsize=16)
plt.xlim([min(data[:, 0]), max(data[:, 0])])
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.70, box.height])
ax.legend(legendNames, loc='center right', bbox_to_anchor=(1.60, 0.5))
filename = outputPrefix + 'matrixMultiplication' + suffix + '.pdf'
plt.savefig(filename)
print 'saved file to %s' % filename
