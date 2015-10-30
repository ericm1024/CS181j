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

prefix = 'data/MaximalIndependentSet_'
suffix = '_shuffler'
outputPrefix = 'figures/MaximalIndependentSet_'
maxZ = 20

numberOfVerticesData = numpy.loadtxt(open(prefix + 'numberOfVertices' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
numberOfThreadsData = numpy.loadtxt(open(prefix + 'numberOfThreads' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
serial = numpy.loadtxt(open(prefix + 'serial' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
serialMask = numpy.loadtxt(open(prefix + 'serialMask' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
threadedMask = numpy.loadtxt(open(prefix + 'threadedMask' + suffix + '.csv','rb'),delimiter=',',skiprows=0)

makeImageFiles = True
#makeImageFiles = False
azimuth = 150
elevation = 6

fig = plt.figure('serialSpeedup', figsize=(9,6))
legendNames = []
plt.xscale('log')
plt.plot(numberOfVerticesData[:,1], serial[:,1] / serialMask[:,1], color='k', linestyle='solid', linewidth=2, hold='on')
plt.xlabel('number of vertices [-]', fontsize=16)
plt.ylabel('speedup [-]', fontsize=16)
plt.grid(b=True, which='major', color='k', linestyle='dotted')
plt.title('speedup of mask over serial set version' , fontsize=16)
filename = outputPrefix + 'serialSpeedup' + suffix + '.pdf'
plt.savefig(filename)
print 'saved file to %s' % filename

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.view_init(elev=elevation, azim=azimuth)
surf = ax.plot_surface(log10(numberOfVerticesData), numberOfThreadsData, serialMask / threadedMask, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.5, antialiased=False)
plt.xlabel('log10(number of vertices)')
plt.ylabel('number of threads')
ax.set_zlabel('speedup')
plt.title('speedup from using threads')
if (makeImageFiles == True):
  filename = outputPrefix + 'threadedMaskVersusSerialMask' + suffix + '.pdf'
  plt.savefig(filename)
  print 'saved file to %s' % filename
else:
  plt.show()
