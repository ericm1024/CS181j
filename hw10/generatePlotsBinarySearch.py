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

prefix = 'data/BinarySearch_'
suffix = '_shuffler'
outputPrefix = 'figures/BinarySearch_'

stl = numpy.loadtxt(open(prefix + 'stl' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
cpu = numpy.loadtxt(open(prefix + 'cpu' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
gpu = numpy.loadtxt(open(prefix + 'gpu' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
inputSize = numpy.loadtxt(open(prefix + 'inputSize' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
tableSize = numpy.loadtxt(open(prefix + 'tableSize' + suffix + '.csv','rb'),delimiter=',',skiprows=0)

makeImageFiles = True
#makeImageFiles = False
elevation = 9
azimuth = 153

fig = plt.figure()
ax = fig.gca(projection='3d')
#ax.set_zlim(-0.5, 0.5)
surf = ax.plot_surface(log10(inputSize), log10(tableSize), stl / cpu, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.5, antialiased=False, shade=True)
plt.xlabel('log10(inputSize) [-]')
plt.ylabel('log10(tableSize) [-]')
ax.set_zlabel('speedup [-]')
plt.title('speedup of custom cpu over stl')
ax.view_init(elev=elevation, azim=azimuth)
if (makeImageFiles == True):
  filename = outputPrefix + 'cpu' + suffix + '.pdf'
  plt.savefig(filename)
  print 'saved file to %s' % filename
else:
  plt.show()

for logOrLinear in range(2):
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  #ax.set_zlim(-2, 2)
  if (logOrLinear == 0):
    zValues = log10(stl / gpu)
    zLabel = 'log10(speedup) [-]'
    logOrLinear = 'log'
    ax.set_zlim(-1.0, 2.0)
  else:
    zValues = stl / gpu
    zLabel = 'speedup [-]'
    logOrLinear = 'linear'
    #ax.set_zlim(-1.0, 2.0)
  surf = ax.plot_surface(log10(inputSize), log10(tableSize), zValues, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.5, antialiased=False, shade=True)
  plt.xlabel('log10(inputSize) [-]')
  plt.ylabel('log10(tableSize) [-]')
  ax.set_zlabel(zLabel)
  plt.title('speedup of gpu over stl')
  ax.view_init(elev=elevation, azim=azimuth)
  if (makeImageFiles == True):
    filename = outputPrefix + 'gpu_' + logOrLinear + suffix + '.pdf'
    plt.savefig(filename)
    print 'saved file to %s' % filename
  else:
    plt.show()
