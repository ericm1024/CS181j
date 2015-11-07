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

prefix = 'data/FindIndexOfClosestPoint_'
suffix = '_shuffler'
outputPrefix = 'figures/FindIndexOfClosestPoint_'

makeImageFiles = True
#makeImageFiles = False
azimuth = -166
elevation = 7
  
numberOfPoints = numpy.loadtxt(open(prefix + 'numberOfPoints' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
numberOfThreads = numpy.loadtxt(open(prefix + 'numberOfThreads' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
serial = numpy.loadtxt(open(prefix + 'serial' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
omp = numpy.loadtxt(open(prefix + 'omp' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
tbb = numpy.loadtxt(open(prefix + 'tbb' + suffix + '.csv','rb'),delimiter=',',skiprows=0)

threadingFlavors = ['omp', 'tbb']

for threadingFlavor in threadingFlavors:
  if (threadingFlavor == 'omp'):
    data = omp
  else:
    data = tbb
  for logOrLinear in range(2):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #ax.set_zlim(-2, 2)
    if (logOrLinear == 0):
      ax.set_zlim(-1, 1)
      zValues = log10(serial / data)
      zlabel = 'log10(speedup from threading) [-]'
      logOrLinear = 'log'
    else:
      zValues = serial / data
      zlabel = 'speedup from threading [-]'
      logOrLinear = 'linear'
    surf = ax.plot_surface(log10(numberOfPoints), numberOfThreads, zValues, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.5, antialiased=False, shade=True)
    plt.xlabel('log10(number of points) [-]')
    plt.ylabel('numberOfThreads [-]')
    ax.set_zlabel(zlabel)
    plt.title('speedup of ' + threadingFlavor + ' over serial, FindIndexOfClosestPoint')
    ax.view_init(elev=elevation, azim=azimuth)
    if (makeImageFiles == True):
      filename = outputPrefix + threadingFlavor + 'VersusSerial_' + logOrLinear + suffix + '.pdf'
      plt.savefig(filename)
      print 'saved file to %s' % filename
    else:
      plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(log10(numberOfPoints), numberOfThreads, omp / tbb, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.5, antialiased=False, shade=True)
plt.xlabel('log10(number of points) [-]')
plt.ylabel('numberOfThreads [-]')
ax.set_zlabel(zlabel)
plt.title('speedup of tbb over omp, FindIndexOfClosestPoint')
ax.view_init(elev=elevation, azim=azimuth)
if (makeImageFiles == True):
  filename = outputPrefix + 'tbbVersusOmp' + suffix + '.pdf'
  plt.savefig(filename)
  print 'saved file to %s' % filename
else:
  plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(log10(numberOfPoints), numberOfThreads, tbb / omp, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.5, antialiased=False, shade=True)
plt.xlabel('log10(number of points) [-]')
plt.ylabel('numberOfThreads [-]')
ax.set_zlabel(zlabel)
plt.title('speedup of omp over tbb, FindIndexOfClosestPoint')
ax.view_init(elev=elevation, azim=azimuth)
if (makeImageFiles == True):
  filename = outputPrefix + 'ompVersusTbb' + suffix + '.pdf'
  plt.savefig(filename)
  print 'saved file to %s' % filename
else:
  plt.show()
