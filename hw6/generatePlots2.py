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

prefix = 'data/Main2_'
suffix = '_shuffler'
outputPrefix = 'figures/Main2_'
maxZ = 20

numberOfIntervalsData = numpy.loadtxt(open(prefix + 'numberOfIntervals' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
numberOfThreadsData = numpy.loadtxt(open(prefix + 'numberOfThreads' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
serial = numpy.loadtxt(open(prefix + 'serial' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
stdThread = numpy.loadtxt(open(prefix + 'stdThread' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
omp = numpy.loadtxt(open(prefix + 'omp' + suffix + '.csv','rb'),delimiter=',',skiprows=0)

makeImageFiles = True
#makeImageFiles = False
azimuth = -164
elevation = 16

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.view_init(elev=elevation, azim=azimuth)
ax.set_zlim(0, maxZ)
surf = ax.plot_surface(log10(numberOfIntervalsData), numberOfThreadsData, serial / stdThread, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.5, antialiased=False)
plt.xlabel('log10(number of intervals)')
plt.ylabel('number of threads')
ax.set_zlabel('speedup')
plt.title('speedup from using std::thread')
if (makeImageFiles == True):
  filename = outputPrefix + 'stdThreadVersusSerial' + suffix + '.pdf'
  plt.savefig(filename)
  print 'saved file to %s' % filename
else:
  plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.view_init(elev=elevation, azim=azimuth)
ax.set_zlim(0, maxZ)
surf = ax.plot_surface(log10(numberOfIntervalsData), numberOfThreadsData, serial / omp, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.5, antialiased=False)
plt.xlabel('log10(number of intervals)')
plt.ylabel('number of threads')
ax.set_zlabel('speedup')
plt.title('speedup from using omp')
if (makeImageFiles == True):
  filename = outputPrefix + 'ompVersusSerial' + suffix + '.pdf'
  plt.savefig(filename)
  print 'saved file to %s' % filename
else:
  plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.view_init(elev=elevation, azim=azimuth)
#ax.set_zlim(0, maxZ)
surf = ax.plot_surface(log10(numberOfIntervalsData), numberOfThreadsData, stdThread / omp, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.5, antialiased=False)
plt.xlabel('log10(number of intervals)')
plt.ylabel('number of threads')
ax.set_zlabel('speedup')
plt.title('speedup from using omp over using std::thread')
if (makeImageFiles == True):
  filename = outputPrefix + 'ompVersusStdThread' + suffix + '.pdf'
  plt.savefig(filename)
  print 'saved file to %s' % filename
else:
  plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.view_init(elev=elevation, azim=azimuth)
#ax.set_zlim(0, maxZ)
surf = ax.plot_surface(log10(numberOfIntervalsData), numberOfThreadsData, omp / stdThread, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.5, antialiased=False)
plt.xlabel('log10(number of intervals)')
plt.ylabel('number of threads')
ax.set_zlabel('speedup')
plt.title('speedup from using std::thread over using omp')
if (makeImageFiles == True):
  filename = outputPrefix + 'stdThreadVersusOmp' + suffix + '.pdf'
  plt.savefig(filename)
  print 'saved file to %s' % filename
else:
  plt.show()
