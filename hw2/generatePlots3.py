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

prefix = 'data/Main3_'
suffix = '_shuffler'
outputPrefix = 'figures/Main3_'

makeImageFiles = True
#makeImageFiles = False
azimuth = -121
elevation = 5
  
duplicationRate = numpy.loadtxt(open(prefix + 'duplicationRate' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
power = numpy.loadtxt(open(prefix + 'power' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
unMemoized = numpy.loadtxt(open(prefix + 'unMemoized' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
mapMemoized = numpy.loadtxt(open(prefix + 'mapMemoized' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
arrayMemoized = numpy.loadtxt(open(prefix + 'arrayMemoized' + suffix + '.csv','rb'),delimiter=',',skiprows=0)

fig = plt.figure()

plt.clf()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(duplicationRate, power, log10(unMemoized / mapMemoized), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.5, antialiased=False, shade=True)
plt.xlabel('duplication rate [-]')
plt.ylabel('power [-]')
ax.set_zlabel('log10(speedup)')
plt.title('speedup of map memoized over unmemoized')
ax.view_init(elev=elevation, azim=azimuth)
if (makeImageFiles == True):
  filename = outputPrefix + 'mapMemoized_vs_unMemoized' + suffix + '.pdf'
  plt.savefig(filename)
  print 'saved file to %s' % filename
else:
  plt.show()

plt.clf()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(duplicationRate, power, log10(unMemoized / arrayMemoized), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.5, antialiased=False, shade=True)
plt.xlabel('duplication rate [-]')
plt.ylabel('power [-]')
ax.set_zlabel('log10(speedup)')
plt.title('speedup of array memoized over unmemoized')
ax.view_init(elev=elevation, azim=azimuth)
if (makeImageFiles == True):
  filename = outputPrefix + 'arrayMemoized_vs_unMemoized' + suffix + '.pdf'
  plt.savefig(filename)
  print 'saved file to %s' % filename
else:
  plt.show()

plt.clf()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(duplicationRate, power, log10(mapMemoized / arrayMemoized), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.5, antialiased=False, shade=True)
plt.xlabel('duplication rate [-]')
plt.ylabel('power [-]')
ax.set_zlabel('log10(speedup)')
plt.title('speedup of array memoized over map memoized')
ax.view_init(elev=elevation, azim=azimuth)
if (makeImageFiles == True):
  filename = outputPrefix + 'arrayMemoized_vs_mapMemoized' + suffix + '.pdf'
  plt.savefig(filename)
  print 'saved file to %s' % filename
else:
  plt.show()
