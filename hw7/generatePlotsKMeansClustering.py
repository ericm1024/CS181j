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

prefix = 'data/KMeansClustering_'
suffix = '_shuffler'
outputPrefix = 'figures/KMeansClustering_'

makeImageFiles = True
#makeImageFiles = False
azimuth = 180
elevation = 10
  
numberOfPoints = numpy.loadtxt(open(prefix + 'numberOfPoints' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
numberOfThreads = numpy.loadtxt(open(prefix + 'numberOfThreads' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
serial = numpy.loadtxt(open(prefix + 'serial' + suffix + '.csv','rb'),delimiter=',',skiprows=0)

for name in ['omp_locked', 'omp_thread_local', 'omp_atomic' \
             , 'omp_moved_pragma', 'omp_atomic_false_sharing' \
             ]:
  data = numpy.loadtxt(open(prefix + name + suffix + '.csv','rb'),delimiter=',',skiprows=0)
  for i in range(2):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    if (i == 0):
      zValues = log10(serial / data)
      zlabel = 'log10(speedup from '+name+') [-]'
      logOrLinear = 'log'
    else:
      zValues = serial / data
      zlabel = 'speedup from '+name+' [-]'
      logOrLinear = 'linear'
    surf = ax.plot_surface(log10(numberOfPoints), numberOfThreads, zValues, rstride=1, \
                           cstride=1, cmap=cm.coolwarm, linewidth=0.5, antialiased=False, shade=True)
    plt.xlabel('log10(number of points) [-]')
    plt.ylabel('numberOfThreads [-]')
    ax.set_zlabel(zlabel)
    plt.title('speedup of '+ name + ' k-means over serial')
    ax.view_init(elev=elevation, azim=azimuth)
    if (makeImageFiles == True):
      filename = outputPrefix+name+'VersusSerial_' + logOrLinear + suffix + '.pdf'
      plt.savefig(filename)
      print 'saved file to %s' % filename
    else:
      plt.show()
