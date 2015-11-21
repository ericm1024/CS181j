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
azimuth = -132
elevation = 3
  
numberOfPoints = numpy.loadtxt(open(prefix + 'numberOfPoints' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
numberOfCentroids = numpy.loadtxt(open(prefix + 'numberOfCentroids' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
cpu = numpy.loadtxt(open(prefix + 'cpu' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
gpu_SoA = numpy.loadtxt(open(prefix + 'gpu_SoA' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
gpuPointStyle = 'SoA'

for logOrLinear in range(2):
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  #ax.set_zlim(-2, 2)
  if (logOrLinear == 0):
    zValues = log10(cpu / gpu_SoA)
    zlabel = 'log10(speedup from using the gpu) [-]'
    logOrLinear = 'log'
  else:
    zValues = cpu / gpu_SoA
    zlabel = 'speedup from using the gpu [-]'
    logOrLinear = 'linear'
  surf = ax.plot_surface(log10(numberOfPoints), numberOfCentroids, zValues, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.5, antialiased=False, shade=True)
  plt.xlabel('log10(number of points) [-]')
  plt.ylabel('number of centroids [-]')
  ax.set_zlabel(zlabel)
  plt.title('speedup of gpu k-means over serial cpu, point style ' + gpuPointStyle)
  ax.view_init(elev=elevation, azim=azimuth)
  if (makeImageFiles == True):
    filename = outputPrefix + gpuPointStyle + 'VersusSerial_' + logOrLinear + suffix + '.pdf'
    plt.savefig(filename)
    print 'saved file to %s' % filename
  else:
    plt.show()
