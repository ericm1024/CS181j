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
azimuth = -23
elevation = 2
  
numberOfPoints = numpy.loadtxt(open(prefix + 'numberOfPoints' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
numberOfCentroids = numpy.loadtxt(open(prefix + 'numberOfCentroids' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
scalar = numpy.loadtxt(open(prefix + 'scalar' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
vectorized = numpy.loadtxt(open(prefix + 'vectorized' + suffix + '.csv','rb'),delimiter=',',skiprows=0)

fig = plt.figure()

plt.clf()
ax = fig.gca(projection='3d')
#ax.set_zlim(-2, 2)
surf = ax.plot_surface(log10(numberOfPoints), numberOfCentroids, scalar / vectorized, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.5, antialiased=False, shade=True)
plt.xlabel('log10(number of points) [-]')
plt.ylabel('numberOfCentroids [-]')
ax.set_zlabel('speedup from vectorization')
plt.title('speedup of vectorized k-means over scalar')
ax.view_init(elev=elevation, azim=azimuth)
if (makeImageFiles == True):
  filename = outputPrefix + 'speedup' + suffix + '.pdf'
  plt.savefig(filename)
  print 'saved file to %s' % filename
else:
  plt.show()
