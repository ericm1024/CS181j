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
#maxZ = 5

#lockless
serialTimes = numpy.loadtxt(open(prefix + 'serialTimes' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
reductionTimes = numpy.loadtxt(open(prefix + 'reductionTimes' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
atomicsTimes = numpy.loadtxt(open(prefix + 'atomicsTimes' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
tbbReductionTimes = numpy.loadtxt(open(prefix + 'tbbReductionTimes' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
tbbAtomicsTimes = numpy.loadtxt(open(prefix + 'tbbAtomicsTimes' + suffix + '.csv','rb'),delimiter=',',skiprows=0)

lockBucketSizeData = numpy.loadtxt(open(prefix + 'lockBucketSize' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
numberOfThreadsData = numpy.loadtxt(open(prefix + 'numberOfThreads' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
serialMatrixTimes = numpy.loadtxt(open(prefix + 'serialMatrixTimes' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
atomicFlagLocksTimes = numpy.loadtxt(open(prefix + 'atomicFlagLocksTimes' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
tbbLocksTimes = numpy.loadtxt(open(prefix + 'tbbLocksTimes' + suffix + '.csv','rb'),delimiter=',',skiprows=0)

makeImageFiles = True
elevation=5
azimuth=100

numberOfThreadsColumn = numberOfThreadsData[:, 0]

colors = cm.jet(numpy.linspace(1, 0, 5))

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.view_init(elev=elevation, azim=azimuth)
ax.set_zlim(0, 10)
surf = ax.plot_surface(numberOfThreadsData, log10(lockBucketSizeData), serialMatrixTimes / atomicFlagLocksTimes, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.5, antialiased=False, shade=True)
plt.xlabel('number of threads')
plt.ylabel('log10(lockBucketSize)')
ax.set_zlabel('speedup')
plt.title('speedup for non-tbb locks')
if (makeImageFiles == True):
  filename = outputPrefix + 'atomicFlagLocks' + suffix + '.pdf'
  plt.savefig(filename)
  print 'saved file to %s' % filename
else:
  plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.view_init(elev=elevation, azim=azimuth)
ax.set_zlim(0, 10)
surf = ax.plot_surface(numberOfThreadsData, log10(lockBucketSizeData), serialMatrixTimes / tbbLocksTimes, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.5, antialiased=False, shade=True)
plt.xlabel('number of threads')
plt.ylabel('log10(lockBucketSize)')
ax.set_zlabel('speedup')
plt.title('speedup for tbb locks')
if (makeImageFiles == True):
  filename = outputPrefix + 'tbbLocks' + suffix + '.pdf'
  plt.savefig(filename)
  print 'saved file to %s' % filename
else:
  plt.show()


#summary
plt.figure(figsize=(12, 6))
ax = plt.subplot(111)
plt.xscale('linear')
legendNames = []
plt.plot(numberOfThreadsColumn, serialTimes / tbbReductionTimes, color='k', hold='on', linewidth=2)
legendNames.append('tbb reduction')
plt.plot(numberOfThreadsColumn, serialTimes / reductionTimes, '--', color='k', hold='on', linewidth=2)
legendNames.append('reduction')
plt.plot(numberOfThreadsColumn, serialTimes / tbbAtomicsTimes, color='b', hold='on', linewidth=2)
legendNames.append('tbb atomics')
plt.plot(numberOfThreadsColumn, serialTimes / atomicsTimes, '--', color='b', hold='on', linewidth=2)
legendNames.append('atomics')
#TODO: change the column of the locks times to whatever is best
plt.plot(numberOfThreadsColumn, serialTimes / tbbLocksTimes[:, 0], color='r', hold='on', linewidth=2)
legendNames.append('tbb locks')
plt.plot(numberOfThreadsColumn, serialTimes / atomicFlagLocksTimes[:, 0], '--', color='r', hold='on', linewidth=2)
legendNames.append('atomic flag locks')
plt.title('speedup of threading methods', fontsize=16)
plt.xlabel('number of threads', fontsize=16)
plt.ylabel('speedup', fontsize=16)
plt.ylim([0, 20])
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
ax.legend(legendNames, loc='center right', bbox_to_anchor=(1.5, 0.5))
if (makeImageFiles == True):
  filename = outputPrefix + 'summary' + suffix + '.pdf'
  plt.savefig(filename)
  print 'saved file to %s' % filename
else:
  plt.show()

