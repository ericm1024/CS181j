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

prefix = 'data/ReductionRace_'
outputPrefix = 'figures/ReductionRace_'
maxNumbersOfBlocks = [80, 800, 8000]

for maxNumberOfBlocks in maxNumbersOfBlocks:
  suffix = '_shuffler_%05d' % maxNumberOfBlocks
  filename = prefix + 'times' + suffix + '.csv'
  data = numpy.loadtxt(open(filename,'rb'),delimiter=',',skiprows=1)
  with open(filename, 'r') as f:
    flavorNames = f.readline()
    flavorNames = flavorNames.strip().split(',')
  
  colors      = cm.jet(numpy.linspace(1, 0, len(flavorNames)-1))
  
  legendNames = flavorNames[2:]
  plt.figure(figsize=(12, 6))
  ax = plt.subplot(111)
  plt.xscale('log')
  plt.yscale('log')
  for i in range(len(legendNames)):
    print 'plotting \"%s\"' % legendNames[i]
    plt.plot(data[:,0], data[:,1] / data[:,i+2], color=colors[i], hold='on', linewidth=2)
  plt.plot([min(data[:,0]), max(data[:,0])], [1,1], '--', color='k', hold='on', linewidth=2)
  plt.title('array reduction speedup, %d blocks' % maxNumberOfBlocks, fontsize=16)
  plt.xlabel('array size [-]', fontsize=16)
  plt.ylabel('speedup [-]', fontsize=16)
  plt.xlim([min(data[:, 0]), max(data[:, 0])])
  #plt.ylim([1e-2, 1e3])
  box = ax.get_position()
  ax.set_position([box.x0, box.y0, box.width * 0.70, box.height])
  legendNames.append('unity')
  ax.legend(legendNames, loc='upper left', bbox_to_anchor=(1.00, 0.8))
  filename = outputPrefix + 'speedup' + suffix + '.pdf'
  plt.savefig(filename)
  print 'saved file to %s' % filename

suffix = '_shuffler'
cpuTimes = numpy.loadtxt(open(prefix + 'cpuTimes' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
gpuTimes = numpy.loadtxt(open(prefix + 'gpuTimes' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
maxNumberOfBlocks = numpy.loadtxt(open(prefix + 'maxNumberOfBlocks' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
numberOfThreadsPerBlock = numpy.loadtxt(open(prefix + 'numberOfThreadsPerBlock' + suffix + '.csv','rb'),delimiter=',',skiprows=0)

fig = plt.figure()
ax = fig.gca(projection='3d')
#ax.set_zlim(-1.5, 1.5)
surf = ax.plot_surface(log10(maxNumberOfBlocks), log10(numberOfThreadsPerBlock), log10(cpuTimes / gpuTimes), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.5, antialiased=False, shade=True)
plt.xlabel('log10(maxNumberOfBlocks)')
plt.ylabel('log10(numberOfThreadsPerBlock)')
ax.set_zlabel('speedup [-]')
plt.title('speedup of gpu over serial cpu')
ax.view_init(elev=8, azim=134)
filename = outputPrefix + 'gpuSpeedups' + suffix + '.pdf'
plt.savefig(filename)
print 'saved file to %s' % filename
