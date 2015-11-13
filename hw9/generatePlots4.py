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

prefix = 'data/Main4_'
suffix = '_shuffler'
outputPrefix = 'figures/Main4_'

inputSizeData = numpy.loadtxt(open(prefix + 'inputSizeMatrixForPlotting' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
polynomialOrderData = numpy.loadtxt(open(prefix + 'polynomialOrderMatrixForPlotting' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
cpuTimes = numpy.loadtxt(open(prefix + 'cpuTimes' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
gpuTimes = numpy.loadtxt(open(prefix + 'gpuTimes' + suffix + '.csv','rb'),delimiter=',',skiprows=0)

fig = plt.figure()
ax = fig.gca(projection='3d')
#ax.set_zlim(-2, 1)
surf = ax.plot_surface(log10(inputSizeData), polynomialOrderData, log10(cpuTimes / gpuTimes), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.5, antialiased=False, shade=True)
plt.xlabel('log10(inputSize)')
plt.ylabel('polynomialOrder')
ax.set_zlabel('log10(speedup)')
plt.title('speedup of gpu over serial cpu')
ax.view_init(elev=5, azim=-38)
filename = outputPrefix + 'versusInputSizeAndOrder' + suffix + '.pdf'
plt.savefig(filename)
print 'saved file to %s' % filename

gpuSpeedupsVersusBlocksAndThreads = numpy.loadtxt(open(prefix + 'gpuSpeedupsVersusBlocksAndThreads' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
maxNumberOfBlocks = numpy.loadtxt(open(prefix + 'maxNumberOfBlocks' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
numberOfThreadsPerBlock = numpy.loadtxt(open(prefix + 'numberOfThreadsPerBlock' + suffix + '.csv','rb'),delimiter=',',skiprows=0)

fig = plt.figure()
ax = fig.gca(projection='3d')
#ax.set_zlim(-1.5, 1.5)
surf = ax.plot_surface(log10(maxNumberOfBlocks), log10(numberOfThreadsPerBlock), log10(gpuSpeedupsVersusBlocksAndThreads), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.5, antialiased=False, shade=True)
plt.xlabel('log10(maxNumberOfBlocks)')
plt.ylabel('log10(numberOfThreadsPerBlock)')
ax.set_zlabel('log10(speedup)')
plt.title('speedup of gpu over serial cpu')
ax.view_init(elev=10, azim=-39)
filename = outputPrefix + 'versusBlocksAndThreads' + suffix + '.pdf'
plt.savefig(filename)
print 'saved file to %s' % filename
