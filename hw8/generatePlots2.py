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

basePrefix = 'data/Main2_'
suffix = '_shuffler'
baseOutputPrefix = 'figures/Main2_'
numberOfLogicalCores = 24
maxZ = 16

makeImageFiles = True
#makeImageFiles = False
azimuth = -165
elevation = 9
  
PolynomialOrderStyles = ['Fixed', 'ProportionalToIndex']

for polynomialOrderStyle in PolynomialOrderStyles:
  prefix = basePrefix + polynomialOrderStyle + '_'
  outputPrefix = baseOutputPrefix + polynomialOrderStyle + '_'

  inputSizeData = numpy.loadtxt(open(prefix + 'inputSize' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
  numberOfThreadsData = numpy.loadtxt(open(prefix + 'numberOfThreads' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
  serialTimes = numpy.loadtxt(open(prefix + 'serialTimes' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
  stdThreadTimes = numpy.loadtxt(open(prefix + 'stdThreadTimes' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
  ompStaticTimes = numpy.loadtxt(open(prefix + 'ompStaticTimes' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
  ompDynamicTimes = numpy.loadtxt(open(prefix + 'ompDynamicTimes' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
  ompGuidedTimes = numpy.loadtxt(open(prefix + 'ompGuidedTimes' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
  tbbTimes = numpy.loadtxt(open(prefix + 'tbbTimes' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
  
  fig = plt.figure()
  
  plt.clf()
  ax = fig.gca(projection='3d')
  ax.set_zlim(0, maxZ)
  surf = ax.plot_surface(log10(inputSizeData), numberOfThreadsData, serialTimes / stdThreadTimes, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.5, antialiased=False, shade=True)
  plt.xlabel('log10(inputSize)')
  plt.ylabel('number of threads')
  ax.set_zlabel('speedup')
  plt.title('speedup for std::thread, style %s' % polynomialOrderStyle)
  ax.view_init(elev=elevation, azim=azimuth)
  if (makeImageFiles == True):
    filename = outputPrefix + 'stdThread' + suffix + '.pdf'
    plt.savefig(filename)
    print 'saved file to %s' % filename
  else:
    plt.show()

  plt.clf()
  ax = fig.gca(projection='3d')
  ax.set_zlim(0, maxZ)
  surf = ax.plot_surface(log10(inputSizeData), numberOfThreadsData, serialTimes / ompStaticTimes, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.5, antialiased=False, shade=True)
  plt.xlabel('log10(inputSize)')
  plt.ylabel('number of threads')
  ax.set_zlabel('speedup')
  plt.title('speedup for openmp static, style %s' % polynomialOrderStyle)
  ax.view_init(elev=elevation, azim=azimuth)
  if (makeImageFiles == True):
    filename = outputPrefix + 'ompStatic' + suffix + '.pdf'
    plt.savefig(filename)
    print 'saved file to %s' % filename
  else:
    plt.show()

  plt.clf()
  ax = fig.gca(projection='3d')
  ax.set_zlim(0, maxZ)
  surf = ax.plot_surface(log10(inputSizeData), numberOfThreadsData, serialTimes / ompDynamicTimes, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.5, antialiased=False, shade=True)
  plt.xlabel('log10(inputSize)')
  plt.ylabel('number of threads')
  ax.set_zlabel('speedup')
  plt.title('speedup for openmp dynamic, style %s' % polynomialOrderStyle)
  if (makeImageFiles == True):
    ax.view_init(elev=elevation, azim=azimuth)
    filename = outputPrefix + 'ompDynamic' + suffix + '.pdf'
    plt.savefig(filename)
    print 'saved file to %s' % filename
  else:
    plt.show()

  plt.clf()
  ax = fig.gca(projection='3d')
  ax.set_zlim(0, maxZ)
  surf = ax.plot_surface(log10(inputSizeData), numberOfThreadsData, serialTimes / ompGuidedTimes, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.5, antialiased=False, shade=True)
  plt.xlabel('log10(inputSize)')
  plt.ylabel('number of threads')
  ax.set_zlabel('speedup')
  plt.title('speedup for openmp guided, style %s' % polynomialOrderStyle)
  ax.view_init(elev=elevation, azim=azimuth)
  if (makeImageFiles == True):
    filename = outputPrefix + 'ompGuided' + suffix + '.pdf'
    plt.savefig(filename)
    print 'saved file to %s' % filename
  else:
    plt.show()

  plt.clf()
  ax = fig.gca(projection='3d')
  ax.set_zlim(0, maxZ)
  surf = ax.plot_surface(log10(inputSizeData), numberOfThreadsData, serialTimes / tbbTimes, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.5, antialiased=False, shade=True)
  plt.xlabel('log10(inputSize)')
  plt.ylabel('number of threads')
  ax.set_zlabel('speedup')
  plt.title('speedup for tbb, style %s' % polynomialOrderStyle)
  if (makeImageFiles == True):
    ax.view_init(elev=elevation, azim=azimuth)
    filename = outputPrefix + 'tbb' + suffix + '.pdf'
    plt.savefig(filename)
    print 'saved file to %s' % filename
  else:
    plt.show()

  plt.clf()
  ax = fig.gca(projection='3d')
  #ax.set_zlim(0, numberOfLogicalCores)
  surf = ax.plot_surface(log10(inputSizeData), numberOfThreadsData, ompGuidedTimes / tbbTimes, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.5, antialiased=False, shade=True)
  plt.xlabel('log10(inputSize)')
  plt.ylabel('number of threads')
  ax.set_zlabel('speedup')
  plt.title('speedup of tbb over omp guided, style %s' % polynomialOrderStyle)
  if (makeImageFiles == True):
    ax.view_init(elev=elevation, azim=azimuth)
    filename = outputPrefix + 'tbbVersusOmpGuided' + suffix + '.pdf'
    plt.savefig(filename)
    print 'saved file to %s' % filename
  else:
    plt.show()

  plt.clf()
  ax = fig.gca(projection='3d')
  #ax.set_zlim(0, numberOfLogicalCores)
  surf = ax.plot_surface(log10(inputSizeData), numberOfThreadsData, tbbTimes / ompGuidedTimes, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.5, antialiased=False, shade=True)
  plt.xlabel('log10(inputSize)')
  plt.ylabel('number of threads')
  ax.set_zlabel('speedup')
  plt.title('speedup of omp guided over tbb, style %s' % polynomialOrderStyle)
  if (makeImageFiles == True):
    ax.view_init(elev=elevation, azim=azimuth)
    filename = outputPrefix + 'ompGuidedVersusTbb' + suffix + '.pdf'
    plt.savefig(filename)
    print 'saved file to %s' % filename
  else:
    plt.show()



  stdThreadColor = 'r'
  ompColor = 'k'
  tbbColor = 'b'

  plt.figure()
  plt.xscale('linear')
  legendNames = []
  plt.plot(numberOfThreadsData[0, :], serialTimes[-1, :] / stdThreadTimes[-1, :], color=stdThreadColor, hold='on', linewidth=2)
  legendNames.append('stdThread')
  plt.plot(numberOfThreadsData[0, :], serialTimes[-1, :] / ompGuidedTimes[-1, :], color=ompColor, linestyle='solid', hold='on', linewidth=2)
  legendNames.append('omp guided')
  plt.plot(numberOfThreadsData[0, :], serialTimes[-1, :] / ompStaticTimes[-1, :], color=ompColor, linestyle='dashed', hold='on', linewidth=2)
  legendNames.append('omp static')
  plt.plot(numberOfThreadsData[0, :], serialTimes[-1, :] / ompDynamicTimes[-1, :], color=ompColor, linestyle='dashdot', hold='on', linewidth=2)
  legendNames.append('omp dynamic')
  plt.plot(numberOfThreadsData[0, :], serialTimes[-1, :] / tbbTimes[-1, :], color=tbbColor, hold='on', linewidth=2)
  legendNames.append('tbb')
  plt.title('speedups for size %8.2e, %s' % (inputSizeData[-1, 0], polynomialOrderStyle), fontsize=16)
  plt.xlabel('number of threads', fontsize=16)
  plt.ylabel('speedup', fontsize=16)
  plt.ylim([0, numberOfLogicalCores])
  plt.legend(legendNames, loc='upper left')
  if (makeImageFiles == True):
    filename = outputPrefix + '2d_largestSize' + suffix + '.pdf'
    plt.savefig(filename)
    print 'saved file to %s' % filename
  else:
    plt.show()
  
  plt.figure()
  plt.xscale('linear')
  legendNames = []
  plt.plot(numberOfThreadsData[0, :], serialTimes[0, :] / stdThreadTimes[0, :], color=stdThreadColor, hold='on', linewidth=2)
  legendNames.append('stdThread')
  plt.plot(numberOfThreadsData[0, :], serialTimes[0, :] / ompGuidedTimes[0, :], color=ompColor, linestyle='solid', hold='on', linewidth=2)
  legendNames.append('omp guided')
  plt.plot(numberOfThreadsData[0, :], serialTimes[0, :] / ompStaticTimes[0, :], color=ompColor, linestyle='dashed', hold='on', linewidth=2)
  legendNames.append('omp static')
  plt.plot(numberOfThreadsData[0, :], serialTimes[0, :] / ompDynamicTimes[0, :], color=ompColor, linestyle='dashdot', hold='on', linewidth=2)
  legendNames.append('omp dynamic')
  plt.plot(numberOfThreadsData[0, :], serialTimes[0, :] / tbbTimes[0, :], color=tbbColor, hold='on', linewidth=2)
  legendNames.append('tbb')
  plt.title('speedups for size %8.2e, %s' % (inputSizeData[0, 0], polynomialOrderStyle), fontsize=16)
  plt.xlabel('number of threads', fontsize=16)
  plt.ylabel('speedup', fontsize=16)
  plt.ylim([0, 6])
  plt.legend(legendNames, loc='upper right')
  if (makeImageFiles == True):
    filename = outputPrefix + '2d_smallestSize' + suffix + '.pdf'
    plt.savefig(filename)
    print 'saved file to %s' % filename
  else:
    plt.show()
  
  plt.figure()
  plt.xscale('linear')
  legendNames = []
  middleIndex = int(len(serialTimes[:, 0]) / 2)
  plt.plot(numberOfThreadsData[0, :], serialTimes[middleIndex, :] / stdThreadTimes[middleIndex, :], color=stdThreadColor, hold='on', linewidth=2)
  legendNames.append('stdThread')
  plt.plot(numberOfThreadsData[0, :], serialTimes[middleIndex, :] / ompGuidedTimes[middleIndex, :], color=ompColor, linestyle='solid', hold='on', linewidth=2)
  legendNames.append('omp guided')
  plt.plot(numberOfThreadsData[0, :], serialTimes[middleIndex, :] / ompStaticTimes[middleIndex, :], color=ompColor, linestyle='dashed', hold='on', linewidth=2)
  legendNames.append('omp static')
  plt.plot(numberOfThreadsData[0, :], serialTimes[middleIndex, :] / ompDynamicTimes[middleIndex, :], color=ompColor, linestyle='dashdot', hold='on', linewidth=2)
  legendNames.append('omp dynamic')
  plt.plot(numberOfThreadsData[0, :], serialTimes[middleIndex, :] / tbbTimes[middleIndex, :], color=tbbColor, hold='on', linewidth=2)
  legendNames.append('tbb')
  plt.title('speedups for size %8.2e, %s' % (inputSizeData[middleIndex, 0], polynomialOrderStyle), fontsize=16)
  plt.xlabel('number of threads', fontsize=16)
  plt.ylabel('speedup', fontsize=16)
  plt.ylim([0, numberOfLogicalCores])
  plt.legend(legendNames, loc='upper left')
  if (makeImageFiles == True):
    filename = outputPrefix + '2d_middleSize' + suffix + '.pdf'
    plt.savefig(filename)
    print 'saved file to %s' % filename
  else:
    plt.show()
  
