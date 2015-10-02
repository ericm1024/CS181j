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

prefix = 'data/Main1_'
outputPrefix = 'figures/'
suffix = '_shuffler'

if (os.path.isdir('figures') == False):
  print 'please make figures directory'
  sys.exit(1)

output = numpy.loadtxt(open(prefix + 'data' + suffix + '.csv','rb'),delimiter=',',skiprows=1)

numberOfNeighbors                   = output[:, 0]
vector_staticObject                 = output[:, 1]
vector_dynamicObject                = output[:, 2]
vector_arrayOfPointersObject        = output[:, 3]
vector_vectorOfPointersObject       = output[:, 4]
vector_listObject                   = output[:, 5]
set_staticObject                    = output[:, 6]
set_dynamicObject                   = output[:, 7]
set_arrayOfPointersObject           = output[:, 8]
set_vectorOfPointersObject          = output[:, 9]
set_listObject                      = output[:,10]
vector_staticObject_improved        = output[:,11]
set_vectorOfPointersObject_improved = output[:,12]

colors = cm.jet(numpy.linspace(1, 0, 5))
colorIter = iter(colors)

for i in range(2):
  plt.figure(figsize=(12,6))
  ax = plt.subplot(111)
  plt.xscale('log')
  if (i == 0):
    plt.yscale('log')
  legendNames = []
  maxSlowdown = 1
  slowdown = set_listObject / vector_staticObject
  maxSlowdown = max(maxSlowdown, numpy.amax(slowdown))
  plt.plot(numberOfNeighbors, slowdown, linestyle='solid', color=colors[0], hold='on', linewidth=2)
  legendNames.append('set_listObject')
  slowdown = set_vectorOfPointersObject / vector_staticObject
  maxSlowdown = max(maxSlowdown, numpy.amax(slowdown))
  plt.plot(numberOfNeighbors, slowdown, linestyle='solid', color=colors[1], hold='on', linewidth=2)
  legendNames.append('set_vectorOfPointersObject')
  slowdown = set_arrayOfPointersObject / vector_staticObject
  maxSlowdown = max(maxSlowdown, numpy.amax(slowdown))
  plt.plot(numberOfNeighbors, slowdown, linestyle='solid', color=colors[2], hold='on', linewidth=2)
  legendNames.append('set_arrayOfPointersObject')
  slowdown = set_dynamicObject / vector_staticObject
  maxSlowdown = max(maxSlowdown, numpy.amax(slowdown))
  plt.plot(numberOfNeighbors, slowdown, linestyle='solid', color=colors[3], hold='on', linewidth=2)
  legendNames.append('set_dynamicObject')
  slowdown = set_staticObject / vector_staticObject
  maxSlowdown = max(maxSlowdown, numpy.amax(slowdown))
  plt.plot(numberOfNeighbors, slowdown, linestyle='solid', color=colors[4], hold='on', linewidth=2)
  legendNames.append('set_staticObject')
  slowdown = vector_listObject / vector_staticObject
  maxSlowdown = max(maxSlowdown, numpy.amax(slowdown))
  plt.plot(numberOfNeighbors, slowdown, linestyle='dashed', color=colors[0], hold='on', linewidth=2)
  legendNames.append('vector_listObject')
  slowdown = vector_vectorOfPointersObject / vector_staticObject
  maxSlowdown = max(maxSlowdown, numpy.amax(slowdown))
  plt.plot(numberOfNeighbors, slowdown, linestyle='dashed', color=colors[1], hold='on', linewidth=2)
  legendNames.append('vector_vectorOfPointersObject')
  slowdown = vector_arrayOfPointersObject / vector_staticObject
  maxSlowdown = max(maxSlowdown, numpy.amax(slowdown))
  plt.plot(numberOfNeighbors, slowdown, linestyle='dashed', color=colors[2], hold='on', linewidth=2)
  legendNames.append('vector_arrayOfPointersObject')
  slowdown = vector_dynamicObject / vector_staticObject
  maxSlowdown = max(maxSlowdown, numpy.amax(slowdown))
  plt.plot(numberOfNeighbors, slowdown, linestyle='dashed', color=colors[3], hold='on', linewidth=2)
  legendNames.append('vector_dynamicObject')
  slowdown = vector_staticObject / vector_staticObject
  maxSlowdown = max(maxSlowdown, numpy.amax(slowdown))
  plt.plot(numberOfNeighbors, slowdown, linestyle='solid', color='k', hold='on', linewidth=2)
  legendNames.append('unity')
  slowdown = set_vectorOfPointersObject_improved / vector_staticObject
  maxSlowdown = max(maxSlowdown, numpy.amax(slowdown))
  plt.plot(numberOfNeighbors, slowdown, linestyle='dashdot', color=colors[1], hold='on', linewidth=2)
  legendNames.append('set_vectorOfPointersObject improved')
  slowdown = vector_staticObject_improved / vector_staticObject
  maxSlowdown = max(maxSlowdown, numpy.amax(slowdown))
  plt.plot(numberOfNeighbors, slowdown, linestyle='dashdot', color=colors[4], hold='on', linewidth=2)
  legendNames.append('vector_staticObject improved')
  box = ax.get_position()
  ax.set_position([box.x0, box.y0, box.width * 0.65, box.height])
  ax.legend(legendNames, loc='upper left', bbox_to_anchor=(1.00, 0.8))
  plt.title('slowdown w.r.t. vector of statically-sized objects')
  plt.xlabel('number Of neighbors [-]', fontsize=16)
  plt.ylabel('slowdown', fontsize=16)
  if (i == 0):
    plt.ylim([1e-2, 1e3])
    plt.yticks([0.01, 0.1, 1, 10, 100, 1000], ['0.01', '0.1', '1.0', '10', '100', '1000'])
  else:
    plt.ylim([1e-2, maxSlowdown])
  plt.grid(b=True, which='major', color='k', linestyle='dotted')
  filename = outputPrefix + '2d_slowdownSummary'
  if (i == 0):
    filename += '_log'
  else:
    filename += '_linear'
  filename += suffix + '.pdf'
  plt.savefig(filename)
  print 'saved file to %s' % filename
