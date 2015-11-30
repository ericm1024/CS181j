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
from multiprocessing import Process

def createMatplotlibPlots(prefix, outputPrefix, processId, numberOfProcesses, flavor):
  fileIndex = processId
  plt.figure('%d' % processId)
  try:
    while (fileIndex < 10000):
      filename = '%s%05d.csv' % (prefix, fileIndex)
      data = numpy.loadtxt(open('%s%05d.csv' % (prefix, fileIndex),'rb'),delimiter=',',skiprows=0)
  
      plt.clf()
      plt.plot(data[:, 0], data[:, 1], color='k', hold='on', linewidth=2)
      plt.xlabel('x')
      plt.ylabel('displacement')
      plt.ylim([-1, 1])
      plt.title('Cuda finite difference wave equation %s' % flavor)
      filename = '%s%05d.jpg' % (outputPrefix, fileIndex)
      plt.savefig(filename)
      print 'saved file to %s' % filename
      fileIndex += numberOfProcesses
  
  except IOError as exception:
    print 'looks like process %d ran out of data on file %3u' % (processId, fileIndex)
    #print exception
    if (fileIndex == 0):
      print 'couldn\'t find file at %s%05d.csv' % (prefix, fileIndex)

if __name__ == '__main__':
  if (len(sys.argv) == 1):
    print 'You need to provide an argument of either \"NonInvasive\" or \"Invasive\"'
    sys.exit(1)
  flavor = sys.argv[1]
  if (flavor != 'NonInvasive' and flavor != 'Invasive'):
    print 'flavor %s is unknown' % flavor
    sys.exit(1)
  prefix = 'data/Cuda1DFDWave' + flavor + '_'
  outputPrefix = 'figures/Cuda1DFDWave' + flavor + '_'
  numberOfProcesses = 8
  
  numberOfFiles = 0
  try:
    while (numberOfFiles < 10000):
      filename = '%s%05d.csv' % (prefix, numberOfFiles)
      f = open('%s%05d.csv' % (prefix, numberOfFiles),'rb')
      numberOfFiles = numberOfFiles + 1
  except IOError as exception:
    print 'found %d files to post-process' % numberOfFiles

  if (numberOfFiles > 0):
    #spawn a pool of processes
    processes = []
    for processId in range(numberOfProcesses):
      p = Process(target = createMatplotlibPlots, args=(prefix, outputPrefix, processId, numberOfProcesses, flavor))
      p.start()
      processes.append(p)
    
    for processId in range(numberOfProcesses):
      processes[processId].join()

    print 'you can now generate a movie with the script, like \"sh MakeMovie.sh %s\"' % (flavor)
  else:
    print 'error: couldn\'t find files to post process for flavor %s' % (flavor)
