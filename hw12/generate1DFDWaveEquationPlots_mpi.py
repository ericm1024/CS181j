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

def createMatplotlibPlots(prefix, outputPrefix, processId, numberOfRenderingProcesses, numberOfSimulationProcesses):
  fileIndex = processId
  plt.figure('%d' % processId)
  try:
    while (fileIndex < 10000):
      plt.clf()

      colors = cm.gnuplot(numpy.linspace(0, 1, numberOfSimulationProcesses))

      for rank in range(numberOfSimulationProcesses):
        data = numpy.loadtxt(open('%s%05d_%03d.csv' % (prefix, fileIndex, rank),'rb'),delimiter=',',skiprows=0)
        plt.plot(data[:, 0], data[:, 1], color=colors[rank], hold='on', linewidth=2)

      plt.xlabel('x')
      plt.ylabel('displacement')
      plt.ylim([-1, 1])
      plt.xlim([0, 1])
      plt.title('Parallel finite difference wave equation, %3u processes' % numberOfSimulationProcesses)
      filename = '%s%05d.jpg' % (outputPrefix, fileIndex)
      plt.savefig(filename)
      print 'saved file to %s' % filename
      fileIndex += numberOfRenderingProcesses
  
  except IOError as exception:
    print 'looks like process %d ran out of data on file %3u' % (processId, fileIndex)
    #print exception
    if (fileIndex == 0):
      print 'couldn\'t find file at %s%05d.csv' % (prefix, fileIndex)

if __name__ == '__main__':
  if len(sys.argv) != 2:
    print 'you need to specify a number of processes with which you ran the code, like \"python generate1DFDWaveEquationPlots.py X\"'
    sys.exit(1)
  numberOfSimulationProcesses = int(sys.argv[1])
  prefix = 'data/Parallel1DFDWave_%03d_' % numberOfSimulationProcesses
  outputPrefix = 'figures/Parallel1DFDWave_%03d_' % numberOfSimulationProcesses

  numberOfRenderingProcesses = 8

  #spawn a pool of processes
  processes = []
  for processId in range(numberOfRenderingProcesses):
    p = Process(target = createMatplotlibPlots, args=(prefix, outputPrefix, processId, numberOfRenderingProcesses, numberOfSimulationProcesses))
    p.start()
    processes.append(p)

  for processId in range(numberOfRenderingProcesses):
    processes[processId].join()
