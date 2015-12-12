# random circles in Tkinter
# a left mouse double click will idle action for 5 seconds and
# save the canvas drawing to an image file
# the Tkinter canvas can only be saved in postscript format 
# run PIL imagedraw simultaneously which
# draws in memory, but can be saved in many formats
# modified vegaseat's code from (circles):
# http://www.daniweb.com/software-development/python/code/216626
# and (simultaneous PIL imagedraw):
# http://www.daniweb.com/software-development/python/code/216929
from PIL import Image, ImageDraw
import sys
import numpy
import math
import matplotlib
import matplotlib.cm as cm
from multiprocessing import Process

def createPilPlots(prefix, outputPrefix, processId, numberOfRenderingProcesses, numberOfSimulationProcesses):
  fileIndex = processId
  try:
    while (fileIndex < 10000):
      imageWidth = 1920
      imageHeight = 1080
      black = (0, 0, 0)
      backgroundColor = (255, 255, 255)
      pointColor = (150, 150, 150)
      pilImage = Image.new("RGB", (imageWidth, imageHeight), backgroundColor)
      pilDraw = ImageDraw.Draw(pilImage)
      pointRadius = 10
      colors     = cm.jet(numpy.linspace(0, 1, numberOfSimulationProcesses))
      filename = '%s%05d.csv' % (prefix, fileIndex)
      try:
        numberOfLinesInThisFile = sum(1 for line in open(filename))
        if (numberOfLinesInThisFile > 2):
          data = numpy.loadtxt(open(filename,'rb'),delimiter=',',skiprows=0)
          simulationXExtrema = [data[0, 0], data[0, 1]]
          simulationYExtrema = [data[1, 0], data[1, 1]]
          for pointIndex in range(len(data[:,0])-2):
            x = data[pointIndex+2, 0]
            y = data[pointIndex+2, 1]
            xPixel = ((x - simulationXExtrema[0]) / (simulationXExtrema[1] - simulationXExtrema[0])) * imageWidth - pointRadius/2
            yPixel = imageHeight - (((y - simulationYExtrema[0]) / (simulationYExtrema[1] - simulationYExtrema[0])) * imageHeight) - pointRadius/2
            pilDraw.ellipse((xPixel, yPixel, xPixel + pointRadius, yPixel + pointRadius), fill=pointColor, outline=black)
      except IndexError as exception:
        print 'error readying file %s' % filename
        print exception
        raise
      filename = outputPrefix + '%05d.jpg' % fileIndex
      pilImage.save(filename)
      print 'saved file to %s' % filename
      fileIndex += numberOfRenderingProcesses
  
  except IOError as exception:
    print 'looks like process %d ran out of data on file %3u' % (processId, fileIndex)
    #print exception
    if (fileIndex == 0):
      print 'couldn\'t find file at %s%05d.csv' % (prefix, fileIndex)

if __name__ == '__main__':
  if len(sys.argv) != 2:
    print 'you need to specify a number of processes with which you ran the code, like \"python generateMini2dMDPlots.py X\"'
    sys.exit(1)
  numberOfSimulationProcesses = int(sys.argv[1])
  prefix = 'data/Mini2dMD_BigPicture_%03d_' % (numberOfSimulationProcesses)
  outputPrefix = 'figures/Mini2dMD_BigPicture_%03d_' % (numberOfSimulationProcesses)

  numberOfRenderingProcesses = 8

  #spawn a pool of processes
  processes = []
  for processId in range(numberOfRenderingProcesses):
    p = Process(target = createPilPlots, args=(prefix, outputPrefix, processId, numberOfRenderingProcesses, numberOfSimulationProcesses))
    p.start()
    processes.append(p)

  for processId in range(numberOfRenderingProcesses):
    processes[processId].join()
