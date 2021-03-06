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

def createPilPlots(prefix, outputPrefix, processId, numberOfRenderingProcesses, numberOfSimulationProcesses, debugRank):
  fileIndex = processId
  try:
    while (fileIndex < 10000):
      imageWidth = 1920
      imageHeight = 1080
      black = (0, 0, 0)
      backgroundColor = (255, 255, 255)
      transferColor = (255, 20, 147) # bright pink!
      shadowColor = (150, 150, 150)
      pilImage = Image.new("RGB", (imageWidth, imageHeight), backgroundColor)
      pilDraw = ImageDraw.Draw(pilImage)
      # draw the domain boundaries
      numberOfProcessesPerSide = int(math.sqrt(numberOfSimulationProcesses)+1e-6)
      for i in range(numberOfProcessesPerSide- 1):
        xPixel = (i+1) * imageWidth / numberOfProcessesPerSide
        pilDraw.line((xPixel, 0, xPixel, imageHeight), fill=128)
        yPixel = (i+1) * imageHeight / numberOfProcessesPerSide
        pilDraw.line((0, yPixel, imageWidth, yPixel), fill=128)
      pointRadius = 10
      colors     = cm.jet(numpy.linspace(0, 1, numberOfSimulationProcesses))
      filename = '%s%05d_%03d.csv' % (prefix, fileIndex, debugRank)
      try:
        numberOfLinesInThisFile = sum(1 for line in open(filename))
        if (numberOfLinesInThisFile > 2):
          data = numpy.loadtxt(open(filename,'rb'),delimiter=',',skiprows=0)
          simulationXExtrema = [data[0, 0], data[0, 1]]
          simulationYExtrema = [data[1, 0], data[1, 1]]
          pointColor = (int(colors[debugRank][0] * 255), int(colors[debugRank][1] * 255), int(colors[debugRank][2] * 255))
          for pointIndex in range(len(data[:,0])-2):
            x = data[pointIndex+2, 0]
            y = data[pointIndex+2, 1]
            transfer = data[pointIndex + 2, 2]
            xPixel = ((x - simulationXExtrema[0]) / (simulationXExtrema[1] - simulationXExtrema[0])) * imageWidth - pointRadius/2
            yPixel = imageHeight - (((y - simulationYExtrema[0]) / (simulationYExtrema[1] - simulationYExtrema[0])) * imageHeight) - pointRadius/2
            if (transfer == 1):
              color = transferColor
            elif (transfer == 2):
              color = shadowColor
            else:
              color = pointColor
            pilDraw.ellipse((xPixel, yPixel, xPixel + pointRadius, yPixel + pointRadius), fill=color, outline=black)
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
  if len(sys.argv) != 3:
    print 'you need to specify a number of processes with which you ran the code and which rank you want to draw the output from, like \"python generateMini2dMDPlots.py 12 4\" if you ran it with 12 processes and want to see rank 4\'s output'
    sys.exit(1)
  numberOfSimulationProcesses = int(sys.argv[1])
  debugRank = int(sys.argv[2])
  prefix = 'data/Mini2dMD_NeedToKnow_Debug_%03d_' % (numberOfSimulationProcesses)
  outputPrefix = 'figures/Mini2dMD_NeedToKnow_Debug_%03d_%03d_' % (numberOfSimulationProcesses, debugRank)

  numberOfRenderingProcesses = 8

  #spawn a pool of processes
  processes = []
  for processId in range(numberOfRenderingProcesses):
    p = Process(target = createPilPlots, args=(prefix, outputPrefix, processId, numberOfRenderingProcesses, numberOfSimulationProcesses, debugRank))
    p.start()
    processes.append(p)

  for processId in range(numberOfRenderingProcesses):
    processes[processId].join()
