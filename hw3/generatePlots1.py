import math
import numpy
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

prefix = 'data/Main1_'
suffix = '_shuffler'
outputPrefix = 'figures/Main1_'

if (os.path.isdir('figures') == False):
  print 'please make figures directory'
  sys.exit(1)

output = numpy.loadtxt(open(prefix + 'data' + suffix + '.csv','rb'),delimiter=',',skiprows=1)

matrixSizes                         = output[:, 0]
numberOfTrials                      = output[:, 1]
colRowL1CacheMisses                 = output[:, 2]
colRowTimes                         = output[:, 3]
rowColL1CacheMisses                 = output[:, 4]
rowColTimes                         = output[:, 5]
improvedRowColL1CacheMisses         = output[:, 6]
improvedRowColTimes                 = output[:, 7]
scalarTileSizes = []
tileSizes = []
tiledMatrixSizes = []
tiledL1CacheMisses = []
tiledTimes = []
column = 8
while (column < len(output[0,:])):
  tileSizes.append(output[:, column])
  scalarTileSizes.append(output[0, column])
  column += 1
  tiledMatrixSizes.append(output[:, column])
  column += 1
  tiledL1CacheMisses.append(output[:, column])
  column += 1
  tiledTimes.append(output[:, column])
  column += 1
numberOfTileSizes = len(tileSizes)
flops = 2 * matrixSizes * matrixSizes * matrixSizes

colors = ['b', 'r', 'g', 'y', 'c', 'm']
peakTheoreticalFlopsRate = 2.6e9

fig = plt.figure('SummaryFlopsRate', figsize=(12,6))
ax = plt.subplot(111)
legendNames = []
plt.xscale('log')
plt.plot(matrixSizes, 100 * flops/colRowTimes/peakTheoreticalFlopsRate, color='k', linestyle='dashed', linewidth=2, hold='on')
legendNames.append('colRow')
plt.plot(matrixSizes, 100 * flops/rowColTimes/peakTheoreticalFlopsRate, color='k', linestyle='solid', linewidth=2, hold='on')
legendNames.append('rowCol')
plt.plot(matrixSizes, 100 * flops/improvedRowColTimes/peakTheoreticalFlopsRate, color='k', linestyle='dashdot', linewidth=2, hold='on')
legendNames.append('improvedRowCol')
for i in range(numberOfTileSizes):
  tiledFlops = 2 * tiledMatrixSizes[i] * tiledMatrixSizes[i] * tiledMatrixSizes[i]
  plt.plot(tiledMatrixSizes[i], 100 * tiledFlops/tiledTimes[i]/peakTheoreticalFlopsRate, color=colors[i], linestyle='solid', linewidth=2, hold='on')
  legendNames.append('tile size %3d' % tileSizes[i][0])
plt.xlabel('size of matrix', fontsize=16)
plt.ylabel('percent of peak flops rate achieved [-]', fontsize=16)
plt.grid(b=True, which='major', color='k', linestyle='dotted')
plt.title('Achieved Flops Rate', fontsize=16)
plt.xlim([matrixSizes[0], matrixSizes[-1]])
plt.ylim([0, 100])
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.65, box.height])
ax.legend(legendNames, loc='upper left', bbox_to_anchor=(1.00, 0.8))
filename = outputPrefix + 'FlopsRate' + suffix + '.pdf'
plt.savefig(filename)
print 'saved file to %s' % filename

fig = plt.figure('Summary_FlopsPerCacheMiss', figsize=(12,6))
ax = plt.subplot(111)
legendNames = []
plt.xscale('log')
plt.yscale('log')
plt.plot(matrixSizes, flops / colRowL1CacheMisses, color='k', linestyle='dashed', linewidth=2, hold='on')
legendNames.append('colRow')
plt.plot(matrixSizes, flops / rowColL1CacheMisses, color='k', linestyle='solid', linewidth=2, hold='on')
legendNames.append('rowCol')
for i in range(numberOfTileSizes):
  tiledFlops = 2 * tiledMatrixSizes[i] * tiledMatrixSizes[i] * tiledMatrixSizes[i]
  plt.plot(tiledMatrixSizes[i], tiledFlops/tiledL1CacheMisses[i], color=colors[i], linestyle='solid', linewidth=2, hold='on')
  legendNames.append('tile size %3d' % tileSizes[i][0])
plt.xlabel('size of matrix', fontsize=16)
plt.ylabel('Flops per L1 Cache Miss', fontsize=16)
plt.grid(b=True, which='major', color='k', linestyle='dotted')
plt.title('Flops per Cache Miss', fontsize=16)
plt.xlim([matrixSizes[0], matrixSizes[-1]])
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.65, box.height])
ax.legend(legendNames, loc='upper left', bbox_to_anchor=(1.00, 0.8))
filename = outputPrefix + 'FlopsPerCacheMiss' + suffix + '.pdf'
plt.savefig(filename)
print 'saved file to %s' % filename

