import math
import numpy
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

prefix = 'data/Main2_'
suffix = '_shuffler'
outputPrefix = 'figures/Main2_'

if (os.path.isdir('figures') == False):
  print 'please make figures directory'
  sys.exit(1)

output = numpy.loadtxt(open(prefix + 'data' + suffix + '.csv','rb'),delimiter=',',skiprows=1)

matrixSizes                             = output[:,0]
repeats                                 = output[:,1]
flops                                   = output[:,2]
colRowL1CacheMisses                     = output[:,3]
colRowTimes                             = output[:,4]
rowColL1CacheMisses                     = output[:,5]
rowColTimes                             = output[:,6]
improvedRowColL1CacheMisses             = output[:,7]
improvedRowColTimes                     = output[:,8]

rowColMaxFlopsRate = numpy.amax(flops / rowColTimes)
improvedRowColMaxFlopsRate = numpy.amax(flops / improvedRowColTimes)
peakTheoreticalFlopsRate = 2.6e9
print 'peak flops rates are %8.2e (%%%5.1f) for rowCol and %8.2e (%%%5.1f) for improved' % ( rowColMaxFlopsRate, 100 * rowColMaxFlopsRate / peakTheoreticalFlopsRate, improvedRowColMaxFlopsRate, 100 * improvedRowColMaxFlopsRate / peakTheoreticalFlopsRate)

fig = plt.figure('FlopsRate', figsize=(9,6))
legendNames = []
plt.xscale('log')
plt.plot(matrixSizes, 100 * flops/colRowTimes/peakTheoreticalFlopsRate, color='b', linestyle='dashed', linewidth=3, hold='on')
legendNames.append('Cache-unfriendly')
plt.plot(matrixSizes, 100 * flops/rowColTimes/peakTheoreticalFlopsRate, color='b', linestyle='solid', linewidth=3, hold='on')
legendNames.append('Cache-friendly')
plt.plot(matrixSizes, 100 * flops/improvedRowColTimes/peakTheoreticalFlopsRate, color='b', linestyle='dashdot', linewidth=3, hold='on')
legendNames.append('Improved Cache-friendly')
plt.xlabel('size of matrix', fontsize=16)
plt.ylabel('percent of peak flops rate achieved', fontsize=16)
plt.grid(b=True, which='major', color='k', linestyle='dotted')
plt.legend(legendNames, loc='lower left')
plt.title('Achieved Flops Rate', fontsize=16)
plt.xlim([matrixSizes[0], matrixSizes[-1]])
filename = outputPrefix + 'FlopsRate' + suffix + '.pdf'
plt.savefig(filename)
print 'saved file to %s' % filename

fig = plt.figure('CacheMisses', figsize=(9,6))
legendNames = []
plt.xscale('log')
plt.yscale('log')
plt.plot(matrixSizes, colRowL1CacheMisses, color='b', linestyle='dashed', linewidth=3, hold='on')
legendNames.append('Cache-unfriendly')
plt.plot(matrixSizes, rowColL1CacheMisses, color='b', linestyle='solid', linewidth=3, hold='on')
legendNames.append('Cache-friendly')
plt.plot(matrixSizes, improvedRowColL1CacheMisses, color='b', linestyle='dashdot', linewidth=3, hold='on')
legendNames.append('Improved Cache-friendly')
plt.xlabel('size of matrix', fontsize=16)
plt.ylabel('Number of L1 Cache Misses', fontsize=16)
plt.grid(b=True, which='major', color='k', linestyle='dotted')
plt.legend(legendNames, loc='upper left')
plt.title('Cache Misses', fontsize=16)
plt.xlim([matrixSizes[0], matrixSizes[-1]])
filename = outputPrefix + 'CacheMisses' + suffix + '.pdf'
plt.savefig(filename)
print 'saved file to %s' % filename

fig = plt.figure('FlopsPerCacheMiss', figsize=(9,6))
legendNames = []
plt.xscale('log')
plt.yscale('log')
plt.plot(matrixSizes, flops/colRowL1CacheMisses, color='b', linestyle='dashed', linewidth=3, hold='on')
legendNames.append('Cache-unfriendly')
plt.plot(matrixSizes, flops/rowColL1CacheMisses, color='b', linestyle='solid', linewidth=3, hold='on')
legendNames.append('Cache-friendly')
plt.plot(matrixSizes, flops/improvedRowColL1CacheMisses, color='b', linestyle='dashdot', linewidth=3, hold='on')
legendNames.append('Improved Cache-friendly')
plt.xlabel('size of matrix', fontsize=16)
plt.ylabel('Number of Flops per Cache Miss', fontsize=16)
plt.grid(b=True, which='major', color='k', linestyle='dotted')
plt.legend(legendNames, loc='lower left')
plt.title('Flops per Cache Miss', fontsize=16)
plt.xlim([matrixSizes[0], matrixSizes[-1]])
filename = outputPrefix + 'FlopsPerCacheMiss' + suffix + '.pdf'
plt.savefig(filename)
print 'saved file to %s' % filename

