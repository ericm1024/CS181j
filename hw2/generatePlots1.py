#!/usr/bin/env python

import math
import numpy
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

prefix = 'data/Main1_'
suffix = ''
outputPrefix = 'figures/Main1_'

if (os.path.isdir('figures') == False):
  print 'please make figures directory'
  sys.exit(1)

output = numpy.loadtxt(open(prefix + 'data' + suffix + '.csv','rb'),delimiter=',',skiprows=1)

node_sizes = output[:,0]
insert_cache_misses = output[:,1]
insert_time = output[:,2]
iterate_cache_misses = output[:,3]
iterate_time = output[:,4]
erase_cache_misses = output[:,5]
erase_time = output[:,6]


fig = plt.figure('Runtime vs Node Size', figsize=(9,6))
legendNames = []
plt.xscale('log')
plt.plot(node_sizes, insert_time, color='b', linestyle='dashed', linewidth=3, hold='on')
legendNames.append('insert')
plt.plot(node_sizes, iterate_time, color='b', linestyle='solid', linewidth=3, hold='on')
legendNames.append('iterate')
plt.plot(node_sizes, erase_time, color='b', linestyle='dashdot', linewidth=3, hold='on')
legendNames.append('erase')
plt.xlabel('size of node', fontsize=16)
plt.ylabel('runtime of 1 million operations', fontsize=16)
plt.grid(b=True, which='major', color='k', linestyle='dotted')
plt.legend(legendNames, loc='lower left')
plt.title('Runtime vs Node Size', fontsize=16)
plt.xlim([node_sizes[0], node_sizes[-1]])
filename = outputPrefix + 'Runtime' + suffix + '.pdf'
plt.savefig(filename)
print 'saved file to %s' % filename

fig = plt.figure('Runtime vs Node Size (Log)', figsize=(9,6))
legendNames = []
plt.xscale('log')
plt.yscale('log')
plt.plot(node_sizes, insert_time, color='b', linestyle='dashed', linewidth=3, hold='on')
legendNames.append('insert')
plt.plot(node_sizes, iterate_time, color='b', linestyle='solid', linewidth=3, hold='on')
legendNames.append('iterate')
plt.plot(node_sizes, erase_time, color='b', linestyle='dashdot', linewidth=3, hold='on')
legendNames.append('erase')
plt.xlabel('size of node', fontsize=16)
plt.ylabel('runtime of 1 million operations', fontsize=16)
plt.grid(b=True, which='major', color='k', linestyle='dotted')
plt.legend(legendNames, loc='lower left')
plt.title('Runtime vs Node Size (Log)', fontsize=16)
plt.xlim([node_sizes[0], node_sizes[-1]])
filename = outputPrefix + 'RuntimeLog' + suffix + '.pdf'
plt.savefig(filename)
print 'saved file to %s' % filename





fig = plt.figure('Cache Misses vs Node Size', figsize=(9,6))
legendNames = []
plt.xscale('log')
plt.plot(node_sizes, insert_cache_misses, color='b', linestyle='dashed', linewidth=3, hold='on')
legendNames.append('insert')
plt.plot(node_sizes, iterate_cache_misses, color='b', linestyle='solid', linewidth=3, hold='on')
legendNames.append('iterate')
plt.plot(node_sizes, erase_cache_misses, color='b', linestyle='dashdot', linewidth=3, hold='on')
legendNames.append('erase')
plt.xlabel('size of node', fontsize=16)
plt.ylabel('cache misses for 1 million operations', fontsize=16)
plt.grid(b=True, which='major', color='k', linestyle='dotted')
plt.legend(legendNames, loc='lower left')
plt.title('Cache Misses vs Node Size', fontsize=16)
plt.xlim([node_sizes[0], node_sizes[-1]])
filename = outputPrefix + 'CacheMisses' + suffix + '.pdf'
plt.savefig(filename)
print 'saved file to %s' % filename

fig = plt.figure('Cache Misses vs Node Size (Log)', figsize=(9,6))
legendNames = []
plt.xscale('log')
plt.yscale('log')
plt.plot(node_sizes, insert_cache_misses, color='b', linestyle='dashed', linewidth=3, hold='on')
legendNames.append('insert')
plt.plot(node_sizes, iterate_cache_misses, color='b', linestyle='solid', linewidth=3, hold='on')
legendNames.append('iterate')
plt.plot(node_sizes, erase_cache_misses, color='b', linestyle='dashdot', linewidth=3, hold='on')
legendNames.append('erase')
plt.xlabel('size of node', fontsize=16)
plt.ylabel('cache misses for 1 million operations', fontsize=16)
plt.grid(b=True, which='major', color='k', linestyle='dotted')
plt.legend(legendNames, loc='lower left')
plt.title('Cache Misses vs Node Size (Log)', fontsize=16)
plt.xlim([node_sizes[0], node_sizes[-1]])
filename = outputPrefix + 'CacheMissesLog' + suffix + '.pdf'
plt.savefig(filename)
print 'saved file to %s' % filename
