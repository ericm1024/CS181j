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
suffix = '_shuffler'
outputPrefix = 'figures/Main1_'

sdotData = numpy.loadtxt(open(prefix + 'sdot_results' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
offsetsData = numpy.loadtxt(open(prefix + 'offsets_results' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
fixedPolynomialData = numpy.loadtxt(open(prefix + 'fixedPolynomial_results' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
expData = numpy.loadtxt(open(prefix + 'exp_results' + suffix + '.csv','rb'),delimiter=',',skiprows=0)

manualColor      = 'b'
compilerColor    = 'r'

makeImageFiles = True

plt.figure(figsize=(9, 6))
ax = plt.subplot(111)
plt.xscale('log')
legendNames = []
index = 2
plt.plot(sdotData[:,0], sdotData[:,1] / sdotData[:,index], color=manualColor, hold='on', linewidth=2)
legendNames.append('manual')
index = index + 1
plt.plot(sdotData[:,0], sdotData[:,1] / sdotData[:,index], '--', color=manualColor, hold='on', linewidth=2)
legendNames.append('sse prefetch')
index = index + 1
plt.plot(sdotData[:,0], sdotData[:,1] / sdotData[:,index], '-.', color=manualColor, hold='on', linewidth=2)
legendNames.append('sse dot product')
index = index + 1
plt.plot(sdotData[:,0], sdotData[:,1] / sdotData[:,index], color=compilerColor, hold='on', linewidth=2)
legendNames.append('compiler')
index = index + 1
plt.title('speedup of sdot', fontsize=16)
plt.xlabel('vector size', fontsize=16)
plt.ylabel('speedup', fontsize=16)
plt.ylim([0, 12])
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.70, box.height])
ax.legend(legendNames, loc='upper left', bbox_to_anchor=(1.00, 0.80))
if (makeImageFiles == True):
  filename = outputPrefix + 'sdot' + suffix + '.pdf'
  plt.savefig(filename)
  print 'saved file to %s' % filename
else:
  plt.show()

plt.figure(figsize=(9, 6))
ax = plt.subplot(111)
plt.xscale('log')
legendNames = []
index = 2
plt.plot(fixedPolynomialData[:,0], fixedPolynomialData[:,1] / fixedPolynomialData[:,index], color=manualColor, hold='on', linewidth=2)
legendNames.append('manual')
index = index + 1
plt.plot(fixedPolynomialData[:,0], fixedPolynomialData[:,1] / fixedPolynomialData[:,index], color=compilerColor, hold='on', linewidth=2)
legendNames.append('compiler')
index = index + 1
plt.title('speedup of fixedPolynomial', fontsize=16)
plt.xlabel('vector size', fontsize=16)
plt.ylabel('speedup', fontsize=16)
plt.ylim([0, 12])
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.70, box.height])
ax.legend(legendNames, loc='upper left', bbox_to_anchor=(1.00, 0.80))
if (makeImageFiles == True):
  filename = outputPrefix + 'fixedPolynomial' + suffix + '.pdf'
  plt.savefig(filename)
  print 'saved file to %s' % filename
else:
  plt.show()

plt.figure(figsize=(9, 6))
ax = plt.subplot(111)
plt.xscale('log')
legendNames = []
index = 3
plt.plot(offsetsData[:,0], offsetsData[:,2] / offsetsData[:,index], color=manualColor, hold='on', linewidth=2)
legendNames.append('manual')
index = index + 1
plt.plot(offsetsData[:,0], offsetsData[:,2] / offsetsData[:,index], color=compilerColor, hold='on', linewidth=2)
legendNames.append('compiler')
index = index + 1
plt.title('speedup of offsets', fontsize=16)
plt.xlabel('vector size', fontsize=16)
plt.ylabel('speedup', fontsize=16)
plt.ylim([0, 12])
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.70, box.height])
ax.legend(legendNames, loc='upper left', bbox_to_anchor=(1.00, 0.80))
if (makeImageFiles == True):
  filename = outputPrefix + 'offsets' + suffix + '.pdf'
  plt.savefig(filename)
  print 'saved file to %s' % filename
else:
  plt.show()

plt.figure(figsize=(9, 6))
ax = plt.subplot(111)
plt.xscale('log')
legendNames = []
index = 2
plt.plot(expData[:,0], expData[:,1] / expData[:,index], color=manualColor, hold='on', linewidth=2)
legendNames.append('manual')
index = index + 1
plt.plot(expData[:,0], expData[:,1] / expData[:,index], color=compilerColor, hold='on', linewidth=2)
legendNames.append('compiler')
index = index + 1
plt.title('speedup of exp', fontsize=16)
plt.xlabel('vector size', fontsize=16)
plt.ylabel('speedup', fontsize=16)
plt.ylim([0, 12])
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.70, box.height])
ax.legend(legendNames, loc='upper left', bbox_to_anchor=(1.00, 0.80))
if (makeImageFiles == True):
  filename = outputPrefix + 'taylorExponential' + suffix + '.pdf'
  plt.savefig(filename)
  print 'saved file to %s' % filename
else:
  plt.show()
