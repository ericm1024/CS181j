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

prefix = 'data/Main2_'
suffix = '_shuffler'
outputPrefix = 'figures/Main2_'

sqrtData = numpy.loadtxt(open(prefix + 'sqrt_results' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
fixedPolynomialData = numpy.loadtxt(open(prefix + 'fixedPolynomial_results' + suffix + '.csv','rb'),delimiter=',',skiprows=0)

manualColor      = 'b'
compilerColor    = 'r'

makeImageFiles = True

plt.figure(figsize=(9, 6))
ax = plt.subplot(111)
plt.xscale('log')
legendNames = []
index = 2
plt.plot(sqrtData[:,0], sqrtData[:,1] / sqrtData[:,index], color=manualColor, hold='on', linewidth=2)
legendNames.append('manual')
index = index + 1
plt.plot(sqrtData[:,0], sqrtData[:,1] / sqrtData[:,index], color=compilerColor, hold='on', linewidth=2)
legendNames.append('compiler')
index = index + 1
plt.title('speedup of sqrt', fontsize=16)
plt.xlabel('number of intervals', fontsize=16)
plt.ylabel('speedup', fontsize=16)
plt.ylim([0, 10])
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.80, box.height])
ax.legend(legendNames, loc='center right', bbox_to_anchor=(1.30, 0.5))
if (makeImageFiles == True):
  filename = outputPrefix + 'sqrt' + suffix + '.pdf'
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
plt.xlabel('number of intervals', fontsize=16)
plt.ylabel('speedup', fontsize=16)
plt.ylim([0, 10])
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.80, box.height])
ax.legend(legendNames, loc='center right', bbox_to_anchor=(1.30, 0.5))
if (makeImageFiles == True):
  filename = outputPrefix + 'fixedPolynomial' + suffix + '.pdf'
  plt.savefig(filename)
  print 'saved file to %s' % filename
else:
  plt.show()
