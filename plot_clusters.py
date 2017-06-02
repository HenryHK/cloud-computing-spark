#!/usr/bin/python
# adjusted from plotPoints.py from Apache Flink project
import sys
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import os

if len(sys.argv) < 4 or not sys.argv[1] in ['points', 'points2d', 'centroids']:
  print "Usage: plot_clusters.py (points|points2d|centroids) <src-file> <pdf-file-prefix> [scale X]"
  print "  points   : create 3D diagram of all points in a cluster"
  print "  points2d : create 2D plot of all points in a cluster"
  print "  centroids: create 3D diagram of just cluster centroids with size relative to cluster size"
  print "<src-file> : filename of file in HDFS with output format as specified in A2"
  print "<pdf-file-prefix>: prefix of output PDF file"
  print "  [scale X]: optional paramtere 'scale' with value X; for 'centroids' plot: scales down centroid size; for 'points' plots: only every X-th point plotted; default 25"
  print ""
  print "Use 'centroids' for plotting the standard output of assignment 2 (the cluster centroids and their weights from Task 2)."
  print "For 'points' and 'points2d' plots, this script assumes a slightly different src file that contains _all_ points of the cluster, each point prefixed with the cluster ID and cluster size of 1. This is similar to a debug output of an intermediate step in the k-means clustering of Task3."
  sys.exit(1)

inFile = sys.argv[1]
inFile = sys.argv[2]
outFilePx = sys.argv[3]
if len(sys.argv) == 6 and sys.argv[4]=='size':
   scale = int(sys.argv[5])
else:
   scale = 25
inFileName = os.path.splitext(os.path.basename(inFile))[0]
outFile = os.path.join(".", outFilePx+"-plot.pdf")

if len(sys.argv) == 6 and sys.argv[4]=='scale':
   scale = int(sys.argv[5])
else:
   scale = 25

# feel free to adjust if own axis meanings differ
axistitle = {'x': 'ly6c', 'y': 'cd11b', 'z': 'sca1' }

fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
# change this the following line to modify the angle of a 3D plot
#ax.view_init(30,30);

########### READ DATA

cs = []
xs = []
ys = []
zs = []
sz = []

minX = None
maxX = None
minY = None
maxY = None
minZ = None
maxZ = None

if sys.argv[1] == 'points2d':

  i = 1
  with open(inFile, 'rb') as file:
    for line in file:
      # parse data
      csvData = line.strip().split("\t")

      c = int(csvData[0])
      s = int(csvData[1])
      x = float(csvData[2])
      y = float(csvData[3])

      if not minX or minX > x:
        minX = x
      if not maxX or maxX < x:
        maxX = x
      if not minY or minY > y:
        minY = y
      if not maxY or maxY < y:
        maxY = y

      if (i % scale) == 0:
        cs.append(c)
        xs.append(x)
        ys.append(y)
      i += 1

    # plot data
    plt.clf()
    plt.scatter(xs, ys, s=1, c=cs, edgecolors='None', alpha=1.0)
    plt.ylim([min(0,minY),max(maxY,2)])
    plt.xlim([min(0,minX),max(maxX,2)])
    fig = plt

elif sys.argv[1] == 'points':

  i = 0
  with open(inFile, 'rb') as file:
    for line in file:
      # parse data
      csvData = line.strip().split("\t")

      c = int(csvData[0])
      s = int(csvData[1])
      x = float(csvData[2])
      y = float(csvData[3])
      z = float(csvData[4])

      if not minX or minX > x:
        minX = x
      if not maxX or maxX < x:
        maxX = x
      if not minY or minY > y:
        minY = y
      if not maxY or maxY < y:
        maxY = y
      if not minZ or minZ > z:
        minZ = z
      if not maxZ or maxZ < z:
        maxZ = z

      if i % scale == 0:
         cs.append(c)
         xs.append(x)
         ys.append(y)
         zs.append(z)
      i += 1

    # plot data
    ax.scatter(xs, ys, zs, s=1, c=cs, edgecolors='None', alpha=1.0)
    ax.set_xlabel(axistitle['x'])
    ax.set_ylabel(axistitle['y'])
    ax.set_zlabel(axistitle['z'])

elif sys.argv[1] == 'centroids':

  with open(inFile, 'rb') as file:
    for line in file:
      # parse data
      csvData = line.strip().split('\t')

      c = int(csvData[0])
      s = int(csvData[1])
      x = float(csvData[2])
      y = float(csvData[3])
      z = float(csvData[4])

      cs.append(c)
      sz.append(s/scale)
      xs.append(x)
      ys.append(y)
      zs.append(z)

    # plot data
    ax.scatter(xs, ys, zs, s=sz, c=cs, edgecolors='None', alpha=1.0)
    ax.set_xlabel(axistitle['x'])
    ax.set_ylabel(axistitle['y'])
    ax.set_zlabel(axistitle['z'])


fig.savefig(outFile, dpi=600)
print "\nPlotted file: %s" % outFile

sys.exit(0)
