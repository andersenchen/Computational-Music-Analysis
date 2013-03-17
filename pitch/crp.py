#!/usr/bin/python
from __future__ import division

from numpy import *
from matplotlib.pyplot import *
import numpy.random as samples
import scipy.stats as pdfs
import random
import __builtin__
import itertools
import time

from sam.sam import *


def unzip(xys):
    xs = [x for x,y in xys]
    ys = [y for x,y in xys]
    return xs, ys

def ndmean(xs): return __builtin__.sum(xs)/len(xs) 

def coin(p=0.5): return random.random() < p

"""
(generalized?) chinese restaurant process for agnostic monophonic transcription

[add to old table] or [make new table] with probability depending on
a gaussian of the distance with some sample variance

(x,y)
x : vector in frequency domain
y : instrument / frequency class label

eval
how many clusters are made versus how many classes there really are
how homogenous each cluster is
should two clusters obviously be merged into one; or should one obviously be split into two


"""

def P(point, cluster):
    """ P(x|y)
    eg
    point = array([1,1])
    cluster = [ array([0,0]) ]
    loc = array([0,0])
    scale = 1
    
    """

    return product(
        pdfs.norm.pdf(point, loc=ndmean(cluster), scale=1/sqrt(len(cluster))))

def crp(data):
    """
    data 'data' : {(x,y)..}
    each 'x' is a moment in time

    reseat/rerun and average
    """
 
    clusters = []

    clusters.append([data[0]])
    data = data[1:]

    for x in data:
        y,py = max( [(y,P(x,y)) for y in clusters], key=lambda (y,py): py )
        
        if coin(py):
            y.append(x)
        else:
            clusters.append([x])

    return clusters


# # # # # # # # # # # # # # # # # # # # # # # # # # # # 

ion()

# make data from few gaussians
mx,my = 5,10
ax = [samples.normal(loc=mx,scale=0.5) for _ in range(10)]
ay = [samples.normal(loc=my,scale=0.5) for _ in range(10)]

mx,my = 10,5
bx = [samples.normal(loc=mx,scale=1.0) for _ in range(5)]
by = [samples.normal(loc=my,scale=1.0) for _ in range(5)]

mx,my = 10,10
cx = [samples.normal(loc=mx,scale=2.0) for _ in range(5)]
cy = [samples.normal(loc=my,scale=2.0) for _ in range(5)]

figure(); axis((0,20,0,20))
scatter(ax,ay,c='r'); scatter(bx,by,c='b'); scatter(cx,cy,c='y')
draw()

# for crp
data = [array([x,y]) for x,y in zip(ax,ay)+zip(bx,by)+zip(cx,cy)]
random.shuffle(data)

clusters = crp(data)
colors = iter([ 'r', 'g', 'b', 'y', 'c', 'm', 'k', 'w' ])
#  TODO need more colors
figure(); axis((0,20,0,20))
#for cluster,color in zip(clusters,colors):
for cluster in clusters:
    if len(cluster)>1: # only show non-singleton clusters
        xs,ys = unzip(cluster)
        scatter(xs,ys, c=colors.next())
draw()

time.sleep(60)
