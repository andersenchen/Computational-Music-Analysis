from __future__ import division

from numpy import *
from matplotlib.pyplot import *

def about(x,y, eps=0.01): # tolerance within 1%
    if x==0 or y==0: return False
    return abs(x - y) / abs(min(x,y)) < eps

def closer(a, x,y):
    """ returns whichever (x or y) is closer to a (x if tie) """
    return y if abs(y-a) < abs(x-a) else x

def nearer(a, x,y):
    """ returns whether x is closer to a than y is (or as close as) (x if tie) """
    return abs(x-a) < abs(y-a) if abs(x-a) != abs(y-a) else True

def half(x): return x[:len(x)//2]

a=array

def smooth(x, eps=1e-9): return x if abs(x) > eps else 0

# inclusive lazy range
def r(x,y): return range(x, y+1)

import sys
def p(x): sys.stdout.write(str(x) + ' ')

def basename(s):
    s = s.split('/')[-1] # rem dirs
    return s[:s.index('.')] # rem ext

#eg [[1],[[2,[3]],[4]],5] => [1,2,3,4,5]
def flatten(l):
    if type(l) != type([]): return [l]
    if l==[]: return []
    return reduce(lambda x,y:x+y, map(flatten,l))

class D(Exception):
    def __init__(self,*args):
        self.env = args
#        self.all = __dict__

def nones(n): return [None for _ in xrange(n)]

def fst(x): return x[0]
def snd(x): return x[1]

# to nearest integer
#:: Real => Int
def round(n): return int(n+0.5)

def onechannel(audio):
    if len(audio.shape)==1:
        nSamples, = audio.shape
        nChannels = 1
        audio = audio[:int32(nSamples)]
    else:
        nSamples, nChannels = audio.shape
        audio = audio[:int32(nSamples), 0]

    return audio, nSamples, nChannels
