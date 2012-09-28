#!/usr/bin/python

# matrix multiplication
#  numpy 'dot(A,B)'  is  matlab 'A*B'
# elemwise multiplication
#  numpy 'A*B'  is  matlab 'A.*B'

from __future__ import division

from sam import *
from pitch import *
#from train_joint import train_joint

from numpy import *
from numpy.fft import fft
from numpy.linalg import pinv
from matplotlib.pyplot import *
import argparse


# train the classifier
classifier, tclass, freqs = train_joint()
#  classifier :: window_size by bit_precision

file = 'chord.wav'
spectrum, _ = process_wav(file)
n_windows = spectrum.shape[0]

# pseudoinverse solution
# solve Ax=b for x given A,b
# x : (d,1) = notes
# b : (d',1) = audio (known from input)
# A : (d',d) = instrument/note overtone/undertone profile (known from training)
# d = dimensionality of hidden var
# d' = dimensionality of observed var
# eg d = 8 = |keys in 1 piano 8ve| . d' = 4096 = |linear bins of audible pitches|
def pitch_pinv():
    Ai = pinv(classifier)

    x = zeros((n_windows, classifier.shape[0]))

    print 'd', classifier.shape[0]
    print 'sr', classifier.shape[1]
    print 'nW', n_windows
    print 'us', spectrum.shape
    print 'A', Ai.shape
    print 'b', spectrum[0,:].shape
    print 'x', x.shape
    
    for i in xrange(n_windows-1):
        # normalize frequency to unit vector
        b = spectrum[i,:] / sum(spectrum[i,:])
        # x=A\b
        x[i,:] = dot( Ai.transpose(), b )

    return x

# nmf solution (ie with multiplicative update)
def pitch_nmf():
    pass

# gradient descent solution (ie with additive update)
def pitch_gd():
    pass

def pitch(how='nmf'):
    try:
        return {'nmf' : pitch_nmf(),
                'pinv' : pitch_pinv(),
                'gd' : pitch_gd(),
                }[how]
    except KeyError:
        raise ValueError('\n'.join(["pitch()...", " wants how={'nmf'|'pinv'|'gd'}", " got how=%s" % how]))

# main
x = pitch('pinv')
imshow(x.transpose(), origin='lower', aspect=344/16)
show()

# plot
"""
from mpl_toolkits.mplot3d import Axes3D
fig = figure()
ax = fig.gca(projection='3d')
times, freqs = x.shape
X = a(r(1,times))
Y = a(r(1,freqs))
X, Y = meshgrid(X, Y)
Z = x
print X.shape
print Y.shape
print Z.shape
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,
        linewidth=0, antialiased=False)
show()
"""
