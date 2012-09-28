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
# x = notes
# b = audio (known from input)
# A = instrument/note overtone/undertone profile (known from training)
def pitch_pinv():
    Ai = pinv(classifier)

    x = zeros((n_windows, classifier.shape[1]))

    for i in xrange(n_windows):
        # normalize frequency to unit vector
        b = spectrum[i,:] / sum(spectrum[i,:])
        # x=A\b
        print 'us', spectrum.shape
        print 'nW', n_windows
        print 'A', Ai.shape
        print 'b', b.shape
        print 'x', x.shape
        exit()
        x[i,:] = dot( Ai, b )

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

pitch('pinv')
