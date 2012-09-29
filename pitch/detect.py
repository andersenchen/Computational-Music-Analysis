#!/usr/bin/python

# matrix multiplication
#  numpy 'dot(A,B)'  is  matlab 'A*B'
# elemwise multiplication
#  numpy 'A*B'  is  matlab 'A.*B'

from __future__ import division

from compmusic.sambo.sam import *
from pitch import *
#from train_joint import train_joint

from numpy import *
from numpy.fft import fft
from numpy.linalg import pinv
from matplotlib.pyplot import *
import argparse

from sam import *


# enum
pitch_detection_algorithms = ['nmf','pinv', 'gd']
def pitch_detection_algorithm(s):
    if s in pitch_detection_algorithms: return s
    else: raise ValueError()

# cli
p=argparse.ArgumentParser(description="Pitch Detection")
p.add_argument('how', type=pitch_detection_algorithm, nargs='?', default='nmf',
               help='nmf | pinv | gd')
args = p.parse_args()
how = args.how

# train classifier
classifier, freqs = train_joint()
#  classifier :: |notes| by window_size

# read file
file = 'chord.wav'
spectrum, sample_rate = process_wav(file)

# consts
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
    x  = zeros((n_windows, classifier.shape[0]))

    print
    print 'PINV...'
    print 'd', classifier.shape[0]
    print 'sr', classifier.shape[1]
    print 'nW', n_windows
    print 'us', spectrum.shape
    print 'A', Ai.shape
    print 'b', spectrum[0,:].shape
    print 'x', x.shape
    
    for i in xrange(n_windows-1):
        # nor frequency to unit vector
        b = spectrum[i,:] / sum(spectrum[i,:])
        # x=A\b
        x[i,:] = dot( t(Ai), b )
        
    return x


# nmf solution (ie with multiplicative update)
# solve Ax=b for x
#  where x : nonnegative
def pitch_nmf():
    d = classifier.shape[0]
    A = classifier
    x = 0.5 * ones((n_windows, d))
    
    print 'NMF...'
    print 'd', d
    print 'sr', classifier.shape[1]
    print 'nW', n_windows
    print 'us', spectrum.shape
    print 'A', A.shape
    print 'b', spectrum[0,:].shape
    print 'x', x.shape
    
    for i in xrange(n_windows-1):
        # normalize to unit vector
        b = spectrum[i,:] / sum(spectrum[i,:])
        
        # multiplicative update with euclidean distance
        for k in xrange(10): # until convergence
            numer = dot( A, b ) #: (d,)
            denom = mul( A, t(A), x[i,:] ) #: (d,sr) * (sr,d) * (d,) = (d,)
            x[i,:] = x[i,:] * numer / denom #:(d,)

    return x
            

# gradient descent solution (ie with additive update)
def pitch_gd():
    

    print
    print 'GD...'

def pitch(how='nmf'):
    try:
        return {'nmf'  : pitch_nmf,
                'pinv' : pitch_pinv,
                'gd'   : pitch_gd,
                }[how]()
    except KeyError:
        raise ValueError('\n'.join(["pitch()...", " wants how={'nmf'|'pinv'|'gd'}", " got how=%s" % how]))

# main
x = pitch(how)
#x = (x-x.min())/(x.max()-x.min())
#x=log(x)
#top_percent = 100 # threshold at brightest top_percentage%
#top_percentile = sorted(flatten(x.tolist()), reverse=True)[int(x.size*top_percent/100)-1]
#dullest = x < top_percentile
#x[dullest] = 0
print x.min()
print x.max()
print x.shape
n_windows = x.shape[0]
d = x.shape[1]

# plot
# x-axis = time in seconds
#  window_rate = sample_rate / sample_size * 2
#  j => j / window_rate
# y-axis = pitch as note (frequency in Hz)
#  i => freqs[i]

if __name__=='__main__':
    window_rate = 2 * sample_rate / window_size # windows per second

    axes = gca()
    axes.imshow(t(x), origin='lower', aspect='auto', interpolation='nearest')

    axes.get_xaxis().set_major_locator(
        LinearLocator(1 + ceil(n_windows/window_rate)))
    axes.get_xaxis().set_major_formatter(
        FuncFormatter(lambda x,y: '%ds' % round(x/window_rate)))

    axes.get_yaxis().set_major_locator(
                LinearLocator(2*d+1))
    axes.get_yaxis().set_major_formatter(
        FuncFormatter(lambda x,y: '%s' % (freqs[(y-1)//2][0] if odd(y) else '')))# if y>0 and y<d else ''))
    draw()

    show()

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
