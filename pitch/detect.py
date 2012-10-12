#!/usr/bin/python

# matrix multiplication
#  numpy 'dot(A,B)'  is  matlab 'A*B'
# elemwise multiplication
#  numpy 'A*B'  is  matlab 'A.*B'

from __future__ import division

from compmusic.sambo.sam import *
from pitch import *

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
p.add_argument('file', type=str)
p.add_argument('how', type=pitch_detection_algorithm,
               nargs='?', default='nmf',
               help='nmf | pinv | gd')
args = p.parse_args()
file = args.file
how  = args.how

# train classifier
piano = [glob('train/piano/*.wav')]
cello = [glob('train/cello/*.wav')]
dataset = piano + cello
# classifier :: |notes| by window_size
classifier, freqs = train_joint(dataset)

# read file
spectrum, sample_rate = process_wav(file)

# consts
nWindows = spectrum.shape[0]

# pseudoinverse solution
# solve Ax=b for x given A,b
# x : (d,1) = notes
# b : (d',1) = audio (known from input)
# A : (d',d) = instrument/note overtone/undertone profile (known from training)
# d = dimensionality of hidden var
# d' = dimensionality of observed var
# eg d = 8 = |keys in 1 piano 8ve| . d' = 4096 = |linear bins of audible pitches|
def pitch_pinv():
    A  = classifier
    Ai = pinv(A)
    B  = spectrum / sum(spectrum)
    X  = zeros((nWindows, classifier.shape[0]))
    
    print 'PINV...'
    print 'A', A.shape
    print 'Ai', Ai.shape
    print 'X', X.shape
    print 'B', B.shape
    X = dot( B, Ai )
    
    return X


# nmf solution (ie with multiplicative update)
# solve Ax=b for x
#  where x : nonnegative
def pitch_nmf():
    d, _ = classifier.shape
    A = t(classifier)               #eg 1024, 8
    X = 0.5 * ones((d, nWindows))   #eg 8, 60
    B = t(spectrum / sum(spectrum)) #eg 1024, 60

    # ignore last sample
    X = X[:,:-1]
    B = B[:,:-1]
    
    """
    print 'NMF...'
    print 'd', d
    print 'sr', classifier.shape[1]
    print 'nW', nWindows
    print 'us', spectrum.shape
    print 'A', A.shape
    print 'b', spectrum[0,:].shape
    print 'x', X.shape
    print
    """
    print 'NMF...'
    print 'A', A.shape
    print 'X', X.shape
    print 'B', B.shape
    
    # jointly solve Ax=b forall samples
    # multiplicative update with euclidean distance
    for k in xrange(20): # until convergence
        numer = dot( t(A), B )    #: 8,1024 * 1024,60
        denom = mul( t(A), A, X ) #: 8,1024 * 1024,8 * 8,60
        X = X * numer / denom
            
    return t(X)


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


def threshold(x):
    eps = 0.0001
    x = eps + (x-x.min())/(x.max()-x.min()) # make positive for logarithm
    x=log(x)
    # NOTE wtf!?
    #  i ran this code three times within several seconds, it gave 3 diff plots, only the 3rd looked like the run a few minutes ago.
    x = (x-x.min())/(x.max()-x.min()) # normalize to 0 min
    top_percent = 25 # threshold at brightest top_percentage%
    top_percentile = sorted(flatten(x.tolist()), reverse=True)[int(x.size*top_percent/100)-1] # sort desc
    dullest = x < top_percentile
    x[dullest] = 0

    return x


#Main
x = pitch(how)
#x = threshold(x)

#Plot
# x-axis = time in seconds
#  window_rate = sample_rate / sample_size * 2
#  j => j / window_rate
# y-axis = pitch as note (frequency in Hz)
#  i => freqs[i]

if __name__=='__main__':
    ion()
    import time

    nWindows, d = x.shape

    window_rate = 2 * sample_rate / window_size # windows per second

    axes = gca()
    axes.imshow(t(x), cmap=cm.jet, origin='lower', aspect='auto', interpolation='nearest')

    axes.set_title('Transcription')
    axes.get_xaxis().set_major_locator(
        LinearLocator(1 + ceil(nWindows/window_rate)))
    axes.get_xaxis().set_major_formatter(
        FuncFormatter(lambda x,y: '%ds' % round(x/window_rate)))

    axes.get_yaxis().set_major_locator(
                LinearLocator(2*d+1))
    axes.get_yaxis().set_major_formatter(
        FuncFormatter(lambda x,y: '%s' % (freqs[(y-1)//2][0] if odd(y) else '')))# if y>0 and y<d else ''))

    draw()
    time.sleep(60)
    
    #show()


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
