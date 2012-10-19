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
from scipy.io import wavfile

from sam import *

OUT_DIR = 'out'
IMAGE_DIR = 'images'

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
dataset = piano# + cello
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
    
    print
    print 'PINV...'
    print 'A', A.shape
    print 'Ai', Ai.shape
    print 'X', X.shape
    print 'B', B.shape
    X = dot( B, Ai )
    
    return t(X), Ai


# gradient descent solution (ie with additive update)
def pitch_gd(iters=100, stepsize=100):
    d, sr = classifier.shape
    A = t(classifier)
    X = (1/d) * ones((d, nWindows))
    B = t(spectrum / sum(spectrum))

    X = X[:,:-1]
    B = B[:,:-1]
    
    print
    print 'GD...'
    for _ in xrange(iters):
        # additive update
        AX = mul(A,X)
        update = mul( t(A), AX-B )
        X = X - stepsize*update
        
        # project onto nonnegative
        # |nonnegative R^d| / |R^d| = (1/2)^d  ->  nonnegative space is sparse!
        X[X<0] = 0
        
        # dynamic stepsize
        stepsize = stepsize * 0.9
        
    return X,A

# nmf solution (ie with multiplicative update)
# solve Ax=b for x
#  where x : nonnegative
def pitch_nmf(iters=50):
    d, sr = classifier.shape
    A = t(classifier)
    X = (1/d) * ones((d, nWindows))
    B = t(spectrum / sum(spectrum))

    # ignore last sample
    X = X[:,:-1]
    B = B[:,:-1]

    print
    print 'NMF...'
    print '|notes|', d          #eg 8
    print 'sample rate', sr     #eg 4096
    print '|windows|', nWindows #eg 119
    print 'A', A.shape #eg 4096, 8
    print 'X', X.shape #eg 8, 119
    print 'B', B.shape #eg 4096, 119
    
    # jointly solve Ax=b forall samples
    # multiplicative update with euclidean distance
    for _ in xrange(iters): # until convergence

        numerX = mul( t(A), B )    #: 8,1024 * 1024,60
        denomX = mul( t(A), A, X ) #: 8,1024 * 1024,8 * 8,60
        _X = X * numerX / denomX   #: 8,60
        
        #numerA = mul( B, t(X) )    #: 1024,60 * 60,8
        #denomA = mul( A, X, t(X) ) #: 1024,8 * 8,60 * 60,8
        #A = A * numerA / denomA    #: 1024,8

        X = _X
        
    return X, A


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
    x = (x-x.min())/(x.max()-x.min())
    # NOTE wtf!?
    #  i ran this code three times within several seconds, it gave 3 diff plots, only the 3rd looked like the run a few minutes ago.
    x = (x-x.min())/(x.max()-x.min()) # normalize to 0 min
    top_percent = 10 # threshold at brightest top_percentage%
    top_percentile = sorted(flatten(x.tolist()), reverse=True)[int(x.size*top_percent/100)-1] # sort desc
    dullest = x < top_percentile
    x[dullest] = 0
    
    return x

# Bug#636364: ipython

#Plot
# x-axis = time in seconds
#  window_rate = sample_rate / sample_size * 2
#  j => j / window_rate
# y-axis = pitch as note (frequency in Hz)
#  i => freqs[i]

def d2(x):
    d, n_windows = x.shape

    window_rate = 2 * sample_rate / window_size # windows per second

    axes = gca()
    axes.imshow(x,
                cmap=cm.jet, origin='lower', aspect='auto', interpolation='nearest')

    axes.set_title('Transcription')
    axes.get_xaxis().set_major_locator(
        LinearLocator(1 + ceil(n_windows/window_rate)))
    axes.get_xaxis().set_major_formatter(
        FuncFormatter(lambda x,y: '%ds' % round(x/window_rate)))
    
    axes.get_yaxis().set_major_locator(
                LinearLocator(2*d+1))
    axes.get_yaxis().set_major_formatter(
        FuncFormatter(lambda x,y: '%s' % (freqs[(y-1)//2][0] if odd(y) else '')))
    
    draw()

def d3(Z):
    X = a(r(1,times))
    Y = a(r(1,freqs))
    X, Y = meshgrid(X, Y)
    print X.shape
    print Y.shape
    print Z.shape

    from mpl_toolkits.mplot3d import Axes3D
    fig = figure()
    ax = fig.gca(projection='3d')
    times, freqs = Z.shape
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,
                           linewidth=0, antialiased=False)
    draw()

if __name__=='__main__':
    
    x,a = pitch(how)
    #x = threshold(x)
    
    if True:
        ion()
        import time

    """
    for i in xrange(8):
        chord = ifft(a[:,i])
        plot(chord)
        draw()
        #wavfile.write('%s/chord.%d.wav' % (OUT_DIR, i), sample_rate, chord)
        """
    d2(x)

    image = False
    if image:
        image = 'joint, gd, chord'
        savefig( '%s/%s.png' % (IMAGE_DIR, image), bbox_inches=0 )

    time.sleep(60)

    if not True:
        show()
