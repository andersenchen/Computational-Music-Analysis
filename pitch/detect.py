#!/usr/bin/python

# matrix multiplication
#  numpy 'dot(A,B)'  is  matlab 'A*B'
# elemwise multiplication
#  numpy 'A*B'  is  matlab 'A.*B'

from __future__ import division

from music.sambo.sam import *
from pitch import *

from numpy import *
from numpy.fft import fft
from numpy.linalg import pinv
from matplotlib.pyplot import *
import argparse
from scipy.io import wavfile

from sam import *

ROW = 0
COL = 1

OUT_DIR = 'out'
IMAGE_DIR = 'images'
save = False

def stats(A):
    print
    print 'min  = %.3f' % A.min()
    print 'mean = %.3f' % A.mean()
    print 'std  = %.3f' % A.std()
    print 'max  = %.3f' % A.max()

pitch_detection_algorithms = ['nmf','pinv', 'gd'] # enum
def is_pitch_detection_algorithm(s):
    if s in pitch_detection_algorithms: return s
    else: raise ValueError()

dataset_dirs = ['all', 'white', 'black', 'piano','cello'] # enum
def is_dataset(s):
    if s in dataset_dirs: return s
    else: raise ValueError()    

if __name__=='__main__':
    # cli
    p=argparse.ArgumentParser(description="Pitch Detection")
    p.add_argument('file', type=str)
    p.add_argument('-how', type=is_pitch_detection_algorithm,
                   default='nmf',
                   help='nmf | pinv | gd')
    p.add_argument('-data', type=is_dataset, #nargs='*', #rets list
                   default='white',
                   help='piano | white | black | cello | all')
    p.add_argument('-ioff', action='store_true', dest='ioff', help='ioff = interact with plot (ion = interact with program)')
     
    args = p.parse_args()
    file = args.file
    how  = args.how
    data = args.data

    # train classifier
    piano  = [glob('train/piano/*.wav')]
    white  = [glob('train/white/*.wav')]
    black  = [glob('train/black/*.wav')]
    cello  = [glob('train/cello/*.wav')]
    everything = cello + white
    dataset = {'all' : everything,
               'piano' : piano,
               'white' : white,
               'black' : black,
               'cello' : cello,
               }[data]

else:
    #dataset = [glob('train/octave/*.wav')]
    #file = 'chord.wav'
    #how = 'nmf'
    pass

# classifier : |notes| by window_size
classifier, freqs = train_joint(dataset)

# read file
spectrum, sample_rate = process_wav(file)

# consts
nWindows, _ = spectrum.shape

# pseudoinverse solution
# solve Ax=b for x given A,b
# x : (d,1) = notes
# b : (d',1) = audio (known from input)
# A : (d',d) = instrument/note overtone/undertone profile (known from training)
# d = dimensionality of hidden var
# d' = dimensionality of observed var
# eg d = 8 = |keys in 1 piano 8ve| . d' = 4096 = |linear bins of audible pitches|
def pitch_pinv(classifier, spectrum):
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
    
    params = {}
    return t(X), Ai


# nmf solution (ie with multiplicative update)
# solve Ax=b for x
#  where x : nonnegative
def pitch_nmf(classifier, spectrum, iters=50):
    d, window_size = classifier.shape
    A = t(classifier)
    X = (1/d) * ones((d, nWindows))
    B = t(spectrum / sum(spectrum))

    # ignore last sample
    X = X[:,:-1]
    B = B[:,:-1]

    print
    print 'NMF...'
    print '|notes|', d           #eg 8
    print 'sam/win', window_size #eg 4096
    print '|windows|', nWindows  #eg 119
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

    params = {'iters':iters,
              }

    return X, A, params


# gradient descent solution (ie with additive update)
def pitch_gd(classifier, spectrum, iters=50, stepsize=100, eps=1e-12, delta=1e-6, alpha=0):
    n, _ = spectrum.shape
    d, sr = classifier.shape
    A = t(classifier.copy())
    X = (1/d) * ones((d, n))
    B = t(spectrum.copy() / sum(spectrum))
    
    X, B = X[:,:-1], B[:,:-1]
    prior = zeros(X.shape)
    
    print
    print 'GD...'
    for _ in xrange(iters):
        
        # gaussian distance likelihood
        # P( b | A,x ) = N[Ax,1]( b )
        # d/dx    ||Ax - b||^2 + sum log x
        #  =  2A^T||Ax - b||   + sum 1/x
        likelihood = mul( 2 * t(A), mul(A,X) - B )

        # markov stickiness prior
        # P( x[t] | x[t-1] ) = 
        #  scaled translated complemented logistic( ||x[t] - x[t-1]||^2 )
        # wolfram alpha
        #  d/dx -log( (2*e^(-(x-y)^2)) / (1+e^(-(x-y)^2)) )
        time_diff = X[:,1:] - X[:,:-1]
        stats(time_diff)
        #diff = zeros((d,nWindows+1))
        for (i,j) in ndindex((d, n-3)):
            s = j+1
            #  t-1 in 0..n-3
            # (j   in 0..n-3)
            #  t   in 1..n-2
            #  t+1 in 2..n-1
            ab = X[i,s]   - X[i,s-1]
            bc = X[i,s+1] - X[i,s]
            numer_ab = 2 * ab * exp(ab**2)
            denom_ab = 1 + exp(ab**2)
            numer_bc = 2 * bc * exp(bc**2)
            denom_bc = 1 + exp(bc**2)
            prior[i,s] = - (numer_ab / denom_ab) - (numer_bc / denom_bc)
            
        # additive update
        update = likelihood + alpha * prior
        _X = X - stepsize*update
        
        # project onto feasible subspace
        # project onto nonnegative subspace
        #  X >= 0
        #  |nonnegative R^d| / |R^d| = (1/2)^d  ->  nonnegative space is sparse!
        _X[_X < delta] = delta
        # project onto unit simplex = dirichlet support
        #  sum X over notes (not time) == 1
        normalizer = sum(X) # collapse rows i.e. (m,n) => (1,n)
        _X = _X / normalizer
        
        # dynamic stepsize
        stepsize = stepsize * 0.9

        # convergence
        iter_diff = abs(sum(_X - X))
        #if iter_diff < eps: return X,A

        X = _X
        
    print 'diff:',iter_diff
    #print normalizer[:]
    params = {'iters':iters,
              'stepsize':stepsize,
              'alpha':alpha,
              'eps':eps,
              'delta':delta
              }
              
    return X, A, params


def pitch(classifier, spectrum, how='nmf'):
    try:
        return {'nmf'  : pitch_nmf,
                'pinv' : pitch_pinv,
                'gd'   : pitch_gd,
                }[how](classifier, spectrum)
    except KeyError:
        raise ValueError('\n'.join(["pitch()...", " wants how={'nmf'|'pinv'|'gd'}", " got how=%s" % how]))


def threshold(x):
    eps = 0.0001
    x = (x-x.min())/(x.max()-x.min())
    # NOTE wtf!?
    #  i ran this code three times within several seconds, it gave 3 diff plots, only the 3rd looked like the run a few minutes ago.
    x = (x-x.min())/(x.max()-x.min()) # normalize to 0 min
    top_percent = 10 # cutoff at brightest top_percentage%
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

def d2(x, freqs, sample_rate, window_size, title=how):
    d, n_windows = x.shape

    window_rate = 2 * sample_rate / window_size # windows per second

    axes = gca()
    axes.imshow(x,
                cmap=cm.jet, origin='lower', aspect='auto', interpolation='nearest')
    
    axes.set_title(title)
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
    

    if not args.ioff:
        ion()
        import time

    """
    for i in xrange(8):
        chord = ifft(a[:,i])
        plot(chord)
        draw()
        #wavfile.write('%s/chord.%d.wav' % (OUT_DIR, i), sample_rate, chord)
        """

    x,a,params = pitch(classifier, spectrum, how=how)
    #x = threshold(x)

    title = 'Transcription . %s . %s' % (how, file)
    d2(x, freqs, sample_rate, window_size, title=title)
    
    if save:
        image = 'joint, gd, chord'
        savefig( '%s/%s.png' % (IMAGE_DIR, image), bbox_inches=0 )

    if args.ioff:
        show()

    if not args.ioff:
        time.sleep(600)


"""

from compmusic.pitch.detect import *

dataset = [glob('train/piano/*.wav')]
how = 'nmf'
file = 'chord.wav'

#sorted([int(f.split('/')[-1].split('.')[0][1:]) for f in dataset])
A, freqs = train_joint(dataset)
B, sample_rate = process_wav(file)

_A = zeros(A.shape)
for i in xrange(A.shape[0]): 
    in_max = argmax(half(A[i,:]))
    out_max = max(half(A[i,:]))
    _A[i, in_max] = out_max
x,a = pitch_nmf(_A, B)
d2(x, freqs, sample_rate, window_size, title=how)

"""

