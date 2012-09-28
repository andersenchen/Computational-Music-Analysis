from __future__ import division
from sam import *

from numpy import *
from numpy.fft import *
from matplotlib.pyplot import *

import argparse
from scipy.io import wavfile
from glob import *


def process_wav(file):
    print 'processing %s...', file

    sample_rate, audio = wavfile.read(file)
    #  audio :: |samples| by |channels|

    audio = audio[:int32(audio.shape)[0], 0]
    #  keep first channel
    #  keep first 2^31 samples

    windowSize = 2**12
    nWindows = int32(audio.size/windowSize *2) # double <- overlap windows
    hanningWindow = hanning(windowSize)
    spectrum = zeros((nWindows,windowSize))
    true_spectrum = spectrum.copy()
    #  spectrum :: |time bins| by |frequency bins|

    for i in r(0,nWindows-3): #nWindows-3?
        t = int32(i* windowSize/2)
        window = audio[t : t+windowSize] * hanningWindow # elemwise mult
        true_spectrum[i,:] = fft(window)
        spectrum[i,:] = abs(true_spectrum[i,:])

    return spectrum, true_spectrum


def to_freq(file): return int(basename(file)[1:])

def train_joint(data = [glob('train/piano/*.wav'), glob('train/cello/*.wav')]):
    n  = len(flatten(data))
    print 'n =', n

    classifier = zeros((n, 4096))
    tclass     = classifier.copy()
    
    data = [sorted(files, key=to_freq) for files in data]
    #  sort filenames by frequency ascending
    freqs = zeros(n)

    for i,file in enumerate(flatten(data)):
        spec, tspec = process_wav(file)
        
        # normalize to unit vector
        classifier[i,:] = sum(spec)  /     sum(sum(spec))
        tclass    [i,:] = sum(tspec) / abs(sum(sum(tspec)))
        
        # keep track of file's pitch
        #eg 'A440.wav' => 440
        freqs[i] = to_freq(file)
        
    freqs = sorted(set(freqs))
    print
    print 'freqs...'
    for freq in freqs: print freq
    
    return classifier, tclass, freqs

