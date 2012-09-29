from __future__ import division

from compmusic.sambo.sam import *
from compmusic.sambo.music import *

from numpy import *
from numpy.fft import *
from matplotlib.pyplot import *

import argparse
from scipy.io import wavfile
from glob import *


window_size = 2**12
hanning_window = hanning(window_size)


# spectrum :: |windows| by |frequencies|
#  |windows| / 2 * |frequencies| ~ |seconds| * |sample_rate|
def process_wav(file):
    print 'processing %s' % file

    sample_rate, audio = wavfile.read(file)
    #  audio :: |samples| by |channels|

    audio = audio[:int32(audio.shape)[0], 0]
    #  keep first channel
    #  keep first 2^31 samples
    n_windows = int32(audio.size/window_size *2) # double <- overlap windows

    spectrum = zeros((n_windows,window_size))
    true_spectrum = spectrum.copy()

    for i in xrange(0,n_windows-1):
        t = int32(i* window_size/2)
        window = audio[t : t+window_size] * hanning_window # elemwise mult
        true_spectrum[i,:] = fft(window)
        spectrum[i,:] = abs(true_spectrum[i,:])

    return spectrum, true_spectrum


def to_freq(file): return int(basename(file)[1:])

def train_joint(data = [glob('train/piano/*.wav')]):
#def train_joint(data = [glob('train/piano/*.wav'), glob('train/cello/*.wav')]):
    print
    print 'TRAINING...'
    
    n  = len(flatten(data))
    print 'n =', n

    classifier = zeros((n, window_size))
    
    data = [sorted(files, key=to_freq) for files in data]
    #  sort filenames by frequency ascending

    freqs = [note(to_freq(file)) for file in flatten(data)]
    #  keep track of file's pitch
    #  eg 'A440.wav' => 440
    
    for i,file in enumerate(flatten(data)):
        spec, _ = process_wav(file)
        
        # normalize to unit vector
        classifier[i,:] = sum(spec,0) / sum(spec)
        #note  sum by default reduces along all dims to scalar
        
    freqs = sorted(set(freqs),key=snd)
#    print
#    print 'freqs...'
#    for freq in freqs: print freq
#    print
    
    print 'DONE TRAINING.'
    print
    return classifier, freqs

