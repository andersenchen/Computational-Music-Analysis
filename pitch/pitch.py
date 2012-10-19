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


def process_wav(file):
    """ 
    spectrum, samplerate = process_wav(filename)

    spectrum :: num_windows by window_size
    |frequencies| = window_size
    num_windows / 2 * |frequencies| ~ |seconds| * sample_rate
    """ 
    print 'processing %s...' % file

    sample_rate, audio = wavfile.read(file)
    #  audio :: |samples| by |channels|
    if len(audio.shape)==1:
        nSamples, = audio.shape
        nChannels = 1
        audio = audio[:int32(nSamples)]
    else:
        nSamples, nChannels = audio.shape
        audio = audio[:int32(nSamples), 0]
    
    #  keep first channel
    #  keep first 2^31 samples
    nWindows = int32(audio.size/window_size *2) # double <- overlap windows
    
    spectrum = zeros((nWindows,window_size))
    true_spectrum = spectrum.copy()

    for i in xrange(0,nWindows-1):
        t = int32(i* window_size/2)
        window = audio[t : t+window_size] * hanning_window # elemwise mult
        true_spectrum[i,:] = fft(window)
        spectrum[i,:] = abs(true_spectrum[i,:])

    return spectrum, sample_rate


def to_freq(file): return int(basename(file)[1:])

def train_joint(dataset):
    print
    print 'TRAINING...'
    
    n  = len(flatten(dataset))
    print 'n =', n

    classifier = zeros((n, window_size))
    
    dataset = [sorted(data, key=to_freq) for data in dataset]
#    def instrument(file): 
#    dataset = [instrument(file) for file in flatten(dataset)]
    #  sort filenames by frequency ascending
    
    freqs = flatten(
        [sorted([note(to_freq(file)) for file in data],
                key=snd) 
#                [tuple([ ('(%s) %s' % (instrument(file),k),f) for k,f note(to_freq(file))]) 
         for data in dataset])
    #  keep track of file's pitch
    #  eg 'A440.wav' => 440
    
    for i,file in enumerate(flatten(dataset)):
        spec, _ = process_wav(file)
        
        # normalize to unit vector
        classifier[i,:] = sum(spec,0) / sum(spec)
        #note  sum by default reduces along all dims to scalar
        
#    print
#    print 'freqs...'
#    for freq in freqs: print freq
#    print
    
    print 'DONE TRAINING.'
    print
    return classifier, freqs

