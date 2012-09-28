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


def train_join(files = glob('train/piano/*.wav') + glob('train/cello/*.wav')):
    classifier = zeros(2* length(index), 4096)
    tclass     = classifier.copy()

    n = len(files)
    IDX = zeros((n,1))
    IDXwork = zeros((n,1))
    
    for i,file in enumerate(files):
        spec, tspec = processWav(file)

        # normalize to unit vector
        classifier[i,:] = sum(spec)  /     sum(sum(spec))
        tclass[i,:]     = sum(tspec) / abs(sum(sum(tspec)))

        # keep track of file's pitch
        #eg 'A440.wav' => 440
        IDXwork[i] = int(basename(file)[1:])

    permutation = argsort(IDXwork)
    IDXwork = IDXwork[permutation]
    classifier[1:n, :] = classifier[permutation, :]
    tclass    [1:n, :] = tclass    [permutation, :]
    IDX       [1:n]    = IDXwork
    
    return classifier, tclass

