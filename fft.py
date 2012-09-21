#!/usr/bin/python
from __future__ import division
from sam import *

from numpy import *
from numpy.fft import *

def half(x): return x[:len(x)//2]

def pfft(frequencies, spectrum):
    print
    print 'u\tF(u)'
    for freq, amp in zip(half(frequencies), half(spectrum)):
        print '%s\t%s' % (freq, smooth(amp))
    print

def radians(n, cycles=2): return cycles * a([x * pi / n for x in xrange(0,n)])
samples = 16
cycles  = 2
times   = radians(samples, cycles=cycles)
f1 = cos( 1 * times )
 # period f1 = all signal
 # unit freq for |cycles| seconds period
 # 1/|cycles| freq for 1 second period
f2 = cos( 2 * times ) #a([1,0,-1,0, 1,0,-1,0])
 # period f2 = half signal
 # f2 cycles twice in signal
f3 = cos( 3 * times ) 
f4 = cos( 4 * times ) #a([1,-1, 1,-1, 1,-1, 1,-1])

#signal = f1 + f2 + f3 + f4
signal = f1 + f2 + f3 + f4

def fft_(signal, samples, sample_rate):
    spectrum = abs(fft(signal)) * sample_rate
    freqs    = fftfreq(samples, d=1/sample_rate)
     # normalize signal to |sample_rate| seconds
    return freqs, spectrum

def print_fft(signal, samples, sample_rate):
    freqs, spectrum = fft_(signal, samples, sample_rate)
    pfft(freqs, spectrum)

from matplotlib.pyplot import *
def plot_fft(signal, samples, sample_rate):
    freqs, spectrum = fft_(signal, samples, sample_rate)
    semilogx( half(freqs), half(spectrum) )
    show()

#plot_fft(signal, samples, cycles)
if __name__=='__main__':
    print_fft(signal, samples, cycles)
