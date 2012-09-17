#!/usr/bin/python
from __future__ import division

from numpy import *
from numpy.fft import *

def half(x): return x[:len(x)//2]
a=array
def smooth(x, eps=1e-9): return x if abs(x) > eps else 0
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

def fft_(signal, samples, cycles):
    spectrum = abs(fft(signal)) * cycles/samples
    freqs    = fftfreq(samples, d=cycles/samples)
     # normalize signal to |cycles| seconds
    return freqs, spectrum

def print_fft(signal, samples, cycles):
    freqs, spectrum = fft_(signal, samples, cycles)
    pfft(freqs, spectrum)

from matplotlib.pyplot import *
def plot_fft(signal, samples, cycles):
    freqs, spectrum = fft_(signal, samples, cycles)
    semilogx( half(freqs), half(spectrum) )
    show()

#plot_fft(signal, samples, cycles)
if __name__=='__main__':
    print_fft(signal, samples, cycles)