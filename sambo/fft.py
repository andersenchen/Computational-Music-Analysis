#!/usr/bin/python
from __future__ import division
from sam import *

from numpy import *
from numpy.fft import *
from matplotlib.pyplot import *

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
    spectrum = abs(fft(signal))**2 * sample_rate
    freqs    = fftfreq(samples, d=1/sample_rate)
     # normalize signal to |sample_rate| seconds
    return freqs, spectrum

def print_fft(signal, samples, sample_rate):
    freqs, spectrum = fft_(signal, samples, sample_rate)
    pfft(freqs, spectrum)

def normalize(arr):
    arr = array(arr)
    low = min(arr)
    high = max(arr)
    return (arr-low)/(high-low)

def get_axes(xs, ys, pad=0.1):
    ax = min(xs)
    bx = max(xs)
    ay = min(ys)
    by = max(ys)
    return [ax*(1-pad), bx*(1+pad)] + [ay*(1-pad), by*(1+pad)]

def plot_fft(signal, samples, sample_rate):
    freqs, spectrum = fft_(signal, samples, sample_rate)
    xs = half(freqs)
    ys = normalize(half(spectrum))
    scatter(xs, ys)
    xscale('log')
#    axis(get_axes(xs,ys))
    show()

#plot_fft(signal, samples, cycles)
if __name__=='__main__':
    print_fft(signal, samples, cycles)
