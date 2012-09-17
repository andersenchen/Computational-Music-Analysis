#!/usr/bin/python -W ignore::Warning

from fft import *

# # # sine wave <= sox

import sys
filename = sys.argv[1] if len(sys.argv)>1 else 'short.wav'

from scipy.io import wavfile
sample_rate, audio = wavfile.read(filename)
audio = audio - mean(audio) #normalize

from matplotlib import *
#plot(audio[48000//10 : 2*48000//10]); show(); exit(1)

""" sine.wav
= 440Hz pure tone

soxi sine.wav
 sample rate = 48k Hz = 48k samples per second
 precision = 32 bit
 sample encoding = 32-bit Signed Integer PCM

min(audio) == -2**31
max(audio) == 2**31
"""

#audio = audio*(audio>0) / max(audio) + audio*(audio<0) / abs(min(audio))
# normalize audio -> F(440) ~= 1
# doesnt quite work. adds const (0Hz) signal (= flat line)
samples = len(audio)
cycles  = 3 # 3 sec

freqs, spectrum = fft_(audio, samples, cycles)
print filter(lambda x: smooth(x[1], eps=0.5) > 0, zip(half(freqs), half(spectrum)) )
plot_fft(audio, samples, cycles)
