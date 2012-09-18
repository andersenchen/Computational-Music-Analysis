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


#audio = audio*(audio>0) / max(audio) + audio*(audio<0) / abs(min(audio))
# normalize audio -> F(440) ~= 1
# doesnt quite work. adds const (0Hz) signal (= flat line)

# half(fftfreq(n,d)) => [0/n 1/n 2/n ... 1/2) / d
# half(fftfreq(8,1)) => [0/8 1/8 2/8 3/8]

freqs, spectrum = fft_(audio, samples, sample_rate)
#print filter(lambda x: smooth(x[1], eps=0.5) > 0, zip(half(freqs), half(spectrum)) )
print sorted( zip(half(freqs), half(spectrum)), key=lambda ue: ue[0] )
plot_fft(audio, samples, sample_rate)
