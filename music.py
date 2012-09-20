#!/usr/bin/python -W ignore::Warning
from __future__ import division
from sam import *

from fft import *

K = 0
F = 1

keys = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b']
def piano():
    notes = [('g#0', 25.9565)]
    notes = notes + zip(['a0', 'a#0', 'b0'], [27.5000, 29.1352, 30.8677])
    notes = notes + [ (keys[key] + str(octave), 16.3516 * 2**(octave + key/12))  for octave in xrange(1,7+1)  for key in xrange(0,11+1) ]
    notes = notes + [('c8', 4186.01)]
    return notes

#:: [(key, freq)]
notes = piano()

key, freq = notes[49] 
assert 'a4' == key
assert about( 440, freq )


# note 440 = a4
def note(freq):
    # runtime should be const
    # distance should be log
    # whatevs
    
    return reduce((lambda x,y: x if nearer(freq, x[F],y[F]) else y), notes)

print note(440)
key, freq = note(440)
assert 'a4' == key

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # sine wave <= sox

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

#audio = audio*(audio>0) / max(audio) + audio*(audio<0) / abs(min(audio))
# normalize audio -> F(440) ~= 1
# doesnt quite work. adds const (0Hz) signal (= flat line)

# half(fftfreq(n,d)) => [0/n 1/n 2/n ... 1/2) / d
# half(fftfreq(8,1)) => [0/8 1/8 2/8 3/8]

def main(filename):
    window_size = 1024

    from scipy.io import wavfile
    sample_rate, audio = wavfile.read(filename)
    audio = audio - mean(audio) #normalize

    #eg 44100 // 1024 * 1024 == 4032 (ignore last few samples for consistent signal size)
    for t in arange(0, (sample_rate // window_size) * window_size, window_size):

        signal = audio[t : t+window_size]
        freqs, spectrum = fft_(signal, window_size, sample_rate)
        #print filter(lambda x: smooth(x[1], eps=1), zip(half(freqs), half(spectrum)) )
        print sorted( zip(half(freqs), half(spectrum)), reverse=True, key=lambda ue: ue[0] )[0:3]
        plot_fft(signal, window_size, sample_rate)
        break

if __name__=='__main__':
    import sys
    filename = sys.argv[1] if len(sys.argv)>1 else 'short.wav'
    
    main(filename)
