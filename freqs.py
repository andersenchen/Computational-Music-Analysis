#!/usr/bin/python -W ignore::Warning
from __future__ import division
from sam import *
from music import *
from fft import *

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

def main(filename, gui):
    window_size = 1024

    from scipy.io import wavfile
    sample_rate, audio = wavfile.read(filename)
    #audio = audio - mean(audio) #normalize

    #eg 44100 // 1024 * 1024 == 4032 (ignore last few samples for consistent signal size)
    for t in arange(0, (sample_rate // window_size) * window_size, window_size):
        t = t + 48000
        signal = audio[t : t+window_size]
        freqs, spectrum = fft_(signal, window_size, sample_rate)
        #print filter(lambda x: smooth(x[1], eps=1), zip(half(freqs), half(spectrum)) )

        top_freqs = sorted( zip(map(note, half(freqs)), half(spectrum)), reverse=True, key=lambda ue: ue[1] )
        print_freqs(top_freqs)
        if gui: plot_fft(signal, window_size, sample_rate)

        break

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description="fft music")
    parser.add_argument('-gui', dest='gui', action='store_true', default=False, help='whether to show plot')
    parser.add_argument('filename', help='the audio file to fft')
    args = parser.parse_args()
    
    main(args.filename, args.gui)
