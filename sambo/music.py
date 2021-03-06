#!/usr/bin/python -W ignore::Warning
from __future__ import division

from fft import *
from numpy import *

from sam import *

K = 0
F = 1

keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
def piano():
    notes = [('G#0', 25.9565)]
    notes = notes + zip(['A0', 'A#0', 'B0'], [27.5000, 29.1352, 30.8677])
    notes = notes + [ (keys[key] + str(octave), 16.3516 * 2**(octave + key/12))  for octave in xrange(1,7+1)  for key in xrange(0,11+1) ]
    notes = notes + [('C8', 4186.01)]
    
    notes = [(k,round(f)) for k,f in notes]
    return notes

#:: [(key, freq)]
notes = piano()
freqs = dict(notes)

key, frequency = notes[49] 
assert 'A4' == key
assert about( 440, frequency )


# note 440 = a4
def note(freq):
    # runtime should be const
    # distance should be log
    # whatevs
    
    return reduce((lambda x,y: x if nearer(freq, x[F],y[F]) else y), notes)

key, frequency = note(440)
assert 'A4' == key


def freq(note):
    return freqs[note.upper()] if note.upper() in freqs else None

assert about( freq('a4'), 440 )


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

def sort_fft(freqs, spectrum):
    return sorted( zip(half(freqs), half(spectrum)), reverse=True, key=lambda ue: ue[1] )

def print_fft(fft, k=10, simplify=round):
    for (u,e) in fft[:k]: print '\t'.join([str(note(u)), str(simplify(e))])

def main(filename, gui):
    if gui:
        import time
        ion()

    window_size = 1024
    hanning_window = hanning(window_size)

    from scipy.io import wavfile
    sample_rate, audio = wavfile.read(filename)
    audio, _, _ = onechannel(audio) # casts nChannels to 1
    audio = audio - mean(audio) #normalize
    
    #eg 44100 // 1024 * 1024 == 4032 (ignore last few samples for consistent signal size)
    start = sample_rate # start at 1 second
    k     = 3
    fs    = []
    ss    = []
    for t in arange(start, 
                    start + (sample_rate // window_size) * window_size,
                    window_size):

        signal = audio[t : t+window_size] * hanning_window
        freqs, spectrum = fft_(signal, window_size, sample_rate)
        fs.append(freqs)
        ss.append(spectrum)

    ymin = min( min(ys) for ys in ss )
    ymax = max( max(ys) for ys in ss )
    for freqs,spectrum in zip(fs,ss):

        #print filter(lambda x: smooth(x[1], eps=1), zip(half(freqs), half(spectrum)) )
        top = sort_fft(freqs, spectrum)[:k]
        print
        print_fft(top)
    
        if gui:
            xs = half(freqs)
            ys = half(spectrum)

            gcf().clear()
            ylim( ymin, ymax )
            xscale('log')
            plot(xs,ys,'o-')
            draw()

            time.sleep(1/40)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description="fft music")
    parser.add_argument('-gui', dest='gui', action='store_true', default=False, help='whether to show plot')
    parser.add_argument('filename', help='the audio file to fft')
    args = parser.parse_args()

    filename = args.filename
    gui = args.gui
    main(filename,gui)
