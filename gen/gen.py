#!/usr/bin/python
from __future__ import division

from numpy import *
from matplotlib.pyplot import *
from scipy.stats import *
from numpy.fft import fft,ifft

import nltk, re, pprint
from scipy.io import wavfile
from glob import *

from sam.sam import *
import sam.music as music
from music.pitch import pitch

# # # # # # # # # # # # # # # # # # # # # # # # # # # 
#

#:: [(str,num)]
#notes = music.notes


#:: [str]
src = ['king house piano']
data_dir = 'train/piano/'
data_files = glob(data_dir + '*.wav')
SAMPLE_RATE = 44100

# # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Helpers

#:: str => str
def file2note(s):
    #note, _ = music.note(pitch.to_freq(s))
    return pitch.to_freq(s)

#:: [x] => bool
def unique(xs): return len(xs) == len(set(xs))

class MusicException(Exception):
    def __init__(self, message):
        self.message = message
    def __str__(self):
        return repr(self.message)

def same(xs):
    y = xs[0]
    return all(y==x for x in xs)


# # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Basis

filenames = []
for file in data_files:
    filenames.append(file)
if not unique(filenames):
    raise MusicException('some files in %s have the same name')
else:
    print 'filenames OK'

# basis defining what sounds to listen for and what they sound like
basis  = []
dtypes = []
for file in data_files:
    print 'reading %s...' % file
    note  = file2note(file)
    audio, _ = pitch.audio_wav(file)
    dtypes.append( audio.dtype )
    basis.append( (note, audio) )

if not same(dtypes):
    raise MusicException("some files in the basis have different dtypes -> can't add them up")
print 'dtype of basis is %s' % dtypes[0]

# basis is ordered by notes
# higher index -> higher pitch
# "Y[n,t]" means how loud note "notes[n]" is at time "t"
basis.sort(key=fst)

notes    = [music.note(freq) for freq,_ in basis]
notes_ii = ii(notes) # inverted index for notes

piano =       { music.note(freq) : audio for freq,audio in basis }
piano.update( { notes_ii[freq]   : audio for freq,audio in basis } )
# piano can be keyed by either String (eg 'C1') or Number (eg 3)

#wavfile.write('tritone.wav', 44100, piano['C3']+piano['F#3'])
# sounds like tritone! i know the math, but i had to hear it.
#TODO check to see if record-then-add-notes sounds different than record-interval




# # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Training Data

#:: type
nNotes = len(notes)
nFreqs = 1024

seconds = 2

#:: |time| x |freqs|
X = zeros((sample_rate * seconds, nFreqs))

#:: |time| x |notes|
Y = zeros((sample_rate * seconds, nNotes))

#:: |time| x 1
A = zeros(sample_rate * seconds, dtype=dtypes[0])

def play(n,t,x):
    """ 
    play note n:str (eg 'C3')
    at time t:int (in seconds)
    with loudness x:float (in [0,1], later normalized)
    """
    t = int( t * sample_rate )
    if t < 0 or t > len(A)-1 : raise ValueError('t<0 or t>|A|')
    
    Y[t , notes_ii[n]] = x

    audio = piano[n]
    A[ t:t+len(audio) ] += audio[:len(A)-t]

#TEST
play('F#3', 1, 1.0)
play('C3',  1, 1.0)

#wavfile.write('ifft.wav',
#              sample_rate,
#              ifft(fft(A)).astype('uint16'))
#print 'made tritone'
# $ play tritone.wav
# $ play ifft.wav
# sounds like tritone


# after you make some music
# X = win_fft(A)
