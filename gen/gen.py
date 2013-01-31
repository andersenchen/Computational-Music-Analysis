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

#:: [x] | (x,) | {x:_} => bool
def unique(xs, key=lambda x: x):
    return len(xs) == len(set(key(x) for x in xs))

#:: [x] | (x,) | {x:_} => bool
def same(xs, key=lambda x: x):
    y = key(xs[0])
    return all(y==key(x) for x in xs)

class MusicException(Exception):
    def __init__(self, message):
        self.message = message
    def __str__(self):
        return repr(self.message)

def save_wav(audio, name, sr=SAMPLE_RATE):
    wavfile.write(name, sr, audio.astype('uint16'))

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

# unique note per freq
if not unique(basis, key=lambda x: music.note(fst(x))):
    raise MusicException('notes not unique')

if not same(basis, key=lambda x: len(snd(x))):
    raise MusicException('audio basis not same size')
SIZE_AUDIO = snd(basis[0]).size

notes    = [music.note(freq) for freq,_ in basis]
notes_ii = ii(notes) # inverted index for notes
nNOTES = len(notes)

piano = { music.note(freq) : audio for freq,audio in basis }
piano.update( 
    { notes_ii[key] : audio for key, audio in piano.items() } )
# piano can be keyed by either String or Number
# eg all(piano['C1']==piano[3])

#wavfile.write('tritone.wav', SAMPLE_RATE, piano['C3']+piano['F#3'])
# sounds like tritone! i know the math, but i had to hear it.
#TODO check to see if record-then-add-notes sounds different than record-interval

# # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Training Data

#:: type
nNOTES = len(notes)
nFREQS = 1024

seconds = 2

#:: |time| x |freqs|
#X = zeros((SAMPLE_RATE * seconds, nFREQS))

#:: |time| x |notes|
#Y = zeros((SAMPLE_RATE * seconds, nNOTES))

#:: |time| x 1
#A = zeros(SAMPLE_RATE * seconds, dtype=dtypes[0])

#TEST
#play(1, 1.0, 'F#3')
#play(1, 1.0, 'C3')
#wavfile.write('ifft.wav',
#              SAMPLE_RATE,
#              ifft(fft(A)).astype('uint16'))
#print 'made tritone'
# $ play tritone.wav
# $ play ifft.wav
# sounds like tritone


# after you make some music
# X = win_fft(A)


def play(X,Y, t,x,n):
    """ note at some time => audio from some time => add by loudness
    
    play note n:str|int (eg 'C3' or 5)
    at time t:int (in seconds)
    with loudness x:float (in [0,1], later normalized)

    assumes the time-length of X and Y are equal
    """
    if type(n)==str: n = notes_ii[n]
    T = X.size
    t = int(t)
    if t < 0 or t > T-1 : raise ValueError('t<0 or t>|X|')
    
    Y[t , n] = x 
    
    audio = piano[n]
    X[ t:t+SIZE_AUDIO ] += audio[:T-t] * x

def tritone(n):
    if type(n)==str:
        return notes[notes_ii[n]+6]
    else:
        return n+6

def play_tritone(X,Y, t=0, n='C3', x=1):
    if type(n)==str: n = notes_ii[n]
    play(X,Y, t,x,n)
    play(X,Y, t,x,tritone(n))


# # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Generative Music Model
"""
WHAT
melody
harmony
accelerando
crescendo

HOW
sparsity
ngrams

"""

model = None


def make_notes_and_sounds(model, T=SAMPLE_RATE * 2):
    """ generate notes (t samples) from a generative music model

    notes : T x N : |samples| x |notes|
    default |samples| is 10sec

    prototype models
    which notes "n"
    how loud "x"
    what time "t"
    
    doesn't need to sound like music
    but should locally look like music

    note
    note := periodic modes + aperiodic attack
    note : R | {0 1}
    periodic note := sum-of-sines + phase|amplitude noise + decay
    aperiodic note := correlated gaussian noise process
    
    
    """
    print 'making notes and sounds...'
    
    X = zeros(T)
    Y = zeros((T, nNOTES))
    play_tritone(X,Y, t=SAMPLE_RATE)
    return X,Y



# # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Generate Infinite Training Data


def inputs_from_sound(sound, window_size = 2**12):
    """ sound => split into overlapping samples => window => fft
    """
    print 'making inputs...'

    hanning_window = hanning(window_size)

    # double <- overlap windows
    nWindows = int32(sound.size/window_size *2)

    spectrum = zeros((nWindows,window_size))

    for i in xrange(0,nWindows-1):
        t = int32(i* window_size/2)
        # elemwise mult
        window = sound[t : t+window_size] * hanning_window
        spectrum[i,:] = fft(window)

    return spectrum


def gen(model):    
    """
    model : { params }
    model is dict ~ call model by ref
    whoever uses gen_notes can update the model
    """

    while True:
        X,Y = make_notes_and_sounds(model)
        yield X,Y

# # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Run Function Approximator 

def run(X):
    """ neural network
    X goes in => hidden layer => Y comes out

    
    """
    

def eval(model, Y, fX):
    """ cmp the true "Y" to the guess "fX" => udpate model given err

    model is dict ~ call model by ref
    """
    
    

for i,(X,Y) in enumerate(gen(model)):
    print i
    X_ = inputs_from_sound(X)
    Y_ = run(X_)
    eval(model, Y, Y_)
    break

