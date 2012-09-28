#!/usr/bin/python

# matrix multiplication
#  numpy 'dot(A,B)'  is  matlab 'A*B'
# elemwise multiplication
#  numpy 'A*B'  is  matlab 'A.*B'

from sam import *
from pitch import *
#from train_joint import train_joint

from numpy import *
from numpy.fft import *
from matplotlib.pyplot import *

# Train the classifier
classifier, tclass, freqs = train_joint()

