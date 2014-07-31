'''
Created on 29 Jan 2014

@author: benjamin
'''
#from pylab import *
from scipy import *
import sys
import numpy as np
#import scipy.signal
if len(sys.argv) != 6:
    print "provide output filename, sample count, wave type [sine / noise / impulse], N and P"
    sys.exit(1)

out_file = sys.argv[1]
no_samples = int(sys.argv[2])
tone_generation_mode = sys.argv[3]
N = int(sys.argv[4])
P = int(sys.argv[5])

'''
sine wave constraints
'''
sampling_freq = 800e6
max_freq = 0.5 * sampling_freq
no_bins = N / 2 + 1
tone_freq = [max_freq / float(no_bins) * (100),max_freq / float(no_bins) * (150.45)]
impulse_shift = N*P + 0.25*N

'''
generate a fake tone
'''
print ">>>Generating tone"
tone = np.zeros(no_samples).astype(np.float32) # make sure the tone is a 32bit float tone
if tone_generation_mode == "sine":
    scale_factor = 1/len(tone_freq) #normalization factor
    for f in tone_freq:
        tone += scale_factor * np.sin(2 * np.pi * np.arange(0,no_samples) * (f / float(sampling_freq)))
        print "frequency %f MHz should be in channel %f after filtering" % (f / float(1e6),f / (max_freq / float(no_bins)))
    tone *= 127 #scale to fill signed 8bit integer output
elif tone_generation_mode == "noise":
    tone = np.random.randn(no_samples) #Gausian noise
    tone = (tone/abs(tone).max())*127 #normalize and scale to fill signed 8bit integer output
elif tone_generation_mode == "impulse":
    print "impulse at sample %d" % impulse_shift
    tone[impulse_shift] = 127
else:
    print "invalid tone generation option"
    exit(1)
    
tone.astype(np.int8).tofile(out_file)
