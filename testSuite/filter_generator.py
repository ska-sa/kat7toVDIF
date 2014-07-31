'''
Created on Jan 30, 2014

@author: benjamin
'''
import scipy
import scipy.signal
import sys
import numpy as np


if len(sys.argv) != 4:
    print "provide output filename, N and P"
    sys.exit(1)
out_file = sys.argv[1]
N = int(sys.argv[2])
P = int(sys.argv[3])
print ">>>Creating prototype FIR filter"
#setup the windowing function
w = scipy.signal.firwin(P * N, 1. / N) * N
w.astype(np.float32).tofile(out_file)
