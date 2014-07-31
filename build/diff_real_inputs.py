'''
Created on Jan 30, 2014

@author: benjamin


This script investigates the difference between two REAL-valued input streams and 
should be useful in validating the output of a CUDA inverse PFB process against
its Python counterpart.
'''
#from pylab import *
import sys
import scipy.signal
import numpy as np
if len(sys.argv) != 3:
        print "please specify [input file 1], [input file 2]"
        sys.exit(1)
        
inp1 = np.fromfile(sys.argv[1],dtype=np.int8).astype(np.float32)
inp2 = np.fromfile(sys.argv[2],dtype=np.int8).astype(np.float32)
inp1_mean = np.mean(inp1)
inp2_mean = np.mean(inp2)
inp1_stdev = np.std(inp1)
inp2_stdev = np.std(inp2)
mse = ((inp1 - inp2) ** 2).mean()

print "'%s' has mean = %f and standard deviation = %f" % (sys.argv[1],inp1_mean,inp1_stdev)
print "'%s' has mean = %f and standard deviation = %f" % (sys.argv[2],inp2_mean,inp2_stdev)
print "\033[1;33mThe Mean squared error between the files is %f\033[0m" % (mse)

