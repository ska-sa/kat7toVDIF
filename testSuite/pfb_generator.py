'''
Created on Jan 30, 2014

@author: benjamin

See inverse pfb GPU code for fuller explanation on the PFB process
'''
from scipy import *
import sys
import numpy as np

if len(sys.argv) != 8:
    print "provide tone input filename, prototype FIR filter, output filename for filtered output, output filename for unfiltered output, sample count, N and P"
    sys.exit(1)
in_file = sys.argv[1]
filter_file = sys.argv[2]
out_filtered_file = sys.argv[3]
out_unfiltered_file = sys.argv[4]
no_samples = int(sys.argv[5])
N = int(sys.argv[6])
P = int(sys.argv[7])
assert(no_samples % N == 0) # make sure that we can process the data in complete blocks of size N

tone = np.fromfile(in_file,dtype=np.int8).astype(np.float32)
w = np.fromfile(filter_file,dtype=np.float32).reshape(P,N)

'''      
FORWARD PFB
'''
pad = N*P      
#take only the non-redundant samples N/2 + 1, by the Hermite-symmetric property of real FFTs, and store'em. (note: we're discarding the highest frequency bin due to the KAT-7 infrastructure)
non_redundant_count = N/2
print ">>>Computing forward PFB"
pfb_input = np.zeros(no_samples + pad).astype(np.float32)
    
pfb_input[pad:pad+no_samples] = tone
pfb_filtered_output = np.zeros(no_samples).astype(np.float32)
pfb_output = np.zeros((no_samples/N)*(non_redundant_count)).astype(np.complex64)
for lB in range(0,no_samples,N):
    pfb_filtered_output[lB:lB+N] = (pfb_input[lB:lB+(P*N)].reshape(P,N)*w).sum(axis=0)
    #we're only storing non-redundant samples to the final output so we have to increment the LHS with N/2 and the RHS by N
    output_array_lB = (lB/N)*(non_redundant_count)
    output_array_uB = output_array_lB + non_redundant_count
    #normalize the FFT output according to Parseval's Theorem, otherwise we'll be missing a scaling factor in IFFT implementations other than that provided in numpy
    pfb_output[output_array_lB:output_array_uB] = np.fft.fft(pfb_filtered_output[lB:lB+N])[:non_redundant_count] / N 
       
print ">>>Computing the comparison unfiltered SFFT"
'''
take Short Time FFTs without filtering for comparison
'''
unfiltered_output = np.zeros((no_samples/N)*(non_redundant_count)).astype(np.complex64)
for lB in range(0,no_samples,N):
    #same as before this time we dont filter, but also take only the non-redundant samples
    output_array_lB = (lB/N)*(non_redundant_count)
    output_array_uB = output_array_lB + non_redundant_count	
    unfiltered_output[output_array_lB:output_array_uB] = np.fft.fft(pfb_input[lB:lB+N])[:non_redundant_count] / N

'''
dump the output
'''
pfb_output.astype(np.complex64).view(dtype=np.float32).astype(np.int8).tofile(out_filtered_file)
unfiltered_output.astype(np.complex64).view(dtype=np.float32).astype(np.int8).tofile(out_unfiltered_file)
