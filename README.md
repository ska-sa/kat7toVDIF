INVERSE POLYPHASE FILTERBANK
README v1.0 (RELEASE)
Author: Benjamin Hugo
--------------------------------------------------------------------
Previous testing environment:
University of Cape Town ICTS High Performance Cluster (HEX): http://hex.uct.ac.za
Specs are available on the website
--------------------------------------------------------------------

To compile the cuda code:
--------------------------------------------------------------------
- Ensure you have the NVCC 5.0 installed
- Ensure you have CMAKE installed
- You must have a device with Compute Capability >= 2.0 installed

1. create a build directory in the directory the source files are located (if one exists it's better to empty it first) and navigate into the newly created directory. 
2. Hit 'cmake ../'. This will locate the necessary cuda libraries and link them appropriately for your system. 
3. Now you can hit 'make' to compile the source.

To tweak filtering and or other coefficients 
--------------------------------------------------------------------
All the constants should be located in inv_pfb.h

To generate some test data
--------------------------------------------------------------------
A 'testSuite' directory with python scripts should be distributed with the CUDA code. 
IMPORTANT NOTE:
In that directory you will find the python versions of the pfb and inverse pfb processes
along with a few other script. Don't bother modifying those unless you're bug fixing. 

Locate the BASH script construct_test_data.sh. In this script you can configure the fi:ltering coefficients and number of 
test samples (as well as their type: 'sine', 'impulse' or 'noise'). Executing that script will dump a fake signal, prototype filter (to be used for both analysis and synthesis),
forward pfb output and inverse pfb output. You can use this to test the CUDA code.

There should be a 'correlator.py' X-correlator, a mean squared error 'diff_real_inputs.py' and a signal to noise 'snr.py' analyser included (the last two takes only two REAL-valued streams) included to aid you in any validation.

To run the CUDA code
--------------------------------------------------------------------
arguements: [filter file] [pfb output (non-redundant samples only)] [where to put the inverse pfb output]

