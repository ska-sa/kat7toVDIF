/**
  _____ _   ___      ________ _____   _____ ______   _____  ______ ____  
 |_   _| \ | \ \    / /  ____|  __ \ / ____|  ____| |  __ \|  ____|  _ \ 
   | | |  \| |\ \  / /| |__  | |__) | (___ | |__    | |__) | |__  | |_) |
   | | | . ` | \ \/ / |  __| |  _  / \___ \|  __|   |  ___/|  __| |  _ < 
  _| |_| |\  |  \  /  | |____| | \ \ ____) | |____  | |    | |    | |_) |
 |_____|_| \_|   \/   |______|_|  \_\_____/|______| |_|    |_|    |____/ 
                                                                         
This is a CUDA implementation of the inverse Polyphase Filter Bank (PFB), also known as the Weighted 
Overlap Add Method (WOA). It constructs a basic synthesis filterbank that can process output in strides 
and is therefore suitable for real time operation, where the input length is not known in advance. Note 
that this filterbank construction does not provide Perfect Reconstruction (PR) of the original input 
to the PFB. 

Background theory:
        For the forward process see:
                https://casper.berkeley.edu/wiki/The_Polyphase_Filter_Bank_Technique
        The inverse process has its subfilters flipped, but follows the same trend, see:
                A REVIEW OF POLYPHASE FILTER BANKS AND THEIR APPLICATION --- Daniel Zhou. 
                Air Force Research Laboratory. AFRL-IF-RS-TR-2006-277 In-House Final 
                Technical Report, September 2006.


Copyright (C) 2014, Square Kilometer Array (SKA) South Africa
@author Benjamin Hugo (bennahugo __AT__ aol __DOT__ com)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "inv_pfb.h"

/****************************************************************************************************************
 variables
*****************************************************************************************************************/
cufftReal * d_ifft_output;
float * d_taps;
cufftHandle ifft_plan;
complex_int8 * d_cast_input;
cufftComplex * d_cast_output;
int8_t * d_filtered_output;
cudaStream_t ifft_stream,ifft_mov_output_cpy_stream;

/****************************************************************************************************************
Forward declare kernels
****************************************************************************************************************/
__global__ void array_cast_complex_int8_to_cufftComplex(complex_int8 * in, cufftComplex * out, 
							uint32_t total_no_samples_casted_by_this_kernel, uint32_t no_blocks_in_dim,
							uint32_t size_of_padded_fft_block);
__global__ void ipfb(const cufftReal * input,  int8_t * output, const float * prototype_filter, uint32_t no_samples_to_filter, uint32_t no_blocks_in_dim);
__global__ void move_last_P_iffts_to_front(cufftReal * ifft_array, uint32_t start_of_last_P_N_block);
/**
This method initializes the device. It selects the GPU, allocates all the memory needed to perform the inverse pfb process and copies the prototype filter
onto the device. ***WARNING***: this method must be called before the first stride is processed. The device should be released after all processing is completed.
@args taps pointer to a preloaded prototype filter (preferably a hamming windowed FIR filter with cutoff at 1/N)
@postcondition device (if any) is ready to perform the inverse pfb process on multiple strides of data.
*/
void initDevice(const float * taps){
	//do some checks to see if we're initing a reasonable gpu setup:
	assert((LOOP_LENGTH % FFT_SIZE) == 0); //The data being uploaded to the device must consist of an integral number of FFT blocks.

	//Choose a reasonably good device(https://www.cs.virginia.edu/~csadmin/wiki/index.php/CUDA_Support/Choosing_a_GPU), based on # SMs:
	int num_devices, device;
	cudaGetDeviceCount(&num_devices);
	if (num_devices > 0) {
		//get the argmax{devID}(multiProcessorCounts):
      		int max_multiprocessors = 0, max_device = 0;
      		for (device = 0; device < num_devices; device++) {
              		cudaDeviceProp properties;
	              	cudaGetDeviceProperties(&properties, device);
			if (max_multiprocessors < properties.multiProcessorCount) {
         	             max_multiprocessors = properties.multiProcessorCount;
                	     max_device = device;
			}
		}
		cudaSetDevice(max_device); //select device
        	cudaDeviceReset(); //ensure device is in a safe state before we begin processing
		
		//print some stats:
        	cudaDeviceProp properties;
	        cudaGetDeviceProperties(&properties, max_device);

        	size_t mem_tot = 0;
	        size_t mem_free = 0;
	        cudaMemGetInfo  (&mem_free, & mem_tot);
        	printf("---------------------------------------------------------\n\033[0;31mChosen GPU Statistics\033[0m\n---------------------------------------------------------\n");
	        printf("%s, device %d on PCI Bus #%d, clocked at %f GHz\n",properties.name,properties.pciDeviceID,
        	        properties.pciBusID,properties.clockRate / 1000000.0);
	        printf("Compute capability %d.%d with %f GiB global memory (%f GiB free)\n",properties.major,
        	        properties.minor,mem_tot/1024.0/1024.0/1024.0,mem_free/1024.0/1024.0/1024.0);
	        printf("%d SMs are available\n",properties.multiProcessorCount);
	        printf("---------------------------------------------------------\n");
	} else {
		fprintf(stderr,"Cannot find suitable GPU device. Giving up");
		exit(-1);
	}
	/*For now we'll copy the taps into global memory. It doesn't make sense to copy this to constant memory
	  as the individual threads in each warp will be accessing different locations, and will therefore be serialized.
	  Instead memory calls should be coalesced for each warp of threads due to the nice accessing pattern of the
	  Weighted Window Overlap Add method. Another optimization trick that one may try is to copy this into texture memory
	  where there may be a slight performance increase due to the texture caching properties of the GPU. From experimentation
	  we found that normal coalesced reads are actually faster, so this was a dead end as well.
	*/
	printf("\033[0;31mInitialization routines\033[0m\n---------------------------------------------------------\nINIT: Copying prototype filter of %d taps to device\n",WINDOW_LENGTH);
	cudaSafeCall(cudaMalloc((void**)&d_taps,sizeof(float)*WINDOW_LENGTH));
	cudaSafeCall(cudaMemcpy(d_taps,taps,sizeof(float)*WINDOW_LENGTH,cudaMemcpyHostToDevice));

	/*
	 Setup the ifft buffer where we will keep P extra IFFTs (each of length N) carried over
	 from the previous stride-processing iteration.
	*/
	printf("INIT: Setting up IFFT output buffer of length %d\n", BUFFER_LENGTH);
	cudaSafeCall(cudaMalloc((void**)&d_ifft_output,sizeof(cufftReal) * (BUFFER_LENGTH)));
	printf("INIT: Ensuring initial filter padding of %d elements is set to 0\n",PAD);
	cudaSafeCall(cudaMemset(d_ifft_output,0,sizeof(cufftReal)*PAD));
	printf("INIT: Setting up cast input buffer of length %d*complex_int8\n",LOOP_LENGTH);
	cudaSafeCall(cudaMalloc((void**)&d_cast_input,sizeof(complex_int8) * LOOP_LENGTH));
	//Setup CU IFFT plan to process all the N-sample iffts contained in a single loop, in one go
	printf("INIT: Setting up IFFT plan for %d blocks of FFTs\n",MAX_NO_BLOCKS);
	int rank_sizes[1] = {N};
	int inembed[1] = {SIZE_OF_PADDED_FFT_BLOCK * MAX_NO_BLOCKS};
	int onembed[1] = {BUFFER_LENGTH};
	/*This just checks that COALESCED_MEMORY_ALIGNMENT_BOUNDARY (in bytes) is divisable by 8 bytes. This is a reasonable 
	  assumption since this should be 128 byte under compute capability 2.0. Although this number may differ in future architectures it 
	  should always jump in powers of two. Some resources indicate that compute capability < 2.0 used 64 byte boundaries.

	  We need this to find out how many complex numbers there are between the beginning of each consecutive (padded) strides of elements 
	  we want to IFFT. If this is not a complex number you will have to change the cufft kernel invocation to use batches of size 1 and loop through
	  (in steps of N) when calling the kernel (taking the calculated padding into account). Something like this should do the trick:
	  
	  for (uint32_t block = 0; block < no_blocks_in_stride; ++block)
                        cufftSafeCall(cufftExecC2R(ifft_plan,
                                      (cufftComplex*)((int8_t*)d_ifft_input + block * (SIZE_OF_PADDED_FFT_BLOCK)),
                                      d_ifft_output + PAD + block*N));  
	  
	  You should then still be able to use streams to do multiple kernel invocations asynchronously.
	*/
	assert(SIZE_OF_PADDED_FFT_BLOCK % sizeof(cufftComplex) == 0);
	cufftSafeCall(cufftPlanMany(&ifft_plan,1,rank_sizes,
				    inembed,1,SIZE_OF_PADDED_FFT_BLOCK / sizeof(cufftComplex),
				    onembed,1,N,CUFFT_C2R,MAX_NO_BLOCKS));
	cudaSafeCall(cudaStreamCreate(&ifft_stream));
	cufftSafeCall(cufftSetStream(ifft_plan,ifft_stream));
	//alloc space for the ifft input vector on the device. The input vector should be BATCH * (N/2 + 1) (excluding any memory alignment padding) complex samples long
	printf("INIT: Setting up IFFT input buffer of  %d FFT blocks, each with %d non-redundant (including 1 discarded) samples. Padding to closest %d byte memory boundary (pad by %d bytes)\n",
	       MAX_NO_BLOCKS,NO_NON_REDUNDANT_SAMPLES_PER_FFT,COALESCED_MEMORY_ALIGNMENT_BOUNDARY,SIZE_OF_PAD_FOR_FFT_BLOCK);
        cudaSafeCall(cudaMalloc((void**)&d_cast_output, SIZE_OF_PADDED_FFT_BLOCK * MAX_NO_BLOCKS));
	cudaSafeCall(cudaMemset(d_cast_output,0, SIZE_OF_PADDED_FFT_BLOCK * MAX_NO_BLOCKS));
	//reserve memory for output (this should be BATCH * N real samples long)
	printf("INIT: Setting up PFB output vector of to store %d blocks of output, each with %d samples\n",MAX_NO_BLOCKS,N);
        cudaSafeCall(cudaMalloc((void**)&d_filtered_output,sizeof(uint8_t) * (MAX_NO_BLOCKS*N)));
	printf("---------------------------------------------------------\n");
	cudaSafeCall(cudaStreamCreate(&ifft_mov_output_cpy_stream));
}
/**
 Deallocates any memory associated with the inverse pfb process from the device.
 @precondition device should have been initialized before this method is called
*/
void releaseDevice(){
	printf("\033[0;31mAll done, releasing device\033[0m\n---------------------------------------------------------\n");
	cudaSafeCall(cudaFree(d_taps));
	cudaSafeCall(cudaFree(d_ifft_output));
	cudaSafeCall(cudaFree(d_cast_input));
	cudaSafeCall(cudaFree(d_cast_output));
        cudaSafeCall(cudaFree(d_filtered_output));
	cufftSafeCall(cufftDestroy(ifft_plan));
	cudaSafeCall(cudaStreamDestroy(ifft_stream));
	cudaSafeCall(cudaStreamDestroy(ifft_mov_output_cpy_stream));
	cudaDeviceReset(); //leave the device in a safe state
	printf("DEINIT: Device safely released\n---------------------------------------------------------\n");
}

/**
This method computes the inverse polyphase filter bank (pfb) operation (Weighted Window Overlap Add method) on a subset ("stride") 
of the output of a previous forward pfb operation.

The outline of the process is as follows:
 1. Perform N-element IFFTs on all the blocks in this stride. The output has an initial padding of P * N blocks.
 2. Filter the IFFTed samples (N * no_blocks_in_stride) with the inverse filter. This filtering operation starts
 at the beginning of the IFFT_output array and stops P*N samples short of the last index in the IFFT_output array. This
 is due to the fact that the filter looks ahead to compute samples at its current position.
 3. Move the last P*N IFFT samples to the start of the IFFT_output_array to maintain the state of the inverse pfb operation
 for processing the next stride of pfb output.

NOTE: the last point implies that this method has state associated with it. It is critical to maintain a persistant IFFT 
output buffer on the CUDA device to ensure that we do not loose P*N samples between processing consecutive strides of pfb output.

We could have equivalently achieved step 3 using a ringbuffer. However this would have meant that we could not process
all the IFFTs in one batch (it would have been split up into two batches). Copying P*N samples from one memory location
to another ***ON THE DEVICE*** should be relatively quick (considering we do not have to do a memory copy over a relatively
slow PCI-e bus. This also helps us get rid of the indexing nightmares associated with maintaining a ring buffer.

The initial setup and tairdown costs associated with memory allocation + deallocation is mitigated through initializing the
device once off before processing starts and tairing down the memory allocations only after all processing stops.

@args input list of complex numbers as output by a pfb process (these are interleved IEEE 32-bit floating point numbers, of the
	form 'riririri...'). Note that although this output is produced by an N-point real FFT operation only the first N/2 + 1 samples
	are useful (by the Hermite-symmetric property of the real FFT). The input to this method should therefore be an integral
	number of N/2+1 point FFTs.
@args output_buffer preallocated buffer of size no_blocks_in_stride*N in which the inverse pfb output will be dumped
@args no_blocks_in_stride The integral number of FFT blocks passed to this method. no_blocks_in_stride should be less than or
	equal to LOOP_LENGTH/(N/2+1)
@precondition call initDevice BEFORE calling this method
*/
void processNextStride(const complex_int8 * input, int8_t * output_buffer, uint32_t no_blocks_in_stride){
	assert(no_blocks_in_stride <= LOOP_LENGTH/(FFT_SIZE)); //this is the maximum number of blocks we can sent to the GPU
	//copy everything in this stride into the device ifft input vector
	printf("Copying %d blocks of FFT data, each of length %d to the device\n", no_blocks_in_stride,FFT_SIZE);
	{        
		cudaSafeCall(cudaMemcpy(d_cast_input,input,sizeof(complex_int8) * FFT_SIZE * no_blocks_in_stride,cudaMemcpyHostToDevice));
	}
	printf("Casting FFT data from int8 samples to 32-bit floating point samples\n");
	{
		/*Split the threads over a 2-D grid. This is to get past indexing restrictions (especially on older GPUs).  We're 
		  catering for a minimum of Compute Capability 2.0 here, which can only run 65535 blocks per dimension.
                */
		dim3 threads_per_block(CASTING_THREADS_PER_BLOCK,1,1);
		uint32_t no_samples_to_cast = no_blocks_in_stride * FFT_SIZE; 
		uint32_t no_blocks = (uint32_t)ceil(no_samples_to_cast / (float) CASTING_THREADS_PER_BLOCK);
		uint32_t no_blocks_in_dim = (uint32_t)ceil(sqrt(no_blocks));
                dim3 no_blocks_per_grid(no_blocks_in_dim,no_blocks_in_dim,1);
                array_cast_complex_int8_to_cufftComplex<<<no_blocks_per_grid,threads_per_block,0>>>(d_cast_input,d_cast_output,
													no_samples_to_cast,no_blocks_in_dim,
													SIZE_OF_PADDED_FFT_BLOCK);

                cudaThreadSynchronize();
                cudaError code = cudaGetLastError();
                if (code != cudaSuccess){
                        fprintf (stderr,"Error while executing gpu casting operation -- %s\n", cudaGetErrorString(code));
                        exit(-1);
                }
	}
	printf("Executing batched IFFT on data and saving with offset %d\n",PAD);
	//ifft the data:
	{
		cufftSafeCall(cufftExecC2R(ifft_plan, d_cast_output,d_ifft_output + PAD)); 
		cudaSafeCall(cudaStreamSynchronize(ifft_stream));//wait for all the iffts to finish before moving on...
	}
	printf("Performing inverse filtering, %d threads per block for %d blocks\n",N,no_blocks_in_stride);
	//now do the inverse filtering
	{
		/*Split the threads over a 2-D grid. This is to get past indexing restrictions (especially on older GPUs).  We're 
                  catering for a minimum of Compute Capability 2.0 here, which can only run 65535 blocks per dimension.
		*/
		uint32_t no_samples_to_filter = N * no_blocks_in_stride;
		dim3 threads_per_block(N,1,1);
		uint32_t no_blocks_in_dim = (uint32_t)ceil(sqrt(no_blocks_in_stride));
		dim3 no_blocks_per_grid(no_blocks_in_dim,no_blocks_in_dim,1);
		ipfb<<<no_blocks_per_grid,threads_per_block,0>>>(d_ifft_output,d_filtered_output,d_taps,no_samples_to_filter,no_blocks_in_dim);
	
		cudaThreadSynchronize();
		cudaError code = cudaGetLastError();
		if (code != cudaSuccess){
			fprintf (stderr,"Error while executing inverse pfb -- %s\n", cudaGetErrorString(code)); 
			exit(-1);
		}
	}
	//copy N-sized chunks to the output array:
        printf("Finished inverse pfb on stride, copying %d blocks, each of %d elements from the device\n",no_blocks_in_stride,N);
        cudaSafeCall(cudaMemcpyAsync(output_buffer, d_filtered_output,sizeof(int8_t)*(no_blocks_in_stride*N),cudaMemcpyDeviceToHost,ifft_mov_output_cpy_stream)); //we can start copying the output while we're moving the iffts
	printf("Moving %d IFFT samples from position %d of the IFFT persistant buffer to index 0 of the buffer.\n",N * P,N * no_blocks_in_stride);
	//move the last PAD samples in the ifft output array to the front of the ifft output array for processing the next stride of elements:
	{
		dim3 threads_per_block(N,1,1);
		dim3 no_blocks(P,1,1);
		move_last_P_iffts_to_front<<<no_blocks,threads_per_block,0,ifft_mov_output_cpy_stream>>>(d_ifft_output, N * no_blocks_in_stride);
	}
	cudaSafeCall(cudaStreamSynchronize(ifft_mov_output_cpy_stream));
	cudaError code = cudaGetLastError();
                if (code != cudaSuccess){
                        fprintf (stderr,"Error while executing moving last N*P IFFTs -- %s\n", cudaGetErrorString(code));
                        exit(-1);
                }
	
}
/**
This kernel takes in a precopied complex 8-bit signed integer array and casts the array elements to complex 32-bit floating points,
preserving coalesced memory accesses for both the input and output.

This kernel should be invoked with num_blocks = [CASTING_THREADS_PER_BLOCK,1,1] and 
num_blocks_per_grid = [ceil(sqrt(no_blocks_of_size_CASTING_THREADS_PER_BLOCK)),ceil(sqrt(no_blocks_of_size_CASTING_THREADS_PER_BLOCK)),1]

If N is a multiple of the warpsize (usually 32 threads) we should achieve coalesced memory accesses due to the fact that we're copying
the output into strides each of length N (ready to perform an IFFT on).
*/
__global__ void array_cast_complex_int8_to_cufftComplex(complex_int8 * in, cufftComplex * out, 
							uint32_t total_no_samples_casted_by_this_kernel, uint32_t no_blocks_in_dim,
                                                        uint32_t size_of_padded_fft_block){
	/*This 2-D grid indexing scheme is used to ensure we can process more than 65535 blocks in total on older architectures
          as discussed earlier.
        */
	register uint32_t tI = (blockIdx.x + blockIdx.y*no_blocks_in_dim)*blockDim.x + threadIdx.x;
	if (tI < total_no_samples_casted_by_this_kernel){
		//cast in reverse to ensure no samples in the original set is overwritten in order to make this inplace-casting-compatible:
		//register uint32_t reversed_index = total_no_samples_casted_by_this_kernel - tI - 1;
		register uint32_t no_preceeding_fft_blocks = tI / FFT_SIZE;
		register uint32_t sample_no_in_fft_block = tI % FFT_SIZE;
		register complex_int8 inElem = in[tI]; //input should be coalesced
		register cufftComplex outElem = {(cufftReal) inElem.r,(cufftReal) inElem.i};
		//output should be coalesced if we align the output to the memory byte boundary for coalesced accesses:
		register cufftComplex * mem_aligned_out = (cufftComplex*)((int8_t*)out + 
							  no_preceeding_fft_blocks * size_of_padded_fft_block) + 
			 				  sample_no_in_fft_block; 
		mem_aligned_out[0] = outElem;
	}
}

/**
This kernel computes the filter bank operation of the inverse Polyphase Filter Bank (Weighted Window Overlap Add method). 

The kernel should be invoked with blockDim = [N,1,1] and 
numBlocks = [ceil(sqrt(number_of_blocks_in_stride)),ceil(sqrt(number_of_blocks_in_stride)),1].

It will perform the following filterbank algorithm:
for (l = 0; l < stride_length; l += N) in parallel
	for (n = 0; n < N; ++n) in parallel
		accum = x[l+n]h[N - n - 1]
		for (p = 1; p < P; ++p)
			accum += x[l+n+p*N]h[p*N + (N - n - 1)]
		y[l+n] = accum
		endfor
	endfor
endfor

Technically the prototype filter remains the same for the synthesis filter bank. Each subfilter, however has to be read in reverse. Furthermore
the commutator technically runs in reverse as well, meaning that we should flip the order subfilters are executed in the bank. But whether we accumulate
each y[n] in forward or reverse does not matter. It should be clear that if the 3rd loop is run backwards with the the initialization of the 
accumulation set to a position in the last subfilter the result should remain the same.

All global memory calls made (P tap reads + P input reads + 1 output write per thread) should be coalesced provided N is a multiple of the warpsize
(usually this is 32 threads).
*/
__global__ void ipfb(const cufftReal * input,  int8_t * output, const float * prototype_filter, uint32_t no_samples_to_filter, uint32_t no_blocks_in_dim){
	/*This grid indexing scheme is used to ensure we can process more than 65535 blocks in total on older architectures
	  as discussed earlier.
	*/
	register uint32_t lB = (blockIdx.x + blockIdx.y * no_blocks_in_dim) * blockDim.x;
	register uint32_t n = threadIdx.x;
	if (lB + n < no_samples_to_filter){
		register uint32_t filter_index = N - n - 1;
		register double accum = input[lB + n]*prototype_filter[filter_index]; //Fetching data from both the filter and the input should be coalesced
		#pragma unroll
		for (uint32_t p = 1; p < P; ++p)
			accum += input[lB + n + p*N]*prototype_filter[p*N + filter_index]; //Fetching data from both the filter and the input should be coalesced
		output[lB + n] = (int8_t)accum; //Output should be coalesced
	}
}

/**
This kernel moves the last P*N iffts to the front of the ifft array, so that we can maintain the state of our filterbank between successive strides. It is critical
to call this kernel after the ipfb kernel has fully completed to ensure that the next stride isn't missing P*N samples.

The kernel should be invoked with blockDim = N and numBlocks = P
*/
__global__ void move_last_P_iffts_to_front(cufftReal * ifft_array, uint32_t start_of_last_P_N_block){
	register uint32_t tI = blockIdx.x*blockDim.x + threadIdx.x;
	ifft_array[tI] = ifft_array[start_of_last_P_N_block + tI];
} 
