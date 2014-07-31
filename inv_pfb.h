#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <assert.h>
#include "math.h"

#ifndef INV_PFB_H
#define INV_PFB_H

/****************************************************************************************************************
 constants
*****************************************************************************************************************/
//Number of FFT samples (safe to tweak, BUT: this number should be divisable by COALESCED_MEMORY_ALIGNMENT_BOUNDARY, otherwise there will be serious performance penalties!):
const uint16_t N = 1024;
const uint16_t P = 8; //Number of Filterbanks (safe to tweak)
/*size of each input FFT (non-redundant samples). Technically this should be N/2 + 1 by the Hermite-symmetry of
Real FFTs. However the KAT-7 B-engine discards the last element. We should pad each block of FFT data by 1 before
invoking any IFFT
*/
const uint32_t FFT_SIZE = N/2;
const uint32_t NO_NON_REDUNDANT_SAMPLES_PER_FFT = N/2 + 1;
const uint32_t COALESCED_MEMORY_ALIGNMENT_BOUNDARY = 128; //128 bytes is normally the memory alignment boundary that will result in coalesced reads
const uint32_t SIZE_OF_NON_REDUNDANT_SAMPLES_OF_FFT = NO_NON_REDUNDANT_SAMPLES_PER_FFT * sizeof(cufftComplex); //in bytes
const uint32_t SIZE_OF_PAD_FOR_FFT_BLOCK = (uint32_t)ceil(SIZE_OF_NON_REDUNDANT_SAMPLES_OF_FFT / (float) COALESCED_MEMORY_ALIGNMENT_BOUNDARY) * 
	  				    COALESCED_MEMORY_ALIGNMENT_BOUNDARY - SIZE_OF_NON_REDUNDANT_SAMPLES_OF_FFT; //size of coalesced fft padding in bytes
const uint32_t SIZE_OF_PADDED_FFT_BLOCK = SIZE_OF_NON_REDUNDANT_SAMPLES_OF_FFT + SIZE_OF_PAD_FOR_FFT_BLOCK;
const uint32_t WINDOW_LENGTH = N*P;
const uint32_t PAD = N*P;
/*Size of chunk to send off to the GPU. Safe to tweak, **BUT**: this number must be divisable by FFT_SIZE (we should 
send an integral number of FFTs to the GPU):
*/
const uint32_t LOOP_LENGTH = 32000 * FFT_SIZE;
const uint32_t BUFFER_LENGTH = LOOP_LENGTH / FFT_SIZE * N + PAD; //Number of elements in the persistant ifft output buffer
const uint32_t MAX_NO_BLOCKS = LOOP_LENGTH / FFT_SIZE;
//to accomodate the discarded sample for every block we must pad the number of blocks:
const uint32_t PADDING_NEEDED_FOR_IFFT_INPUT = MAX_NO_BLOCKS;
/*Block size of the int8 to cufftReal casting kernel. 256 threads per block seem to be a magic number in CUDA that works 
well accross different generations of cards:
*/
const uint16_t CASTING_THREADS_PER_BLOCK = 256;

/****************************************************************************************************************
 necessary Abstract Data Types
*****************************************************************************************************************/
typedef struct {
        int8_t r;
        int8_t i;
} complex_int8;

/****************************************************************************************************************
 CUDA error handling macros
****************************************************************************************************************/
#define cudaSafeCall(value) {                                                                                   \
        cudaError_t _m_cudaStat = value;                                                                                \
        if (_m_cudaStat != cudaSuccess) {                                                                               \
                fprintf(stderr, "Error %s at line %d in file %s\n",                                     \
                                cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);           \
                exit(1);                                                                                                                        \
        } }

inline void __cufftSafeCall( uint32_t err, const char *file, const int line ){
        if ( CUFFT_SUCCESS != err ){
                fprintf( stderr, "cufftSafeCall() failed at %s:%i\n", file, line);
                exit( -1 );
        }
        return;
}
#define cufftSafeCall(err)  __cufftSafeCall(err, __FILE__, __LINE__)

/****************************************************************************************************************
 Forward declarations
*****************************************************************************************************************/
void initDevice(const float * taps);
void releaseDevice();
void processNextStride(const complex_int8 * input, int8_t * output_buffer, uint32_t no_blocks_in_stride = LOOP_LENGTH/(N/2+1));
#endif //ifndef INV_PFB_H
