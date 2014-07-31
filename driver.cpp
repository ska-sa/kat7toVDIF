#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <sys/stat.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "inv_pfb.h"

uint32_t readDataFromDisk(const char * filename, uint32_t element_size, uint32_t length, void * buffer, uint32_t seek = 0);
uint32_t writeDataToDisk(const char * filename, uint32_t element_size, uint32_t length, const void * buffer, bool blankfile = false);
void transpose(complex_int8* array, int row, int col);
void transposeData(complex_int8 * pfb_data,uint32_t length);
/**Reads data from disk
 @args filename
 @args element_size size of the individual elements in bytes
 @args length number of elents to read
 @args buffer pre-allocated buffer of length 'element_size*length' in bytes
 @args seek number of bytes to seek through the file before reading. Default 0.
 @returns 0 if file could not be opened or seek failed, else number of elements of size "element_size" read
*/

uint32_t readDataFromDisk(const char * filename, uint32_t element_size, uint32_t length, void * buffer, uint32_t seek){
        FILE * hnd = fopen(filename,"r");
        uint32_t elemsRead = 0;
//	elemsRead = fread(buffer,element_size,length,hnd);
        if (hnd != NULL){
		if (seek != 0){
                        if (fseek(hnd,seek,SEEK_SET) != 0)
                                return 0;
                }
                elemsRead = fread(buffer,element_size,length,hnd);
		fclose(hnd);
        }
	return elemsRead;
}
/***
 *calls transpose method to  transpose the data in pfb_data in chuncks of n_chans*128
 *@args pfb_data contants data read from a binary file that needs to be transposed
 *@args length is the length of the file
 */
void transposeData(complex_int8 * pfb_data,uint32_t length){
	int i;
	int n_chans=512;
	for(i=0;i<length;i+=n_chans*128)
	{
		//I read a specific amount of data chunks
		transpose((pfb_data+i),n_chans,128);
	}
}

/**Writes data to disk (to blank file or appending)
 @args filename
 @args element_size size of the individual elements in bytes
 @args length number of elents to write
 @args buffer pre-allocated buffer of length 'element_size*length' in bytes
 @args blank_file if true a file will be created or an existing file overwritten, otherwise append mode requires the file to exist prior to opening.
 @returns 0 if file could not be opened, else number of elements of size "element_size" successfully written to disk
*/
uint32_t writeDataToDisk(const char * filename, uint32_t element_size, uint32_t length, const void * buffer, bool blank_file){
	FILE * hnd = fopen(filename, blank_file ? "w" : "a");
	uint32_t elemsWrote = 0;
        if (hnd != NULL){
                elemsWrote = fwrite(buffer,element_size,length,hnd);
                fclose(hnd);
        }
	return elemsWrote;
}
/***
 *Performs transpose to 2D array
 *@args array is 1D array which is treated as 2D. It will be transposed
 *@args row is length of the row of 2D array
 *@args col is length of the column of 2D array
 **/


void transpose(complex_int8 * array, int row, int col){ // the from is where i will start transposing
         complex_int8* arrayT= (complex_int8*)malloc(sizeof(complex_int8)*row*col);
         int x, y;
         for(x=0; x<row;x++){
                for(y=0;y<col;y++){
                         arrayT[y*row+x]=array[x*col+y];
                  }
        }
	int i;
	for(i=0;i<row*col;i++){
		array[i]= arrayT[i];
	}
}

int main ( int argc, char ** argv ){
        char * taps_filename;
        char * pfb_output_filename;
        char * output_filename;
        uint32_t num_samples;
        if (argc != 4){
                fprintf(stderr, "expected arguements: 'prototype_filter_file' 'ipfb_input_file' 'output_file'\n");
                return 1;
        }
        taps_filename = argv[1];
	pfb_output_filename = argv[2];
        output_filename = argv[3];
	
	struct stat st;
	stat(pfb_output_filename, &st);
	uint32_t input_file_size = st.st_size; //in bytes
	assert(input_file_size % sizeof(complex_int8) == 0); //ensure we're at least dealing with a file that can be interpreted as a file of complex 8-bit ints
        num_samples = input_file_size / sizeof(complex_int8);
	assert(num_samples % FFT_SIZE == 0); //ensure we're only processing an integral number of non-redundant real FFT samples
        printf("Performing operation on %d blocks of complex non-redundant FFT samples (total of %d complex elements) from '%s'\n",num_samples/FFT_SIZE,num_samples,pfb_output_filename);

        //Read in taps file (prototype filter)  
        float * taps = (float*) malloc(sizeof(float)*WINDOW_LENGTH);
        if (readDataFromDisk(taps_filename,sizeof(float),WINDOW_LENGTH,taps) != WINDOW_LENGTH){
                fprintf(stderr, "Prototype filter has to be %d long\n",WINDOW_LENGTH);
                return 1;
        }
	
        //Setup the device and copy the taps
        initDevice(taps);
	
	//Read in input file (output of earlier pfb process, so these will be complex numbers
        complex_int8 * pfb_data= (complex_int8*)malloc(sizeof(complex_int8)*num_samples);
	
        cudaSafeCall(cudaMallocHost((void **)&pfb_data,sizeof(complex_int8)*num_samples)); //critical optimization: pin host input memory to disable paging and speed up transfers to device
        if (readDataFromDisk(pfb_output_filename,sizeof(complex_int8),num_samples,pfb_data,0) != num_samples){
                fprintf(stderr, "input pfb data does not contain %d non-redundant samples\n", num_samples);
                return 1;
        }
	transposeData(pfb_data,num_samples);

	//added transpose vusi
	


	uint32_t ipfb_output_size = (num_samples / FFT_SIZE) * N; //The iPFB process produces blocks of size N
        int8_t * output;
	cudaSafeCall(cudaMallocHost((void **)&output,sizeof(int8_t)*ipfb_output_size)); //critical optimization: pin host output memory to disable paging and speed up transfers from device

        //do some processing
	printf("\033[0;31mProcessing starting\033[0m\n---------------------------------------------------------\n%d non-redundant samples in the input data\n",num_samples);
	uint32_t num_loops = (uint32_t)ceil(num_samples / float(LOOP_LENGTH));
	printf("Only %d samples can be processed by the GPU at a time. %d iterations needed to complete the inverse pfb\n",LOOP_LENGTH,num_loops);
	printf("---------------------------------------------------------\n");
	float total_time_seconds = 0;
	cudaEvent_t tic,toc;
	cudaSafeCall(cudaEventCreate(&tic));
	cudaSafeCall(cudaEventCreate(&toc));
	for (uint32_t i = 0; i < num_loops; ++i){
		uint32_t num_blocks_to_process = ((uint32_t)fmin(num_samples - i * LOOP_LENGTH,LOOP_LENGTH)) / FFT_SIZE; 
		printf("\033[0;32mExecuting loop %d/%d. Processing %d blocks of size %d complex numbers\033[0m\n\n",i+1,num_loops,num_blocks_to_process,FFT_SIZE);
		cudaSafeCall(cudaEventRecord(tic));
	        processNextStride(pfb_data + (i * LOOP_LENGTH), output, num_blocks_to_process);
		cudaSafeCall(cudaEventRecord(toc));
		cudaSafeCall(cudaEventSynchronize(toc));
		float loop_exec_time = 0;
		cudaSafeCall(cudaEventElapsedTime(&loop_exec_time, tic, toc));
		total_time_seconds += loop_exec_time / 1000;
        	writeDataToDisk(output_filename,sizeof(int8_t),num_blocks_to_process*N,output,i == 0);
	}
	cudaSafeCall(cudaEventDestroy(tic));
	cudaSafeCall(cudaEventDestroy(toc));
	printf("---------------------------------------------------------\n");
	printf("\033[44;33mTOTAL PROCESSING TIME: %f seconds (@ %f Gbps)\033[0m\n",total_time_seconds,
		(num_samples * sizeof(complex_int8) / 1024.0 / 1024.0 / 1024.0 * 8.0) / total_time_seconds);
	printf("---------------------------------------------------------\n");

        //free any hostside memory
	free(taps);
        cudaSafeCall(cudaFreeHost(output));
        cudaSafeCall(cudaFreeHost(pfb_data));

	//safely release the device, freeing up all allocated memory
        releaseDevice();

        printf("\033[1;33mApplication Terminated Normally\033[0m\n---------------------------------------------------------\n");
}
