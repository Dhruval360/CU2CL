cl_int err;
void __cu2cl_Init_transpose_cu() {
    #ifdef WITH_ALTERA
    progLen = __cu2cl_LoadProgramSource("transpose_cu_cl.aocx", &progSrc);
    __cu2cl_Program_transpose_cu = clCreateProgramWithBinary(__cu2cl_Context, 1, &__cu2cl_Device, &progLen, (const unsigned char **)&progSrc, NULL, &err);
    #else
    progLen = __cu2cl_LoadProgramSource("transpose.cu-cl.cl", &progSrc);
    __cu2cl_Program_transpose_cu = clCreateProgramWithSource(__cu2cl_Context, 1, &progSrc, &progLen, &err);
    //printf("clCreateProgramWithSource for transpose.cu-cl.cl: %s\n", getErrorString(err));
    #endif
    free((void *) progSrc);
    err = clBuildProgram(__cu2cl_Program_transpose_cu, 1, &__cu2cl_Device, "-I . ", NULL, NULL);
    /*printf("clBuildProgram : %s\n", getErrorString(err)); //Uncomment this line to access the error string of the error code returned by clBuildProgram*/
    if(err != CL_SUCCESS){
        std::vector<char> buildLog;
        size_t logSize;
        err = clGetProgramBuildInfo(__cu2cl_Program_transpose_cu, __cu2cl_Device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        printf("clGetProgramBuildInfo : %s\n", getErrorString(err));
        buildLog.resize(logSize);
        clGetProgramBuildInfo(__cu2cl_Program_transpose_cu, __cu2cl_Device, CL_PROGRAM_BUILD_LOG, logSize, &buildLog[0], NULL);
        printf("%s\n", &buildLog[0]);
    }
    __cu2cl_Kernel_transpose_serial = clCreateKernel(__cu2cl_Program_transpose_cu, "transpose_serial", &err);
    /*printf("__cu2cl_Kernel_transpose_serial creation: %s
", getErrorString(err)); // Uncomment this line to get error string for the error code returned by clCreateKernel while creating the Kernel: transpose_serial*/
    __cu2cl_Kernel_transpose_parallel_per_row = clCreateKernel(__cu2cl_Program_transpose_cu, "transpose_parallel_per_row", &err);
    /*printf("__cu2cl_Kernel_transpose_parallel_per_row creation: %s
", getErrorString(err)); // Uncomment this line to get error string for the error code returned by clCreateKernel while creating the Kernel: transpose_parallel_per_row*/
    __cu2cl_Kernel_transpose_parallel_per_element = clCreateKernel(__cu2cl_Program_transpose_cu, "transpose_parallel_per_element", &err);
    /*printf("__cu2cl_Kernel_transpose_parallel_per_element creation: %s
", getErrorString(err)); // Uncomment this line to get error string for the error code returned by clCreateKernel while creating the Kernel: transpose_parallel_per_element*/
    __cu2cl_Kernel_transpose_parallel_per_element_tiled = clCreateKernel(__cu2cl_Program_transpose_cu, "transpose_parallel_per_element_tiled", &err);
    /*printf("__cu2cl_Kernel_transpose_parallel_per_element_tiled creation: %s
", getErrorString(err)); // Uncomment this line to get error string for the error code returned by clCreateKernel while creating the Kernel: transpose_parallel_per_element_tiled*/
    __cu2cl_Kernel_transpose_parallel_per_element_tiled16 = clCreateKernel(__cu2cl_Program_transpose_cu, "transpose_parallel_per_element_tiled16", &err);
    /*printf("__cu2cl_Kernel_transpose_parallel_per_element_tiled16 creation: %s
", getErrorString(err)); // Uncomment this line to get error string for the error code returned by clCreateKernel while creating the Kernel: transpose_parallel_per_element_tiled16*/
    __cu2cl_Kernel_transpose_parallel_per_element_tiled_padded = clCreateKernel(__cu2cl_Program_transpose_cu, "transpose_parallel_per_element_tiled_padded", &err);
    /*printf("__cu2cl_Kernel_transpose_parallel_per_element_tiled_padded creation: %s
", getErrorString(err)); // Uncomment this line to get error string for the error code returned by clCreateKernel while creating the Kernel: transpose_parallel_per_element_tiled_padded*/
    __cu2cl_Kernel_transpose_parallel_per_element_tiled_padded16 = clCreateKernel(__cu2cl_Program_transpose_cu, "transpose_parallel_per_element_tiled_padded16", &err);
    /*printf("__cu2cl_Kernel_transpose_parallel_per_element_tiled_padded16 creation: %s
", getErrorString(err)); // Uncomment this line to get error string for the error code returned by clCreateKernel while creating the Kernel: transpose_parallel_per_element_tiled_padded16*/
}

cl_kernel __cu2cl_Kernel_transpose_serial;
cl_kernel __cu2cl_Kernel_transpose_parallel_per_row;
cl_kernel __cu2cl_Kernel_transpose_parallel_per_element;
cl_kernel __cu2cl_Kernel_transpose_parallel_per_element_tiled;
cl_kernel __cu2cl_Kernel_transpose_parallel_per_element_tiled16;
cl_kernel __cu2cl_Kernel_transpose_parallel_per_element_tiled_padded;
cl_kernel __cu2cl_Kernel_transpose_parallel_per_element_tiled_padded16;
cl_program __cu2cl_Program_transpose_cu;
extern const char *progSrc;
extern size_t progLen;

extern cl_platform_id __cu2cl_Platform;
extern cl_device_id __cu2cl_Device;
extern cl_context __cu2cl_Context;
extern cl_command_queue __cu2cl_CommandQueue;

extern size_t globalWorkSize[3];
extern size_t localWorkSize[3];
void __cu2cl_Cleanup_transpose_cu() {
    clReleaseKernel(__cu2cl_Kernel_transpose_serial);
    clReleaseKernel(__cu2cl_Kernel_transpose_parallel_per_row);
    clReleaseKernel(__cu2cl_Kernel_transpose_parallel_per_element);
    clReleaseKernel(__cu2cl_Kernel_transpose_parallel_per_element_tiled);
    clReleaseKernel(__cu2cl_Kernel_transpose_parallel_per_element_tiled16);
    clReleaseKernel(__cu2cl_Kernel_transpose_parallel_per_element_tiled_padded);
    clReleaseKernel(__cu2cl_Kernel_transpose_parallel_per_element_tiled_padded16);
    clReleaseProgram(__cu2cl_Program_transpose_cu);
}
#include <vector>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include "cu2cl_util.h"





#include <stdio.h>

const int N = 1024;		// The matrix used will be of size (N,N)
const int K = 32;		// This is for the tile size (tile size is (K,K))

// To compare the outputs obtained from the GPU with that of the CPU
int compare_matrices(float* gpu, float* ref)
{
	for (int i = 0; i < N * N; i++)
		if (gpu[i] != ref[i])
			return 1;
	return 0;
}

// To fill a matrix with sequential numbers in the range 0..N-1
void fill_matrix(float* mat)
{
	for (int j = 0; j < N * N; j++)
		mat[j] = (float)j;
}

// CPU code for the serial execution of matrix transpose
void transpose_CPU(float in[], float out[])
{
	for (int j = 0; j < N; j++)
		for (int i = 0; i < N; i++)
			out[j + i * N] = in[i + j * N]; 
}

// Serial function to be launched on a single thread of the GPU


// One thread per row of output matrix (unrolling by N)


// One thread per element, in KxK threadblocks, thread (x,y) in grid writes element (i,j) of output matrix 


// One thread per element, in (tilesize)x(tilesize) threadblocks
// thread blocks read & write tiles, in coalesced fashion
// adjacent threads read adjacent input elements, write adjacent output elements


// One thread per element, in (tilesize)x(tilesize) threadblocks
// thread blocks read & write tiles, in coalesced fashion
// adjacent threads read adjacent input elements, write adjacent output elmts


// One thread per element, in KxK threadblocks
// thread blocks read & write tiles, in coalesced fashion
// shared memory array has been padded to avoid bank conflicts


// One thread per element, in 16x16 threadblocks
// thread blocks read & write tiles, in coalesced fashion
// shared memory array has been padded to avoid bank conflicts


int main()
{
__cu2cl_Init();

	int numbytes = N * N * sizeof(float);

	float *in = (float*)malloc(numbytes);
	float *out = (float*)malloc(numbytes);
	float *reference_output = (float*)malloc(numbytes);

	fill_matrix(in); 

	printf("The matrix used is of size (%d, %d)\n", N, N);	

	clock_t start, end;
	double cpu_time_used, gpu_time_used;
     
     	start = clock();
	transpose_CPU(in, reference_output); // This creates the reference matrix (the expected output)
	end = clock();
	cpu_time_used = ((double) (end - start) * 1000) / CLOCKS_PER_SEC;
	printf("transpose_CPU                                                : Finished in %lf ms\n", cpu_time_used);

	cl_mem d_in;
cl_mem d_out;

	d_in = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, numbytes, NULL, &err);
//printf("clCreateBuffer for device variable d_in is: %s\n", getErrorString(err));
	d_out = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, numbytes, NULL, &err);
//printf("clCreateBuffer for device variable d_out is: %s\n", getErrorString(err));
	err = clEnqueueWriteBuffer(__cu2cl_CommandQueue, d_in, CL_TRUE, 0, numbytes, in, 0, NULL, NULL);
//printf("Memory copy from host variable in to device variable d_in: %s\n", getErrorString(err));

	// Transpose with each thread taking care of one row in the matrix
     	start = clock();
	err = clSetKernelArg(__cu2cl_Kernel_transpose_parallel_per_row, 0, sizeof(cl_mem), &d_in);
/*printf("clSetKernelArg for argument 0 of kernel __cu2cl_Kernel_transpose_parallel_per_row is: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_transpose_parallel_per_row, 1, sizeof(cl_mem), &d_out);
/*printf("clSetKernelArg for argument 1 of kernel __cu2cl_Kernel_transpose_parallel_per_row is: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
localWorkSize[0] = N;
globalWorkSize[0] = (1)*localWorkSize[0];
err = clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_transpose_parallel_per_row, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
//printf("clEnqueueNDRangeKernel for the kernel __cu2cl_Kernel_transpose_parallel_per_row: %s\n", getErrorString(err));
	end = clock();
	gpu_time_used = ((double) (end - start) * 1000) / CLOCKS_PER_SEC;
	err = clEnqueueReadBuffer(__cu2cl_CommandQueue, d_out, CL_TRUE, 0, numbytes, out, 0, NULL, NULL);
//printf("Memory copy from device variable out to host variable d_out: %s\n", getErrorString(err));	
	printf("transpose_parallel_per_row                                   : Finished in %lf ms and the output obtained is: %s\n", gpu_time_used, compare_matrices(out, reference_output) ? "Incorrect" : "Correct");

	
     	start = clock();
	err = clSetKernelArg(__cu2cl_Kernel_transpose_serial, 0, sizeof(cl_mem), &d_in);
/*printf("clSetKernelArg for argument 0 of kernel __cu2cl_Kernel_transpose_serial is: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_transpose_serial, 1, sizeof(cl_mem), &d_out);
/*printf("clSetKernelArg for argument 1 of kernel __cu2cl_Kernel_transpose_serial is: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
localWorkSize[0] = 1;
globalWorkSize[0] = (1)*localWorkSize[0];
err = clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_transpose_serial, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
//printf("clEnqueueNDRangeKernel for the kernel __cu2cl_Kernel_transpose_serial: %s\n", getErrorString(err));
	end = clock();
	gpu_time_used = ((double) (end - start) * 1000) / CLOCKS_PER_SEC;

	err = clEnqueueReadBuffer(__cu2cl_CommandQueue, d_out, CL_TRUE, 0, numbytes, out, 0, NULL, NULL);
//printf("Memory copy from device variable out to host variable d_out: %s\n", getErrorString(err));
	
	printf("transpose_serial                                             : Finished in %lf ms and the output obtained is: %s\n", gpu_time_used, compare_matrices(out, reference_output) ? "Incorrect" : "Correct");



	size_t blocks[3] = {N / K, N / K, 1}; // blocks per grid
	size_t threads[3] = {K, K, 1};	// threads per block

	// Transpose with a thread for each element in the matrix
     	start = clock();
	err = clSetKernelArg(__cu2cl_Kernel_transpose_parallel_per_element, 0, sizeof(cl_mem), &d_in);
/*printf("clSetKernelArg for argument 0 of kernel __cu2cl_Kernel_transpose_parallel_per_element is: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_transpose_parallel_per_element, 1, sizeof(cl_mem), &d_out);
/*printf("clSetKernelArg for argument 1 of kernel __cu2cl_Kernel_transpose_parallel_per_element is: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
localWorkSize[0] = threads[0];
localWorkSize[1] = threads[1];
localWorkSize[2] = threads[2];
globalWorkSize[0] = blocks[0]*localWorkSize[0];
globalWorkSize[1] = blocks[1]*localWorkSize[1];
globalWorkSize[2] = blocks[2]*localWorkSize[2];
err = clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_transpose_parallel_per_element, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
//printf("clEnqueueNDRangeKernel for the kernel __cu2cl_Kernel_transpose_parallel_per_element: %s\n", getErrorString(err));
	end = clock();
	gpu_time_used = ((double) (end - start) * 1000) / CLOCKS_PER_SEC;
	err = clEnqueueReadBuffer(__cu2cl_CommandQueue, d_out, CL_TRUE, 0, numbytes, out, 0, NULL, NULL);
//printf("Memory copy from device variable out to host variable d_out: %s\n", getErrorString(err));
	printf("transpose_parallel_per_element                               : Finished in %lf ms and the output obtained is: %s\n", gpu_time_used, compare_matrices(out, reference_output) ? "Incorrect" : "Correct");


	// Tiled transpose with a thread for each element in the tile
     	start = clock();
	err = clSetKernelArg(__cu2cl_Kernel_transpose_parallel_per_element_tiled, 0, sizeof(cl_mem), &d_in);
/*printf("clSetKernelArg for argument 0 of kernel __cu2cl_Kernel_transpose_parallel_per_element_tiled is: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_transpose_parallel_per_element_tiled, 1, sizeof(cl_mem), &d_out);
/*printf("clSetKernelArg for argument 1 of kernel __cu2cl_Kernel_transpose_parallel_per_element_tiled is: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
localWorkSize[0] = threads[0];
localWorkSize[1] = threads[1];
localWorkSize[2] = threads[2];
globalWorkSize[0] = blocks[0]*localWorkSize[0];
globalWorkSize[1] = blocks[1]*localWorkSize[1];
globalWorkSize[2] = blocks[2]*localWorkSize[2];
err = clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_transpose_parallel_per_element_tiled, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
//printf("clEnqueueNDRangeKernel for the kernel __cu2cl_Kernel_transpose_parallel_per_element_tiled: %s\n", getErrorString(err));
	end = clock();
	gpu_time_used = ((double) (end - start) * 1000) / CLOCKS_PER_SEC;
	err = clEnqueueReadBuffer(__cu2cl_CommandQueue, d_out, CL_TRUE, 0, numbytes, out, 0, NULL, NULL);
//printf("Memory copy from device variable out to host variable d_out: %s\n", getErrorString(err));
	printf("transpose_parallel_per_element_tiled (32X32)                 : Finished in %lf ms and the output obtained is: %s\n", gpu_time_used, compare_matrices(out, reference_output) ? "Incorrect" : "Correct");



	size_t blocks16x16[3] = {N / 16, N / 16, 1}; // blocks per grid
	size_t threads16x16[3] = {16, 16, 1};	 // threads per block


	// Tiled transpose with each element of the tile given to a thread
     	start = clock();
	err = clSetKernelArg(__cu2cl_Kernel_transpose_parallel_per_element_tiled16, 0, sizeof(cl_mem), &d_in);
/*printf("clSetKernelArg for argument 0 of kernel __cu2cl_Kernel_transpose_parallel_per_element_tiled16 is: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_transpose_parallel_per_element_tiled16, 1, sizeof(cl_mem), &d_out);
/*printf("clSetKernelArg for argument 1 of kernel __cu2cl_Kernel_transpose_parallel_per_element_tiled16 is: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
localWorkSize[0] = threads16x16[0];
localWorkSize[1] = threads16x16[1];
localWorkSize[2] = threads16x16[2];
globalWorkSize[0] = blocks16x16[0]*localWorkSize[0];
globalWorkSize[1] = blocks16x16[1]*localWorkSize[1];
globalWorkSize[2] = blocks16x16[2]*localWorkSize[2];
err = clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_transpose_parallel_per_element_tiled16, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
//printf("clEnqueueNDRangeKernel for the kernel __cu2cl_Kernel_transpose_parallel_per_element_tiled16: %s\n", getErrorString(err));
	end = clock();
	gpu_time_used = ((double) (end - start) * 1000) / CLOCKS_PER_SEC;
	err = clEnqueueReadBuffer(__cu2cl_CommandQueue, d_out, CL_TRUE, 0, numbytes, out, 0, NULL, NULL);
//printf("Memory copy from device variable out to host variable d_out: %s\n", getErrorString(err));
	printf("transpose_parallel_per_element_tiled (16X16)                 : Finished in %lf ms and the output obtained is: %s\n", gpu_time_used, compare_matrices(out, reference_output) ? "Incorrect" : "Correct");


	// Tiled transpose with padding
     	start = clock();
	err = clSetKernelArg(__cu2cl_Kernel_transpose_parallel_per_element_tiled_padded16, 0, sizeof(cl_mem), &d_in);
/*printf("clSetKernelArg for argument 0 of kernel __cu2cl_Kernel_transpose_parallel_per_element_tiled_padded16 is: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_transpose_parallel_per_element_tiled_padded16, 1, sizeof(cl_mem), &d_out);
/*printf("clSetKernelArg for argument 1 of kernel __cu2cl_Kernel_transpose_parallel_per_element_tiled_padded16 is: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
localWorkSize[0] = threads16x16[0];
localWorkSize[1] = threads16x16[1];
localWorkSize[2] = threads16x16[2];
globalWorkSize[0] = blocks16x16[0]*localWorkSize[0];
globalWorkSize[1] = blocks16x16[1]*localWorkSize[1];
globalWorkSize[2] = blocks16x16[2]*localWorkSize[2];
err = clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_transpose_parallel_per_element_tiled_padded16, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
//printf("clEnqueueNDRangeKernel for the kernel __cu2cl_Kernel_transpose_parallel_per_element_tiled_padded16: %s\n", getErrorString(err));
	end = clock();
	gpu_time_used = ((double) (end - start) * 1000) / CLOCKS_PER_SEC;
	err = clEnqueueReadBuffer(__cu2cl_CommandQueue, d_out, CL_TRUE, 0, numbytes, out, 0, NULL, NULL);
//printf("Memory copy from device variable out to host variable d_out: %s\n", getErrorString(err));	
	printf("transpose_parallel_per_element_tiled_padded (16X16)          : Finished in %lf ms and the output obtained is: %s\n", gpu_time_used, compare_matrices(out, reference_output) ? "Incorrect" : "Correct");


	clReleaseMemObject(d_in);
	clReleaseMemObject(d_out);
__cu2cl_Cleanup();
}

