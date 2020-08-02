#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include "cu2cl_util.h"




cl_kernel __cu2cl_Kernel_dotProd;
cl_program __cu2cl_Program_dotprod_cu;
cl_int err;
extern const char *progSrc;
extern size_t progLen;

extern cl_platform_id __cu2cl_Platform;
extern cl_device_id __cu2cl_Device;
extern cl_context __cu2cl_Context;
extern cl_command_queue __cu2cl_CommandQueue;

extern size_t globalWorkSize[3];
extern size_t localWorkSize[3];
void __cu2cl_Cleanup_dotprod_cu() {
    clReleaseKernel(__cu2cl_Kernel_dotProd);
    clReleaseProgram(__cu2cl_Program_dotprod_cu);
}
void __cu2cl_Init_dotprod_cu() {
    #ifdef WITH_ALTERA
    progLen = __cu2cl_LoadProgramSource("dotprod_cu_cl.aocx", &progSrc);
    __cu2cl_Program_dotprod_cu = clCreateProgramWithBinary(__cu2cl_Context, 1, &__cu2cl_Device, &progLen, (const unsigned char **)&progSrc, NULL, &err);
    //printf("clCreateProgramWithBinary for dotprod.cu-cl.cl: %s\n", getErrorString(err));
    #else
    progLen = __cu2cl_LoadProgramSource("dotprod.cu-cl.cl", &progSrc);
    __cu2cl_Program_dotprod_cu = clCreateProgramWithSource(__cu2cl_Context, 1, &progSrc, &progLen, &err);
    //printf("clCreateProgramWithSource for dotprod.cu-cl.cl: %s\n", getErrorString(err));
    #endif
    free((void *) progSrc);
    err = clBuildProgram(__cu2cl_Program_dotprod_cu, 1, &__cu2cl_Device, "-I . ", NULL, NULL);
    /*printf("clBuildProgram : %s\n", getErrorString(err)); //Uncomment this line to access the error string of the error code returned by clBuildProgram*/
    if(err != CL_SUCCESS){
        std::vector<char> buildLog;
        size_t logSize;
        err = clGetProgramBuildInfo(__cu2cl_Program_dotprod_cu, __cu2cl_Device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        printf("clGetProgramBuildInfo : %s\n", getErrorString(err));
        buildLog.resize(logSize);
        clGetProgramBuildInfo(__cu2cl_Program_dotprod_cu, __cu2cl_Device, CL_PROGRAM_BUILD_LOG, logSize, &buildLog[0], NULL);
        printf("%s\n", &buildLog[0]);
    }
    __cu2cl_Kernel_dotProd = clCreateKernel(__cu2cl_Program_dotprod_cu, "dotProd", &err);
    /*printf("__cu2cl_Kernel_dotProd creation: %s\n", getErrorString(err)); // Uncomment this line to get error string for the error code returned by clCreateKernel while creating the Kernel: dotProd*/
}

#include<stdio.h>
#include<iostream>


#define imin(a,b) (a<b?a:b)
#define  N  33 * 1024
#define threadsPerBlock  256
#define blocksPerGrid  imin(32, (N+threadsPerBlock-1) / threadsPerBlock)

//const int N = 33 * 1024;
//const int threadsPerBlock = 256;
//const int blocksPerGrid = imin(32, (N+threadsPerBlock-1) / threadsPerBlock);




int main (void) {
__cu2cl_Init();

	float *a, *b, c, *partial_c;
	cl_mem dev_a;
cl_mem dev_b;
cl_mem dev_partial_c;
	
	// allocate memory on the cpu side
	a = (float*)malloc(N*sizeof(float));
	b = (float*)malloc(N*sizeof(float));
	partial_c = (float*)malloc(blocksPerGrid*sizeof(float));
	
	// allocate the memory on the gpu
	*(void**)&dev_a = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, N*sizeof(float), NULL, &err);
//printf("clCreateBuffer for device variable *(void**)&dev_a: %s\n", getErrorString(err));
	*(void**)&dev_b = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, N*sizeof(float), NULL, &err);
//printf("clCreateBuffer for device variable *(void**)&dev_b: %s\n", getErrorString(err));
	*(void**)&dev_partial_c = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, blocksPerGrid*sizeof(float), NULL, &err);
//printf("clCreateBuffer for device variable *(void**)&dev_partial_c: %s\n", getErrorString(err));
	
	// fill in the host mempory with data
	for(int i=0; i<N; i++) {
		a[i] = i;
		b[i] = i*2;
	}
	
	
	// copy the arrays 'a' and 'b' to the gpu
	err = clEnqueueWriteBuffer(__cu2cl_CommandQueue, dev_a, CL_TRUE, 0, N*sizeof(float), a, 0, NULL, NULL);
//printf("Memory copy from host variable a to device variable dev_a: %s\n", getErrorString(err));
	err = clEnqueueWriteBuffer(__cu2cl_CommandQueue, dev_b, CL_TRUE, 0, N*sizeof(float), b, 0, NULL, NULL);
//printf("Memory copy from host variable b to device variable dev_b: %s\n", getErrorString(err));
	
	err = clSetKernelArg(__cu2cl_Kernel_dotProd, 0, sizeof(cl_mem), &dev_a);
/*printf("clSetKernelArg for argument 0 of kernel __cu2cl_Kernel_dotProd: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_dotProd, 1, sizeof(cl_mem), &dev_b);
/*printf("clSetKernelArg for argument 1 of kernel __cu2cl_Kernel_dotProd: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_dotProd, 2, sizeof(cl_mem), &dev_partial_c);
/*printf("clSetKernelArg for argument 2 of kernel __cu2cl_Kernel_dotProd: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
localWorkSize[0] = threadsPerBlock;
globalWorkSize[0] = (blocksPerGrid)*localWorkSize[0];
err = clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_dotProd, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
//printf("clEnqueueNDRangeKernel for the kernel __cu2cl_Kernel_dotProd: %s\n", getErrorString(err));
	
	// copy the array 'c' back from the gpu to the cpu
	err = clEnqueueReadBuffer(__cu2cl_CommandQueue, dev_partial_c, CL_TRUE, 0, blocksPerGrid*sizeof(float), partial_c, 0, NULL, NULL);
//printf("Memory copy from device variable partial_c to host variable dev_partial_c: %s\n", getErrorString(err));
	
	// finish up on the cpu side
	c = 0;
	for(int i=0; i<blocksPerGrid; i++) {
		c += partial_c[i];
	}
	
	#define sum_squares(x) (x*(x+1)*(2*x+1)/6)
	printf("Does GPU value %.6g = %.6g?\n", c, 2*sum_squares((float)(N-1)));
	
	// free memory on the gpu side
	clReleaseMemObject(dev_a);
	clReleaseMemObject(dev_b);
	clReleaseMemObject(dev_partial_c);
	
	// free memory on the cpu side
	free(a);
	free(b);
	free(partial_c);
__cu2cl_Cleanup();
}
