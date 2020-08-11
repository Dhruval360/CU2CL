#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include "cu2cl_util.h"




cl_kernel __cu2cl_Kernel_kernel;
cl_program __cu2cl_Program_pitch_cu;
cl_int err;
cl_uint cu2cl_align = 0;
extern const char *progSrc;
extern size_t progLen;

extern cl_platform_id __cu2cl_Platform;
extern cl_device_id __cu2cl_Device;
extern cl_context __cu2cl_Context;
extern cl_command_queue __cu2cl_CommandQueue;

extern size_t globalWorkSize[3];
extern size_t localWorkSize[3];
void __cu2cl_Cleanup_pitch_cu() {
    clReleaseKernel(__cu2cl_Kernel_kernel);
    clReleaseProgram(__cu2cl_Program_pitch_cu);
}
void __cu2cl_Init_pitch_cu() {
    #ifdef WITH_ALTERA
    progLen = __cu2cl_LoadProgramSource("pitch_cu_cl.aocx", &progSrc);
    __cu2cl_Program_pitch_cu = clCreateProgramWithBinary(__cu2cl_Context, 1, &__cu2cl_Device, &progLen, (const unsigned char **)&progSrc, NULL, &err);
    //printf("clCreateProgramWithBinary for pitch.cu-cl.cl: %s\n", getErrorString(err));
    #else
    progLen = __cu2cl_LoadProgramSource("pitch.cu-cl.cl", &progSrc);
    __cu2cl_Program_pitch_cu = clCreateProgramWithSource(__cu2cl_Context, 1, &progSrc, &progLen, &err);
    //printf("clCreateProgramWithSource for pitch.cu-cl.cl: %s\n", getErrorString(err));
    #endif
    free((void *) progSrc);
    err = clBuildProgram(__cu2cl_Program_pitch_cu, 1, &__cu2cl_Device, "-I . ", NULL, NULL);
    /*printf("clBuildProgram : %s\n", getErrorString(err)); //Uncomment this line to access the error string of the error code returned by clBuildProgram*/
    if(err != CL_SUCCESS){
        std::vector<char> buildLog;
        size_t logSize;
        err = clGetProgramBuildInfo(__cu2cl_Program_pitch_cu, __cu2cl_Device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        printf("clGetProgramBuildInfo : %s\n", getErrorString(err));
        buildLog.resize(logSize);
        clGetProgramBuildInfo(__cu2cl_Program_pitch_cu, __cu2cl_Device, CL_PROGRAM_BUILD_LOG, logSize, &buildLog[0], NULL);
        printf("%s\n", &buildLog[0]);
    }
    __cu2cl_Kernel_kernel = clCreateKernel(__cu2cl_Program_pitch_cu, "kernel", &err);
    /*printf("__cu2cl_Kernel_kernel creation: %s\n", getErrorString(err)); // Uncomment this line to get error string for the error code returned by clCreateKernel while creating the Kernel: kernel*/
clGetDeviceInfo(__cu2cl_Device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(cl_uint), &cu2cl_align, 0);
cu2cl_align /= 8;
}

#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#define N 760 // side of matrix containing data
#define PDIM 768 // padded dimensions
#define TPB 128 //threads per block
#define DIV 6
 
//load element from da to db to verify correct memcopy

 
void verify(float * A, float * B, int size);
void init(float * array, int size);
 
int main(int argc, char * argv[])
{
__cu2cl_Init();

 cl_mem dA;
float *A;
float *B;
cl_mem dB;
 A = (float *)malloc(sizeof(float)*N*N);
 B = (float *)malloc(sizeof(float)*N*N);
 
 init(A,N*N);
 size_t pitch;
 pitch = (size_t)(cu2cl_ceil(sizeof(float)*N, cu2cl_align))*cu2cl_align;
dA = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, pitch*N, NULL, &err);
//printf("clCreateBuffer for device variable dA: %s\n", getErrorString(err));
 dB = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(float)*N*N, NULL, &err);
//printf("clCreateBuffer for device variable dB: %s\n", getErrorString(err));
 
//copy memory from unpadded array A of 760 by 760 dimensions
//to more efficient dimensions of 768 by 768 on the device
 size_t cu2cl_temp_origin0[3] = {0,0,0};
size_t cu2cl_temp_region0[3] = {sizeof(float)*N,N,1};
err = clEnqueueWriteBufferRect(__cu2cl_CommandQueue, dA, CL_TRUE,cu2cl_temp_origin0,cu2cl_temp_origin0,cu2cl_temp_region0,pitch, 0, sizeof(float)*N, 0, A, 0, NULL, NULL); 
 //printf("Memory copy from host variable A to device variable dA: %s\n", getErrorString(err));
 int threadsperblock = TPB;
 int blockspergrid = PDIM*PDIM/threadsperblock;
 err = clSetKernelArg(__cu2cl_Kernel_kernel, 0, sizeof(cl_mem), &dA);
/*printf("clSetKernelArg for argument 0 of kernel __cu2cl_Kernel_kernel: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_kernel, 1, sizeof(cl_mem), &dB);
/*printf("clSetKernelArg for argument 1 of kernel __cu2cl_Kernel_kernel: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
localWorkSize[0] = threadsperblock;
globalWorkSize[0] = (blockspergrid)*localWorkSize[0];
err = clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
//printf("clEnqueueNDRangeKernel for the kernel __cu2cl_Kernel_kernel: %s\n", getErrorString(err));
 err = clEnqueueReadBuffer(__cu2cl_CommandQueue, dB, CL_TRUE, 0, sizeof(float)*N*N, B, 0, NULL, NULL);
//printf("Memory copy from device variable B to host variable dB: %s\n", getErrorString(err));
 //cudaMemcpy2D(B,N,dB,N,N,N,cudaMemcpyDeviceToHost);
 verify(A,B,N*N);
 
 free(A);
 free(B);
 clReleaseMemObject(dA);
 clReleaseMemObject(dB);
__cu2cl_Cleanup();
}
 

void init(float * array, int size){
 for (int i = 0; i < size; i++){
 array[i] = i;
 }
}

void verify(float * A, float * B, int size){
 for (int i = 0; i < size; i++) {
 assert(A[i]==B[i]);
 }
 printf("Correct!\n");
}

