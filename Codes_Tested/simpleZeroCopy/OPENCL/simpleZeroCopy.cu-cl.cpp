#include <vector>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include "cu2cl_util.h"




cl_int err;
void __cu2cl_Init_simpleZeroCopy_cu() {
    #ifdef WITH_ALTERA
    progLen = __cu2cl_LoadProgramSource("simpleZeroCopy_cu_cl.aocx", &progSrc);
    __cu2cl_Program_simpleZeroCopy_cu = clCreateProgramWithBinary(__cu2cl_Context, 1, &__cu2cl_Device, &progLen, (const unsigned char **)&progSrc, NULL, &err);
    #else
    progLen = __cu2cl_LoadProgramSource("simpleZeroCopy.cu-cl.cl", &progSrc);
    __cu2cl_Program_simpleZeroCopy_cu = clCreateProgramWithSource(__cu2cl_Context, 1, &progSrc, &progLen, &err);
    //printf("clCreateProgramWithSource for simpleZeroCopy.cu-cl.cl: %s\n", getErrorString(err));
    #endif
    free((void *) progSrc);
    err = clBuildProgram(__cu2cl_Program_simpleZeroCopy_cu, 1, &__cu2cl_Device, "-I . ", NULL, NULL);
    /*printf("clBuildProgram : %s\n", getErrorString(err)); //Uncomment this line to access the error string of the error code returned by clBuildProgram*/
    if(err != CL_SUCCESS){
        std::vector<char> buildLog;
        size_t logSize;
        err = clGetProgramBuildInfo(__cu2cl_Program_simpleZeroCopy_cu, __cu2cl_Device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        printf("clGetProgramBuildInfo : %s\n", getErrorString(err));
        buildLog.resize(logSize);
        clGetProgramBuildInfo(__cu2cl_Program_simpleZeroCopy_cu, __cu2cl_Device, CL_PROGRAM_BUILD_LOG, logSize, &buildLog[0], NULL);
        printf("%s\n", &buildLog[0]);
    }
    __cu2cl_Kernel_vectorAddGPU = clCreateKernel(__cu2cl_Program_simpleZeroCopy_cu, "vectorAddGPU", &err);
    /*printf("__cu2cl_Kernel_vectorAddGPU creation: %s
", getErrorString(err)); // Uncomment this line to get error string for the error code returned by clCreateKernel while creating the Kernel: vectorAddGPU*/
}

cl_kernel __cu2cl_Kernel_vectorAddGPU;
cl_program __cu2cl_Program_simpleZeroCopy_cu;
extern const char *progSrc;
extern size_t progLen;

extern cl_platform_id __cu2cl_Platform;
extern cl_device_id __cu2cl_Device;
extern cl_context __cu2cl_Context;
extern cl_command_queue __cu2cl_CommandQueue;

extern size_t globalWorkSize[3];
extern size_t localWorkSize[3];
void __cu2cl_Cleanup_simpleZeroCopy_cu() {
    clReleaseKernel(__cu2cl_Kernel_vectorAddGPU);
    clReleaseProgram(__cu2cl_Program_simpleZeroCopy_cu);
}
////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////


// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime


// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif


/* Add two vectors on the GPU */


// Allocate generic memory with malloc() and pin it laster instead of using cudaHostAlloc()
bool bPinGenericMemory = false;

// Macro to aligned up to the memory size in question
#define MEMORY_ALIGNMENT  4096
#define ALIGN_UP(x,size) ( ((size_t)x+(size-1))&(~(size-1)) )

int main(int argc, char **argv)
{
__cu2cl_Init();

    int n, nelem, deviceCount;
    int idev = 0; // use default device 0
    char *device = NULL;
    unsigned int flags;
    size_t bytes;
    float *a, *b, *c;                      // Pinned memory allocated on the CPU
    float *a_UA, *b_UA, *c_UA;             // Non-4K Aligned Pinned memory on the CPU
    float *d_a, *d_b, *d_c;                // Device pointers for mapped memory
    float errorNorm, refNorm, ref, diff;
    __cu2cl_DeviceProp deviceProp;

    if (checkCmdLineFlag(argc, (const char **)argv, "help"))
    {
        printf("Usage:  simpleZeroCopy [OPTION]\n\n");
        printf("Options:\n");
        printf("  --device=[device #]  Specify the device to be used\n");
        printf("  --use_generic_memory (optional) use generic page-aligned for system memory\n");
        return EXIT_SUCCESS;
    }

    /* Get the device selected by the user or default to 0, and then set it. */
    if (getCmdLineArgumentString(argc, (const char **)argv, "device", &device))
    {
        cudaGetDeviceCount(&deviceCount);
        idev = atoi(device);

        if (idev >= deviceCount || idev < 0)
        {
            fprintf(stderr, "Device number %d is invalid, will use default CUDA device 0.\n", idev);
            idev = 0;
        }
    }

    // if GPU found supports SM 1.2, then continue, otherwise we exit 
    if (!checkCudaCapabilities(1, 2))
    {
        exit(EXIT_SUCCESS);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "use_generic_memory"))
    {
#if defined(__APPLE__) || defined(MACOSX)
        bPinGenericMemory = false;  // Generic Pinning of System Paged memory is not currently supported on Mac OSX
#else
        bPinGenericMemory = true;
#endif
    }

    if (bPinGenericMemory)
    {
        printf("> Using Generic System Paged Memory (malloc)\n");
    }
    else
    {
        printf("> Using CUDA Host Allocated (cudaHostAlloc)\n");
    }

    checkCudaErrors(cudaSetDevice(idev));

    /* Verify the selected device supports mapped memory and set the device
       flags for mapping host memory. */

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, idev));

#if CUDART_VERSION >= 2020

    if (!deviceProp.canMapHostMemory)
    {
        fprintf(stderr, "Device %d does not support mapping CPU host memory!\n", idev);

        exit(EXIT_SUCCESS);
    }

    checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));
#else
    fprintf(stderr, "CUDART version %d.%d does not support <cudaDeviceProp.canMapHostMemory> field\n", , CUDART_VERSION/1000, (CUDART_VERSION%100)/10);

    exit(EXIT_SUCCESS);
#endif

#if CUDART_VERSION < 4000

    if (bPinGenericMemory)
    {
        fprintf(stderr, "CUDART version %d.%d does not support <cudaHostRegister> function\n", CUDART_VERSION/1000, (CUDART_VERSION%100)/10);

        exit(EXIT_SUCCESS);
    }

#endif

    /* Allocate mapped CPU memory. */

    nelem = 1048576;
    bytes = nelem*sizeof(float);

    if (bPinGenericMemory)
    {
#if CUDART_VERSION >= 4000
        a_UA = (float *) malloc(bytes + MEMORY_ALIGNMENT);
        b_UA = (float *) malloc(bytes + MEMORY_ALIGNMENT);
        c_UA = (float *) malloc(bytes + MEMORY_ALIGNMENT);

        // We need to ensure memory is aligned to 4K (so we will need to padd memory accordingly)
        a = (float *) ALIGN_UP(a_UA, MEMORY_ALIGNMENT);
        b = (float *) ALIGN_UP(b_UA, MEMORY_ALIGNMENT);
        c = (float *) ALIGN_UP(c_UA, MEMORY_ALIGNMENT);

        checkCudaErrors(cudaHostRegister(a, bytes, cudaHostRegisterMapped));
        checkCudaErrors(cudaHostRegister(b, bytes, cudaHostRegisterMapped));
        checkCudaErrors(cudaHostRegister(c, bytes, cudaHostRegisterMapped));
#endif
    }
    else
    {
#if CUDART_VERSION >= 2020
        flags = cudaHostAllocMapped;
        checkCudaErrors(cudaHostAlloc((void **)&a, bytes, flags));
        checkCudaErrors(cudaHostAlloc((void **)&b, bytes, flags));
        checkCudaErrors(cudaHostAlloc((void **)&c, bytes, flags));
#endif
    }

    /* Initialize the vectors. */

    for (n = 0; n < nelem; n++)
    {
        a[n] = rand() / (float)RAND_MAX;
        b[n] = rand() / (float)RAND_MAX;
    }

    /* Get the device pointers for the pinned CPU memory mapped into the GPU
       memory space. */

#if CUDART_VERSION >= 2020
    checkCudaErrors(cudaHostGetDevicePointer((void **)&d_a, (void *)a, 0));
    checkCudaErrors(cudaHostGetDevicePointer((void **)&d_b, (void *)b, 0));
    checkCudaErrors(cudaHostGetDevicePointer((void **)&d_c, (void *)c, 0));
#endif

    /* Call the GPU kernel using the CPU pointers residing in CPU mapped memory. */
    printf("> vectorAddGPU kernel will add vectors using mapped CPU memory...\n");
    size_t block[3] = {256, 1, 1};
    size_t grid[3] = {(unsigned int)ceil(nelem/(float)block[0]), 1, 1};
    err = clSetKernelArg(__cu2cl_Kernel_vectorAddGPU, 0, sizeof(float *), &d_a);
/*printf("clSetKernelArg for argument 0 of kernel __cu2cl_Kernel_vectorAddGPU is: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_vectorAddGPU, 1, sizeof(float *), &d_b);
/*printf("clSetKernelArg for argument 1 of kernel __cu2cl_Kernel_vectorAddGPU is: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_vectorAddGPU, 2, sizeof(float *), &d_c);
/*printf("clSetKernelArg for argument 2 of kernel __cu2cl_Kernel_vectorAddGPU is: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_vectorAddGPU, 3, sizeof(int), &nelem);
/*printf("clSetKernelArg for argument 3 of kernel __cu2cl_Kernel_vectorAddGPU is: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
localWorkSize[0] = block[0];
localWorkSize[1] = block[1];
localWorkSize[2] = block[2];
globalWorkSize[0] = grid[0]*localWorkSize[0];
globalWorkSize[1] = grid[1]*localWorkSize[1];
globalWorkSize[2] = grid[2]*localWorkSize[2];
err = clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_vectorAddGPU, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
//printf("clEnqueueNDRangeKernel for the kernel __cu2cl_Kernel_vectorAddGPU: %s\n", getErrorString(err));
    checkCudaErrors(cudaDeviceSynchronize());
    getLastCudaError("vectorAddGPU() execution failed");

    /* Compare the results */

    printf("> Checking the results from vectorAddGPU() ...\n");
    errorNorm = 0.f;
    refNorm = 0.f;

    for (n = 0; n < nelem; n++)
    {
        ref = a[n] + b[n];
        diff = c[n] - ref;
        errorNorm += diff*diff;
        refNorm += ref*ref;
    }

    errorNorm = (float)sqrt((double)errorNorm);
    refNorm = (float)sqrt((double)refNorm);

    /* Memory clean up */

    printf("> Releasing CPU memory...\n");

    if (bPinGenericMemory)
    {
#if CUDART_VERSION >= 4000
        checkCudaErrors(cudaHostUnregister(a));
        checkCudaErrors(cudaHostUnregister(b));
        checkCudaErrors(cudaHostUnregister(c));
        free(a_UA);
        free(b_UA);
        free(c_UA);
#endif
    }
    else
    {
#if CUDART_VERSION >= 2020
        checkCudaErrors(cudaFreeHost(a));
        checkCudaErrors(cudaFreeHost(b));
        checkCudaErrors(cudaFreeHost(c));
#endif
    }

    exit(errorNorm/refNorm < 1.e-6f ? EXIT_SUCCESS : EXIT_FAILURE);
__cu2cl_Cleanup();
}
