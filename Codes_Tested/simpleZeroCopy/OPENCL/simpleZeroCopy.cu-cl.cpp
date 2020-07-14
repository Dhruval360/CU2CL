#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include "cu2cl_util.h"




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
void __cu2cl_Init_simpleZeroCopy_cu() {
    #ifdef WITH_ALTERA
    progLen = __cu2cl_LoadProgramSource("simpleZeroCopy_cu_cl.aocx", &progSrc);
    __cu2cl_Program_simpleZeroCopy_cu = clCreateProgramWithBinary(__cu2cl_Context, 1, &__cu2cl_Device, &progLen, (const unsigned char **)&progSrc, NULL, NULL);
    #else
    progLen = __cu2cl_LoadProgramSource("simpleZeroCopy.cu-cl.cl", &progSrc);
    __cu2cl_Program_simpleZeroCopy_cu = clCreateProgramWithSource(__cu2cl_Context, 1, &progSrc, &progLen, NULL);
    #endif
    free((void *) progSrc);
    clBuildProgram(__cu2cl_Program_simpleZeroCopy_cu, 1, &__cu2cl_Device, "-I . ", NULL, NULL);
    __cu2cl_Kernel_vectorAddGPU = clCreateKernel(__cu2cl_Program_simpleZeroCopy_cu, "vectorAddGPU", NULL);
}

cl_int err;
const char *getErrorString(cl_int error){
switch(error){
// run-time and JIT compiler errors
	case 0: return "CL_SUCCESS";
	case -1: return "CL_DEVICE_NOT_FOUND";
	case -2: return "CL_DEVICE_NOT_AVAILABLE";
	case -3: return "CL_COMPILER_NOT_AVAILABLE";
	case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	case -5: return "CL_OUT_OF_RESOURCES";
	case -6: return "CL_OUT_OF_HOST_MEMORY";
	case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
	case -8: return "CL_MEM_COPY_OVERLAP";
	case -9: return "CL_IMAGE_FORMAT_MISMATCH";
	case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case -11: return "CL_BUILD_PROGRAM_FAILURE";
	case -12: return "CL_MAP_FAILURE";
	case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
	case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
	case -15: return "CL_COMPILE_PROGRAM_FAILURE";
	case -16: return "CL_LINKER_NOT_AVAILABLE";
	case -17: return "CL_LINK_PROGRAM_FAILURE";
	case -18: return "CL_DEVICE_PARTITION_FAILED";
	case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
	// compile-time errors
	case -30: return "CL_INVALID_VALUE";
	case -31: return "CL_INVALID_DEVICE_TYPE";
	case -32: return "CL_INVALID_PLATFORM";
	case -33: return "CL_INVALID_DEVICE";
	case -34: return "CL_INVALID_CONTEXT";
	case -35: return "CL_INVALID_QUEUE_PROPERTIES";
	case -36: return "CL_INVALID_COMMAND_QUEUE";
	case -37: return "CL_INVALID_HOST_PTR";
	case -38: return "CL_INVALID_MEM_OBJECT";
	case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	case -40: return "CL_INVALID_IMAGE_SIZE";
	case -41: return "CL_INVALID_SAMPLER";
	case -42: return "CL_INVALID_BINARY";
	case -43: return "CL_INVALID_BUILD_OPTIONS";
	case -44: return "CL_INVALID_PROGRAM";
	case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
	case -46: return "CL_INVALID_KERNEL_NAME";
	case -47: return "CL_INVALID_KERNEL_DEFINITION";
	case -48: return "CL_INVALID_KERNEL";
	case -49: return "CL_INVALID_ARG_INDEX";
	case -50: return "CL_INVALID_ARG_VALUE";
	case -51: return "CL_INVALID_ARG_SIZE";
	case -52: return "CL_INVALID_KERNEL_ARGS";
	case -53: return "CL_INVALID_WORK_DIMENSION";
	case -54: return "CL_INVALID_WORK_GROUP_SIZE";
	case -55: return "CL_INVALID_WORK_ITEM_SIZE";
	case -56: return "CL_INVALID_GLOBAL_OFFSET";
	case -57: return "CL_INVALID_EVENT_WAIT_LIST";
	case -58: return "CL_INVALID_EVENT";
	case -59: return "CL_INVALID_OPERATION";
	case -60: return "CL_INVALID_GL_OBJECT";
	case -61: return "CL_INVALID_BUFFER_SIZE";
	case -62: return "CL_INVALID_MIP_LEVEL";
	case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
	case -64: return "CL_INVALID_PROPERTY";
	case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
	case -66: return "CL_INVALID_COMPILER_OPTIONS";
	case -67: return "CL_INVALID_LINKER_OPTIONS";
	case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
	// extension errors
	case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
	case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
	case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
	case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
	case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
	case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
	default: return "Unknown OpenCL error";
}
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
    clSetKernelArg(__cu2cl_Kernel_vectorAddGPU, 0, sizeof(float *), &d_a);
clSetKernelArg(__cu2cl_Kernel_vectorAddGPU, 1, sizeof(float *), &d_b);
clSetKernelArg(__cu2cl_Kernel_vectorAddGPU, 2, sizeof(float *), &d_c);
clSetKernelArg(__cu2cl_Kernel_vectorAddGPU, 3, sizeof(int), &nelem);
localWorkSize[0] = block[0];
localWorkSize[1] = block[1];
localWorkSize[2] = block[2];
globalWorkSize[0] = grid[0]*localWorkSize[0];
globalWorkSize[1] = grid[1]*localWorkSize[1];
globalWorkSize[2] = grid[2]*localWorkSize[2];
clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_vectorAddGPU, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
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
