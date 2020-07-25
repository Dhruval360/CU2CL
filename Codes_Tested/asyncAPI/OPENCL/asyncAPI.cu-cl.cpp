#include <vector>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include "cu2cl_util.h"




cl_kernel __cu2cl_Kernel_increment_kernel;
cl_program __cu2cl_Program_asyncAPI_cu;
cl_int err;
extern const char *progSrc;
extern size_t progLen;

extern cl_platform_id __cu2cl_Platform;
extern cl_device_id __cu2cl_Device;
extern cl_context __cu2cl_Context;
extern cl_command_queue __cu2cl_CommandQueue;

extern size_t globalWorkSize[3];
extern size_t localWorkSize[3];
void __cu2cl_Cleanup_asyncAPI_cu() {
    clReleaseKernel(__cu2cl_Kernel_increment_kernel);
    clReleaseProgram(__cu2cl_Program_asyncAPI_cu);
}
void __cu2cl_Init_asyncAPI_cu() {
    #ifdef WITH_ALTERA
    progLen = __cu2cl_LoadProgramSource("asyncAPI_cu_cl.aocx", &progSrc);
    __cu2cl_Program_asyncAPI_cu = clCreateProgramWithBinary(__cu2cl_Context, 1, &__cu2cl_Device, &progLen, (const unsigned char **)&progSrc, NULL, &err);
    //printf("clCreateProgramWithBinary for asyncAPI.cu-cl.cl: %s\n", getErrorString(err));
    #else
    progLen = __cu2cl_LoadProgramSource("asyncAPI.cu-cl.cl", &progSrc);
    __cu2cl_Program_asyncAPI_cu = clCreateProgramWithSource(__cu2cl_Context, 1, &progSrc, &progLen, &err);
    //printf("clCreateProgramWithSource for asyncAPI.cu-cl.cl: %s\n", getErrorString(err));
    #endif
    free((void *) progSrc);
    err = clBuildProgram(__cu2cl_Program_asyncAPI_cu, 1, &__cu2cl_Device, "-I . ", NULL, NULL);
    /*printf("clBuildProgram : %s\n", getErrorString(err)); //Uncomment this line to access the error string of the error code returned by clBuildProgram*/
    if(err != CL_SUCCESS){
        std::vector<char> buildLog;
        size_t logSize;
        err = clGetProgramBuildInfo(__cu2cl_Program_asyncAPI_cu, __cu2cl_Device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        printf("clGetProgramBuildInfo : %s\n", getErrorString(err));
        buildLog.resize(logSize);
        clGetProgramBuildInfo(__cu2cl_Program_asyncAPI_cu, __cu2cl_Device, CL_PROGRAM_BUILD_LOG, logSize, &buildLog[0], NULL);
        printf("%s\n", &buildLog[0]);
    }
    __cu2cl_Kernel_increment_kernel = clCreateKernel(__cu2cl_Program_asyncAPI_cu, "increment_kernel", &err);
    /*printf("__cu2cl_Kernel_increment_kernel creation: %s\n", getErrorString(err)); // Uncomment this line to get error string for the error code returned by clCreateKernel while creating the Kernel: increment_kernel*/
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

//
// This sample illustrates the usage of CUDA events for both GPU timing and
// overlapping CPU and GPU execution.  Events are inserted into a stream
// of CUDA calls.  Since CUDA stream calls are asynchronous, the CPU can
// perform computations while GPU is executing (including DMA memcopies
// between the host and device).  CPU can query CUDA events to determine
// whether GPU has completed tasks.
//

// includes, system
#include <stdio.h>

// includes CUDA Runtime


// includes, project
#include "helper_cuda.h-cl.h"
#include "helper_functions.h-cl.h 



bool correct_output(int *data, const int n, const int x)
{
    for (int i = 0; i < n; i++)
        if (data[i] != x)
        {
            printf("Error! data[%d] = %d, ref = %d\n", i, data[i], x);
            return false;
        }

    return true;
}

int main(int argc, char *argv[])
{
__cu2cl_Init();

    int devID;
    __cu2cl_DeviceProp deviceProps;

    printf("[%s] - Starting...\n", argv[0]);

    // This will pick the best possible CUDA capable device
    devID = findCudaDevice(argc, (const char **)argv);

    // get device name
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
    printf("CUDA device [%s]\n", deviceProps.name);

    int n = 16 * 1024 * 1024;
    int nbytes = n * sizeof(int);
    int value = 26;

    // allocate host memory
    int *a = 0;
    checkCudaErrors(cudaMallocHost((void **)&a, nbytes));
    memset(a, 0, nbytes);

    // allocate device memory
    int *d_a=0;
    checkCudaErrors(cudaMalloc((void **)&d_a, nbytes));
    checkCudaErrors(cudaMemset(d_a, 255, nbytes));

    // set kernel launch configuration
    size_t threads[3] = {512, 1, 1};
    size_t blocks[3]  = {n / threads[0], 1, 1};

    // create cuda event handles
    cl_event start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    checkCudaErrors(cudaDeviceSynchronize());
    float gpu_time = 0.0f;

    // asynchronously issue work to the GPU (all to stream 0)
    sdkStartTimer(&timer);
    err = clEnqueueMarker(__cu2cl_CommandQueue, &start);
//printf("clEnqueMarker for the event start: %s\n", getErrorString(err));
    err = clEnqueueWriteBuffer(__cu2cl_CommandQueue, d_a, CL_FALSE, 0, nbytes, a, 0, NULL, NULL);
//printf("Memory copy from host variable a to device variable d_a in stream __cu2cl_CommandQueue: %s\n", getErrorString(err));
    err = clSetKernelArg(__cu2cl_Kernel_increment_kernel, 0, sizeof(int *), &d_a);
/*printf("clSetKernelArg for argument 0 of kernel __cu2cl_Kernel_increment_kernel: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_increment_kernel, 1, sizeof(int), &value);
/*printf("clSetKernelArg for argument 1 of kernel __cu2cl_Kernel_increment_kernel: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
localWorkSize[0] = threads[0];
localWorkSize[1] = threads[1];
localWorkSize[2] = threads[2];
globalWorkSize[0] = blocks[0]*localWorkSize[0];
globalWorkSize[1] = blocks[1]*localWorkSize[1];
globalWorkSize[2] = blocks[2]*localWorkSize[2];
err = clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_increment_kernel, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
//printf("clEnqueueNDRangeKernel for the kernel __cu2cl_Kernel_increment_kernel: %s\n", getErrorString(err));
    clEnqueueReadBuffer(__cu2cl_CommandQueue, d_a, CL_FALSE, 0, nbytes, a, 0, NULL, NULL);
//printf("Memory copy from device variable a to host variable d_a in stream __cu2cl_CommandQueue: %s\n", getErrorString(err));
    err = clEnqueueMarker(__cu2cl_CommandQueue, &stop);
//printf("clEnqueMarker for the event stop: %s\n", getErrorString(err));
    sdkStopTimer(&timer);

    // have CPU do some work while waiting for stage 1 to finish
    unsigned long int counter=0;

    while (__cu2cl_EventQuery(stop) == cudaErrorNotReady)
    {
        counter++;
    }

    checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));

    // print the cpu and gpu times
    printf("time spent executing by the GPU: %.2f\n", gpu_time);
    printf("time spent by CPU in CUDA calls: %.2f\n", sdkGetTimerValue(&timer));
    printf("CPU executed %lu iterations while waiting for GPU to finish\n", counter);

    // check the output for correctness
    bool bFinalResults = correct_output(a, n, value);

    // release resources
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaFreeHost(a));
    checkCudaErrors(cudaFree(d_a));

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
/*CU2CL Unsupported -- Unsupported CUDA call: cudaDeviceReset*/
    cudaDeviceReset();

    exit(bFinalResults ? EXIT_SUCCESS : EXIT_FAILURE);
__cu2cl_Cleanup();
}

