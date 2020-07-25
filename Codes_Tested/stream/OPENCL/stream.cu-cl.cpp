cl_kernel __cu2cl_Kernel_kernel;
cl_program __cu2cl_Program_stream_cu;
cl_int err;
extern const char *progSrc;
extern size_t progLen;

extern cl_platform_id __cu2cl_Platform;
extern cl_device_id __cu2cl_Device;
extern cl_context __cu2cl_Context;
extern cl_command_queue __cu2cl_CommandQueue;

extern size_t globalWorkSize[3];
extern size_t localWorkSize[3];
void __cu2cl_Cleanup_stream_cu() {
    clReleaseKernel(__cu2cl_Kernel_kernel);
    clReleaseProgram(__cu2cl_Program_stream_cu);
}
void __cu2cl_Init_stream_cu() {
    #ifdef WITH_ALTERA
    progLen = __cu2cl_LoadProgramSource("stream_cu_cl.aocx", &progSrc);
    __cu2cl_Program_stream_cu = clCreateProgramWithBinary(__cu2cl_Context, 1, &__cu2cl_Device, &progLen, (const unsigned char **)&progSrc, NULL, &err);
    //printf("clCreateProgramWithBinary for stream.cu-cl.cl: %s\n", getErrorString(err));
    #else
    progLen = __cu2cl_LoadProgramSource("stream.cu-cl.cl", &progSrc);
    __cu2cl_Program_stream_cu = clCreateProgramWithSource(__cu2cl_Context, 1, &progSrc, &progLen, &err);
    //printf("clCreateProgramWithSource for stream.cu-cl.cl: %s\n", getErrorString(err));
    #endif
    free((void *) progSrc);
    err = clBuildProgram(__cu2cl_Program_stream_cu, 1, &__cu2cl_Device, "-I . ", NULL, NULL);
    /*printf("clBuildProgram : %s\n", getErrorString(err)); //Uncomment this line to access the error string of the error code returned by clBuildProgram*/
    if(err != CL_SUCCESS){
        std::vector<char> buildLog;
        size_t logSize;
        err = clGetProgramBuildInfo(__cu2cl_Program_stream_cu, __cu2cl_Device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        printf("clGetProgramBuildInfo : %s\n", getErrorString(err));
        buildLog.resize(logSize);
        clGetProgramBuildInfo(__cu2cl_Program_stream_cu, __cu2cl_Device, CL_PROGRAM_BUILD_LOG, logSize, &buildLog[0], NULL);
        printf("%s\n", &buildLog[0]);
    }
    __cu2cl_Kernel_kernel = clCreateKernel(__cu2cl_Program_stream_cu, "kernel", &err);
    /*printf("__cu2cl_Kernel_kernel creation: %s\n", getErrorString(err)); // Uncomment this line to get error string for the error code returned by clCreateKernel while creating the Kernel: kernel*/
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





const int N = 1 << 20;



int main()
{
__cu2cl_Init();

    const int num_streams = 8;

    cl_command_queue streams[num_streams];
    cl_mem data[num_streams];

    for (int i = 0; i < num_streams; i++) {
        streams[i] = clCreateCommandQueue(__cu2cl_Context, __cu2cl_Device, CL_QUEUE_PROFILING_ENABLE, &err);
//printf("clCreateCommandQueue for stream streams[i] is: %s\n", getErrorString(err));
 
        data[i] = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, N * sizeof(float), NULL, &err);
//printf("clCreateBuffer for device variable data[i]: %s\n", getErrorString(err));
        
        // launch one worker kernel per stream
        err = clSetKernelArg(__cu2cl_Kernel_kernel, 0, sizeof(cl_mem), &data[i]);
/*printf("clSetKernelArg for argument 0 of kernel __cu2cl_Kernel_kernel: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_kernel, 1, sizeof(int), &N);
/*printf("clSetKernelArg for argument 1 of kernel __cu2cl_Kernel_kernel: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
localWorkSize[0] = 64;
globalWorkSize[0] = (1)*localWorkSize[0];
err = clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
//printf("clEnqueueNDRangeKernel for the kernel __cu2cl_Kernel_kernel: %s\n", getErrorString(err));

        // launch a dummy kernel on the default stream
/*CU2CL Note -- Inserted temporary variable for kernel literal argument 0!*/
/*CU2CL Note -- Inserted temporary variable for kernel literal argument 1!*/
        float * __cu2cl_Kernel_kernel_temp_arg_0 = 0;
err = clSetKernelArg(__cu2cl_Kernel_kernel, 0, sizeof(float *), &__cu2cl_Kernel_kernel_temp_arg_0);
/*printf("clSetKernelArg for argument 0 of kernel __cu2cl_Kernel_kernel: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
int __cu2cl_Kernel_kernel_temp_arg_1 = 0;
err = clSetKernelArg(__cu2cl_Kernel_kernel, 1, sizeof(int), &__cu2cl_Kernel_kernel_temp_arg_1);
/*printf("clSetKernelArg for argument 1 of kernel __cu2cl_Kernel_kernel: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
localWorkSize[0] = 1;
globalWorkSize[0] = (1)*localWorkSize[0];
err = clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
//printf("clEnqueueNDRangeKernel for the kernel __cu2cl_Kernel_kernel: %s\n", getErrorString(err));
    }

/*CU2CL Unsupported -- Unsupported CUDA call: cudaDeviceReset*/
    cudaDeviceReset();

    return 0;
__cu2cl_Cleanup();
}
