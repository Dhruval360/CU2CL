#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include "cu2cl_util.h"




cl_kernel __cu2cl_Kernel_kernel;
cl_program __cu2cl_Program_stream_cu;
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
    __cu2cl_Program_stream_cu = clCreateProgramWithBinary(__cu2cl_Context, 1, &__cu2cl_Device, &progLen, (const unsigned char **)&progSrc, NULL, NULL);
    #else
    progLen = __cu2cl_LoadProgramSource("stream.cu-cl.cl", &progSrc);
    __cu2cl_Program_stream_cu = clCreateProgramWithSource(__cu2cl_Context, 1, &progSrc, &progLen, NULL);
    #endif
    free((void *) progSrc);
    clBuildProgram(__cu2cl_Program_stream_cu, 1, &__cu2cl_Device, "-I . ", NULL, NULL);
    __cu2cl_Kernel_kernel = clCreateKernel(__cu2cl_Program_stream_cu, "kernel", NULL);
}

const int N = 1 << 20;



int main()
{
__cu2cl_Init();

    const int num_streams = 8;

    cl_command_queue streams[num_streams];
    cl_mem data[num_streams];

    for (int i = 0; i < num_streams; i++) {
        *&streams[i] = clCreateCommandQueue(__cu2cl_Context, __cu2cl_Device, CL_QUEUE_PROFILING_ENABLE, NULL);
 
        *&data[i] = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, N * sizeof(float), NULL, NULL);
        
        // launch one worker kernel per stream
        clSetKernelArg(__cu2cl_Kernel_kernel, 0, sizeof(cl_mem), &data[i]);
clSetKernelArg(__cu2cl_Kernel_kernel, 1, sizeof(int), &N);
localWorkSize[0] = 64;
globalWorkSize[0] = (1)*localWorkSize[0];
clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

        // launch a dummy kernel on the default stream
/*CU2CL Note -- Inserted temporary variable for kernel literal argument 0!*/
/*CU2CL Note -- Inserted temporary variable for kernel literal argument 1!*/
        float * __cu2cl_Kernel_kernel_temp_arg_0 = 0;
clSetKernelArg(__cu2cl_Kernel_kernel, 0, sizeof(float *), &__cu2cl_Kernel_kernel_temp_arg_0);
int __cu2cl_Kernel_kernel_temp_arg_1 = 0;
clSetKernelArg(__cu2cl_Kernel_kernel, 1, sizeof(int), &__cu2cl_Kernel_kernel_temp_arg_1);
localWorkSize[0] = 1;
globalWorkSize[0] = (1)*localWorkSize[0];
clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    }

/*CU2CL Unsupported -- Unsupported CUDA call: cudaDeviceReset*/
    cudaDeviceReset();

    return 0;
__cu2cl_Cleanup();
}
