#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include "cu2cl_util.h"




cl_kernel __cu2cl_Kernel_saxpy;
cl_program __cu2cl_Program_sxpy_cu;
extern const char *progSrc;
extern size_t progLen;

extern cl_platform_id __cu2cl_Platform;
extern cl_device_id __cu2cl_Device;
extern cl_context __cu2cl_Context;
extern cl_command_queue __cu2cl_CommandQueue;

extern size_t globalWorkSize[3];
extern size_t localWorkSize[3];
void __cu2cl_Cleanup_sxpy_cu() {
    clReleaseKernel(__cu2cl_Kernel_saxpy);
    clReleaseProgram(__cu2cl_Program_sxpy_cu);
}
void __cu2cl_Init_sxpy_cu() {
    #ifdef WITH_ALTERA
    progLen = __cu2cl_LoadProgramSource("sxpy_cu_cl.aocx", &progSrc);
    __cu2cl_Program_sxpy_cu = clCreateProgramWithBinary(__cu2cl_Context, 1, &__cu2cl_Device, &progLen, (const unsigned char **)&progSrc, NULL, &err);
    #else
    progLen = __cu2cl_LoadProgramSource("sxpy.cu-cl.cl", &progSrc);
    __cu2cl_Program_sxpy_cu = clCreateProgramWithSource(__cu2cl_Context, 1, &progSrc, &progLen, &err);
    //printf("clCreateProgramWithSource for sxpy.cu-cl.cl: %s\n", getErrorString(err));    #endif
    free((void *) progSrc);
    err = clBuildProgram(__cu2cl_Program_sxpy_cu, 1, &__cu2cl_Device, "-I . ", NULL, NULL);
    /*printf("clBuildProgram : %s\n", getErrorString(err)); //Uncomment this line to access the error string of the error code returned by clBuildProgram*/
    if(err != CL_SUCCESS){        std::vector<char> buildLog;        size_t logSize;        err = clGetProgramBuildInfo(__cu2cl_Program_sxpy_cu, &__cu2cl_Device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);        printf("clGetProgramBuildInfo : %s\n", getErrorString(err));        buildLog.resize(logSize);        clGetProgramBuildInfo(__cu2cl_Program_sxpy_cu, __cu2cl_Device, CL_PROGRAM_BUILD_LOG, logSize, &buildLog[0], nullptr);        std::cout << &buildLog[0] << '
';    }    __cu2cl_Kernel_saxpy = clCreateKernel(__cu2cl_Program_sxpy_cu, "saxpy", &err);
    /*printf("__cu2cl_Kernel_saxpy creation: %s
", getErrorString(err)); // Uncomment this line to get error string for the error code returned by clCreateKernel while creating the Kernel: saxpy*/
}

#include <stdio.h>



int main(void)
{
__cu2cl_Init();

  int N = 1<<20;
  cl_mem d_x;
float *x;
float *y;
cl_mem d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  d_x = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, N*sizeof(float), NULL, &err);
//printf("clCreateBuffer for device variable d_x is: %s\n", getErrorString(err)); 
  d_y = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, N*sizeof(float), NULL, &err);
//printf("clCreateBuffer for device variable d_y is: %s\n", getErrorString(err));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  err = clEnqueueWriteBuffer(__cu2cl_CommandQueue, d_x, CL_TRUE, 0, N*sizeof(float), x, 0, NULL, NULL);
//printf("Memory copy from host variable x to device variable d_x: %s\n", getErrorString(err));
  err = clEnqueueWriteBuffer(__cu2cl_CommandQueue, d_y, CL_TRUE, 0, N*sizeof(float), y, 0, NULL, NULL);
//printf("Memory copy from host variable y to device variable d_y: %s\n", getErrorString(err));

  // Perform SAXPY on 1M elements
/*CU2CL Note -- Inserted temporary variable for kernel literal argument 1!*/
  err = clSetKernelArg(__cu2cl_Kernel_saxpy, 0, sizeof(int), &N);
/*printf("clSetKernelArg for argument 0 of kernel __cu2cl_Kernel_saxpy is: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
float __cu2cl_Kernel_saxpy_temp_arg_1 = 2.0f;
err = clSetKernelArg(__cu2cl_Kernel_saxpy, 1, sizeof(float), &__cu2cl_Kernel_saxpy_temp_arg_1);
/*printf("clSetKernelArg for argument 1 of kernel __cu2cl_Kernel_saxpy is: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_saxpy, 2, sizeof(cl_mem), &d_x);
/*printf("clSetKernelArg for argument 2 of kernel __cu2cl_Kernel_saxpy is: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_saxpy, 3, sizeof(cl_mem), &d_y);
/*printf("clSetKernelArg for argument 3 of kernel __cu2cl_Kernel_saxpy is: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
localWorkSize[0] = 256;
globalWorkSize[0] = ((N+255)/256)*localWorkSize[0];
err = clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_saxpy, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
//printf("clEnqueueNDRangeKernel for the kernel __cu2cl_Kernel_saxpy: %s\n", getErrorString(err));

  err = clEnqueueReadBuffer(__cu2cl_CommandQueue, d_y, CL_TRUE, 0, N*sizeof(float), y, 0, NULL, NULL);
//printf("Memory copy from device variable y to host variable d_y: %s\n", getErrorString(err));

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-4.0f));
  printf("Max error: %f\n", maxError);

  clReleaseMemObject(d_x);
  clReleaseMemObject(d_y);
  free(x);
  free(y);
__cu2cl_Cleanup();
}
