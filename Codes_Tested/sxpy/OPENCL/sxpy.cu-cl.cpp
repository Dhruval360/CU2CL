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
    __cu2cl_Program_sxpy_cu = clCreateProgramWithBinary(__cu2cl_Context, 1, &__cu2cl_Device, &progLen, (const unsigned char **)&progSrc, NULL, NULL);
    #else
    progLen = __cu2cl_LoadProgramSource("sxpy.cu-cl.cl", &progSrc);
    __cu2cl_Program_sxpy_cu = clCreateProgramWithSource(__cu2cl_Context, 1, &progSrc, &progLen, NULL);
    #endif
    free((void *) progSrc);
    clBuildProgram(__cu2cl_Program_sxpy_cu, 1, &__cu2cl_Device, "-I . ", NULL, NULL);
    __cu2cl_Kernel_saxpy = clCreateKernel(__cu2cl_Program_sxpy_cu, "saxpy", NULL);
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

  *&d_x = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, N*sizeof(float), NULL, NULL); 
  *&d_y = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, N*sizeof(float), NULL, NULL);

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  clEnqueueWriteBuffer(__cu2cl_CommandQueue, d_x, CL_TRUE, 0, N*sizeof(float), x, 0, NULL, NULL);
  clEnqueueWriteBuffer(__cu2cl_CommandQueue, d_y, CL_TRUE, 0, N*sizeof(float), y, 0, NULL, NULL);

  // Perform SAXPY on 1M elements
/*CU2CL Note -- Inserted temporary variable for kernel literal argument 1!*/
  clSetKernelArg(__cu2cl_Kernel_saxpy, 0, sizeof(int), &N);
float __cu2cl_Kernel_saxpy_temp_arg_1 = 2.0f;
clSetKernelArg(__cu2cl_Kernel_saxpy, 1, sizeof(float), &__cu2cl_Kernel_saxpy_temp_arg_1);
clSetKernelArg(__cu2cl_Kernel_saxpy, 2, sizeof(cl_mem), &d_x);
clSetKernelArg(__cu2cl_Kernel_saxpy, 3, sizeof(cl_mem), &d_y);
localWorkSize[0] = 256;
globalWorkSize[0] = ((N+255)/256)*localWorkSize[0];
clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_saxpy, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

  clEnqueueReadBuffer(__cu2cl_CommandQueue, d_y, CL_TRUE, 0, N*sizeof(float), y, 0, NULL, NULL);

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
