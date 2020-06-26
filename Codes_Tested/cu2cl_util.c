Did this apprear instead of license??
#include "cu2cl_util.h"
extern cl_kernel __cu2cl_Kernel_naive_normalized_cross_correlation;
extern cl_kernel __cu2cl_Kernel_remove_redness_from_coordinates;
extern cl_program __cu2cl_Program_redEYECPU_cu;
extern cl_kernel __cu2cl_Kernel_naive_normalized_cross_correlation;
extern cl_kernel __cu2cl_Kernel_remove_redness_from_coordinates;
extern cl_kernel __cu2cl_Kernel_histogram_kernel;
extern cl_kernel __cu2cl_Kernel_exclusive_scan_kernel;
extern cl_kernel __cu2cl_Kernel_move_kernel;
extern cl_program __cu2cl_Program_redEyeGPU_cu;
const char *progSrc;
size_t progLen;

cl_kernel __cu2cl_Kernel___cu2cl_Memset;
cl_program __cu2cl_Util_Program;
cl_platform_id __cu2cl_Platform;
cl_device_id __cu2cl_Device;
cl_context __cu2cl_Context;
cl_command_queue __cu2cl_CommandQueue;

size_t globalWorkSize[3];
size_t localWorkSize[3];
size_t __cu2cl_LoadProgramSource(const char *filename, const char **progSrc) {
    FILE *f = fopen(filename, "r");
    fseek(f, 0, SEEK_END);
    size_t len = (size_t) ftell(f);
    *progSrc = (const char *) malloc(sizeof(char)*len);
    rewind(f);
    fread((void *) *progSrc, len, 1, f);
    fclose(f);
    return len;
}


cl_int __cu2cl_Memset(cl_mem devPtr, int value, size_t count) {
    clSetKernelArg(__cu2cl_Kernel___cu2cl_Memset, 0, sizeof(cl_mem), &devPtr);
    clSetKernelArg(__cu2cl_Kernel___cu2cl_Memset, 1, sizeof(cl_uchar), &value);
    clSetKernelArg(__cu2cl_Kernel___cu2cl_Memset, 2, sizeof(cl_uint), &count);
    globalWorkSize[0] = count;
    return clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel___cu2cl_Memset, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
}


void __cu2cl_Init() {
    clGetPlatformIDs(1, &__cu2cl_Platform, NULL);
    clGetDeviceIDs(__cu2cl_Platform, CL_DEVICE_TYPE_ALL, 1, &__cu2cl_Device, NULL);
    __cu2cl_Context = clCreateContext(NULL, 1, &__cu2cl_Device, NULL, NULL, NULL);
    __cu2cl_CommandQueue = clCreateCommandQueue(__cu2cl_Context, __cu2cl_Device, CL_QUEUE_PROFILING_ENABLE, NULL);
    __cu2cl_Init_redEyeGPU_cu();
    __cu2cl_Init_redEYECPU_cu();
    #ifdef WITH_ALTERA
    progLen = __cu2cl_LoadProgramSource("cu2cl_util.aocx", &progSrc);
    __cu2cl_Util_Program = clCreateProgramWithBinary(__cu2cl_Context, 1, &__cu2cl_Device, &progLen, (const unsigned char **)&progSrc, NULL, NULL);
    #else
    progLen = __cu2cl_LoadProgramSource("cu2cl_util.cl", &progSrc);
    __cu2cl_Util_Program = clCreateProgramWithSource(__cu2cl_Context, 1, &progSrc, &progLen, NULL);
    #endif
    free((void *) progSrc);
    clBuildProgram(__cu2cl_Util_Program, 1, &__cu2cl_Device, "-I . ", NULL, NULL);
    __cu2cl_Kernel___cu2cl_Memset = clCreateKernel(__cu2cl_Util_Program, "__cu2cl_Memset", NULL);
}

void __cu2cl_Cleanup() {
    clReleaseKernel(__cu2cl_Kernel___cu2cl_Memset);
    clReleaseProgram(__cu2cl_Util_Program);
    __cu2cl_Cleanup_redEYECPU_cu();
    __cu2cl_Cleanup_redEyeGPU_cu();
    clReleaseCommandQueue(__cu2cl_CommandQueue);
    clReleaseContext(__cu2cl_Context);
}
