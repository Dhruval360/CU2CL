//Did this apprear instead of license??
#include "cu2cl_util.h"
extern cl_kernel __cu2cl_Kernel_box_blur;
extern cl_kernel __cu2cl_Kernel_light_edge_detection;
extern cl_kernel __cu2cl_Kernel_separateChannels;
extern cl_kernel __cu2cl_Kernel_recombineChannels;
extern cl_program __cu2cl_Program_BoxBlur_TotalVariation_cu;
const char *progSrc;
size_t progLen;

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


cl_int __cu2cl_EventElapsedTime(float *ms, cl_event start, cl_event end) {
    cl_int ret;
    cl_ulong s, e;
    float fs, fe;
    ret |= clGetEventProfilingInfo(start, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &s, NULL);
    ret |= clGetEventProfilingInfo(end, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &e, NULL);
    s = e - s;
    *ms = ((float) s)/1000000.0;
    return ret;
}


void __cu2cl_Init() {
    clGetPlatformIDs(1, &__cu2cl_Platform, NULL);
    clGetDeviceIDs(__cu2cl_Platform, CL_DEVICE_TYPE_ALL, 1, &__cu2cl_Device, NULL);
    __cu2cl_Context = clCreateContext(NULL, 1, &__cu2cl_Device, NULL, NULL, NULL);
    __cu2cl_CommandQueue = clCreateCommandQueue(__cu2cl_Context, __cu2cl_Device, CL_QUEUE_PROFILING_ENABLE, NULL);
    __cu2cl_Init_BoxBlur_TotalVariation_cu();
}

void __cu2cl_Cleanup() {
    __cu2cl_Cleanup_BoxBlur_TotalVariation_cu();
    clReleaseCommandQueue(__cu2cl_CommandQueue);
    clReleaseContext(__cu2cl_Context);
}
