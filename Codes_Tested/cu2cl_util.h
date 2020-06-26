Did this apprear instead of license??
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif
void __cu2cl_Init();

void __cu2cl_Cleanup();
size_t __cu2cl_LoadProgramSource(const char *filename, const char **progSrc);

cl_int __cu2cl_Memset(cl_mem devPtr, int value, size_t count);

void __cu2cl_Init_redEyeGPU_cu();

void __cu2cl_Cleanup_redEyeGPU_cu();

void __cu2cl_Init_redEYECPU_cu();

void __cu2cl_Cleanup_redEYECPU_cu();


#ifdef __cplusplus
}
#endif
