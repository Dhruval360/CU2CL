//Did this apprear instead of license??
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

cl_int __cu2cl_EventElapsedTime(float *ms, cl_event start, cl_event end);

void __cu2cl_Init_BoxBlur_TotalVariation_cu();

void __cu2cl_Cleanup_BoxBlur_TotalVariation_cu();


#ifdef __cplusplus
}
#endif
