#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include "cu2cl_util.h"




cl_kernel __cu2cl_Kernel_device_add;
cl_program __cu2cl_Program_vectorAdd_cu;
/*
const char *progSrc;
size_t progLen;

 cl_platform_id __cu2cl_Platform;
 cl_device_id __cu2cl_Device;
cl_context __cu2cl_Context;
 cl_command_queue __cu2cl_CommandQueue;

 size_t globalWorkSize[3];
size_t localWorkSize[3];
*/
const char* progSrc;
size_t progLen;

cl_platform_id __cu2cl_Platform;
cl_device_id __cu2cl_Device;
cl_context __cu2cl_Context;
cl_command_queue __cu2cl_CommandQueue;

size_t globalWorkSize[3];
size_t localWorkSize[3];
size_t __cu2cl_LoadProgramSource(const char* filename, const char** progSrc) {
	printf("%s\n", filename);
	FILE* f = fopen(filename, "r");
	if (f == NULL)
	{
		printf("it is coming NULL\n");
	}
	fseek(f, 0, SEEK_END);
	size_t len = (size_t)ftell(f);
	*progSrc = (const char*)malloc(sizeof(char) * len);
	rewind(f);
	fread((void*)*progSrc, len, 1, f);
	fclose(f);
	return len;
}


void __cu2cl_Init() {
	clGetPlatformIDs(1, &__cu2cl_Platform, NULL);
	clGetDeviceIDs(__cu2cl_Platform, CL_DEVICE_TYPE_ALL, 1, &__cu2cl_Device, NULL);
	__cu2cl_Context = clCreateContext(NULL, 1, &__cu2cl_Device, NULL, NULL, NULL);
	__cu2cl_CommandQueue = clCreateCommandQueue(__cu2cl_Context, __cu2cl_Device, CL_QUEUE_PROFILING_ENABLE, NULL);
	__cu2cl_Init_vectorAdd_cu();
}

void __cu2cl_Cleanup() {
	__cu2cl_Cleanup_vectorAdd_cu();
	clReleaseCommandQueue(__cu2cl_CommandQueue);
	clReleaseContext(__cu2cl_Context);
}

void __cu2cl_Cleanup_vectorAdd_cu() {
    clReleaseKernel(__cu2cl_Kernel_device_add);
    clReleaseProgram(__cu2cl_Program_vectorAdd_cu);
}
void __cu2cl_Init_vectorAdd_cu() {
    #ifdef WITH_ALTERA
    progLen = __cu2cl_LoadProgramSource("vectorAdd_cu_cl.aocx", &progSrc);
    __cu2cl_Program_vectorAdd_cu = clCreateProgramWithBinary(__cu2cl_Context, 1, &__cu2cl_Device, &progLen, (const unsigned char **)&progSrc, NULL, NULL);
    #else
    progLen = __cu2cl_LoadProgramSource("vectorAdd.cu-cl.cl", &progSrc);
    __cu2cl_Program_vectorAdd_cu = clCreateProgramWithSource(__cu2cl_Context, 1, &progSrc, &progLen, NULL);
    #endif
    free((void *) progSrc);
    clBuildProgram(__cu2cl_Program_vectorAdd_cu, 1, &__cu2cl_Device, "-I . ", NULL, NULL);
    __cu2cl_Kernel_device_add = clCreateKernel(__cu2cl_Program_vectorAdd_cu, "device_add", NULL);
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
# define N 32


void fill_array(int *data)
{
	for(int i=0;i<N;i++)
		data[i]=i;
}

void host_add(int *a,int* b,int* c)
{
	for(int i=0;i<N;i++)
		c[i] = a[i] + b[i];
}

void print_output(int* op)
{
	for(int i=0;i<N;i++)
		printf("%d\t",op[i]);
}



int main(void)
{
__cu2cl_Init();

	int *a,*b,*c;
	int size = N * sizeof(int);
	a= (int*)malloc(size);
	b= (int*)malloc(size);
	c= (int*)malloc(size);
	fill_array(a);
	fill_array(b);
	//print_output(a);
	//print_output(b);

	cl_mem d_a;
cl_mem d_b;
cl_mem d_c;
	*(void **)&d_a = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, size, NULL, NULL);
	*(void **)&d_b = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, size, NULL, NULL);
	*(void **)&d_c = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, size, NULL, NULL);


	clEnqueueWriteBuffer(__cu2cl_CommandQueue, d_a, CL_TRUE, 0, size, a, 0, NULL, NULL);
	clEnqueueWriteBuffer(__cu2cl_CommandQueue, d_b, CL_TRUE, 0, size, b, 0, NULL, NULL);


	clSetKernelArg(__cu2cl_Kernel_device_add, 0, sizeof(cl_mem), &d_a);
clSetKernelArg(__cu2cl_Kernel_device_add, 1, sizeof(cl_mem), &d_b);
clSetKernelArg(__cu2cl_Kernel_device_add, 2, sizeof(cl_mem), &d_c);
localWorkSize[0] = N/2;
globalWorkSize[0] = (2)*localWorkSize[0];
clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_device_add, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	
	clEnqueueReadBuffer(__cu2cl_CommandQueue, d_c, CL_TRUE, 0, size, c, 0, NULL, NULL);
	// host_add(a,b,c);

	print_output(c);

	free(a);free(b);free(c);
	clReleaseMemObject(d_a);clReleaseMemObject(d_b);clReleaseMemObject(d_c);
return 0;
__cu2cl_Cleanup();
}
