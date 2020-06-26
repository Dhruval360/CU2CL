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
extern const char *progSrc;
extern size_t progLen;

extern cl_platform_id __cu2cl_Platform;
extern cl_device_id __cu2cl_Device;
extern cl_context __cu2cl_Context;
extern cl_command_queue __cu2cl_CommandQueue;

extern size_t globalWorkSize[3];
extern size_t localWorkSize[3];
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

#include <stdio.h>
# define N 512


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
		printf("%d\n",op[i]);
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
