#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include "cu2cl_util.h"




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

void __cu2cl_Init_knap_cu() {
    #ifdef WITH_ALTERA
    progLen = __cu2cl_LoadProgramSource("knap_cu_cl.aocx", &progSrc);
    __cu2cl_Program_knap_cu = clCreateProgramWithBinary(__cu2cl_Context, 1, &__cu2cl_Device, &progLen, (const unsigned char **)&progSrc, NULL, &err);
    #else
    progLen = __cu2cl_LoadProgramSource("knap.cu-cl.cl", &progSrc);
    __cu2cl_Program_knap_cu = clCreateProgramWithSource(__cu2cl_Context, 1, &progSrc, &progLen, &err);
    //printf("clCreateProgramWithSource for knap.cu-cl.cl: %s\n", getErrorString(err));    #endif
    free((void *) progSrc);
    err = clBuildProgram(__cu2cl_Program_knap_cu, 1, &__cu2cl_Device, "-I . ", NULL, NULL);
    /*printf("clBuildProgram : %s\n", getErrorString(err)); //Uncomment this line to access the error string of the error code returned by clBuildProgram*/
    if(err != CL_SUCCESS){
        std::vector<char> buildLog;
        size_t logSize;
        err = clGetProgramBuildInfo(__cu2cl_Program_knap_cu, &__cu2cl_Device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        printf("clGetProgramBuildInfo : %s\n", getErrorString(err));
        buildLog.resize(logSize);
        clGetProgramBuildInfo(__cu2cl_Program_knap_cu, __cu2cl_Device, CL_PROGRAM_BUILD_LOG, logSize, &buildLog[0], nullptr);
        std::cout << &buildLog[0] << '\n';
    }
    __cu2cl_Kernel_knapsackGPU = clCreateKernel(__cu2cl_Program_knap_cu, "knapsackGPU", &err);
    /*printf("__cu2cl_Kernel_knapsackGPU creation: %s
", getErrorString(err)); // Uncomment this line to get error string for the error code returned by clCreateKernel while creating the Kernel: knapsackGPU*/
    __cu2cl_Kernel_knapsackGPU2 = clCreateKernel(__cu2cl_Program_knap_cu, "knapsackGPU2", &err);
    /*printf("__cu2cl_Kernel_knapsackGPU2 creation: %s
", getErrorString(err)); // Uncomment this line to get error string for the error code returned by clCreateKernel while creating the Kernel: knapsackGPU2*/
}

cl_kernel __cu2cl_Kernel_knapsackGPU;
cl_kernel __cu2cl_Kernel_knapsackGPU2;
cl_program __cu2cl_Program_knap_cu;
extern cl_kernel __cu2cl_Kernel_knapsackGPU;
extern cl_kernel __cu2cl_Kernel_knapsackGPU2;
extern cl_program __cu2cl_Program_knapsack_cu;
extern const char *progSrc;
extern size_t progLen;

extern cl_platform_id __cu2cl_Platform;
extern cl_device_id __cu2cl_Device;
extern cl_context __cu2cl_Context;
extern cl_command_queue __cu2cl_CommandQueue;

extern size_t globalWorkSize[3];
extern size_t localWorkSize[3];
void __cu2cl_Cleanup_knap_cu() {
    clReleaseKernel(__cu2cl_Kernel_knapsackGPU);
    clReleaseKernel(__cu2cl_Kernel_knapsackGPU2);
    clReleaseProgram(__cu2cl_Program_knap_cu);
}

#include<time.h>
#include <stdio.h>
#include<iostream>
#define N 10;

// CPU : 0.001s
// GPU : 0.00001 s

/*void knapSack(int value[], int weight[], int capacity, int n)
{
	//int dp[n + 1][capacity + 1];
	int* dp = (int*)malloc(sizeof(int)*(n+1)*(capacity+1));


	for (int i = 0; i <= capacity; i++)
		dp[i*(capacity+1)] = 0;
	for (int i = 0; i <= n; i++)
		dp[i] = 0;

	for (int i = 1; i <= n; i++)
	{
		for (int j = 1; j <= capacity; j++)
		{
			if (j >= weight[i - 1])
				dp[i*(capacity+1)+j] = dp[(i-1) * (capacity + 1) + j] < (value[i - 1] + dp[(i - 1)*(capacity+1) + j - weight[i - 1]]) ? (value[i - 1] + dp[(i - 1) * (capacity + 1) + j - weight[i - 1]]) : dp[(i - 1) * (capacity + 1) + j];
			else
				dp[i * (capacity + 1) + j] = dp[(i - 1) * (capacity + 1) + j];
			std::cout << dp[i * (capacity + 1) + j] << std::endl;
		}
	}
	std::cout << dp[capacity + n * (capacity + 1)] << std::endl;
	free(dp);dp = NULL;
}*/

void knapSack(int value[], int weight[], int capacity, int n)
{
	int** dp = new int* [n+1];
	for (int i = 0; i <=n; i++)
		dp[i] = new int[capacity+1];
	//int dp[n + 1][capacity + 1];

	for (int i = 0; i <= capacity; i++)
		dp[0][i] = 0;
	for (int i = 0; i <= n; i++)
		dp[i][0] = 0;

	for (int i = 1; i <= n; i++)
	{
		for (int j = 1; j <= capacity; j++)
		{
			if (j >= weight[i - 1])
				dp[i][j] = dp[i - 1][j] < (value[i - 1] + dp[i - 1][j - weight[i - 1]]) ? (value[i - 1] + dp[i - 1][j - weight[i - 1]]) : dp[i - 1][j];
			else
				dp[i][j] = dp[i - 1][j];
		}
	}
	/*DEBUGGING
	for (int i = 0; i <= n; i++)
	{
		for (int j = 0; j <= capacity; j++)
		{
			std::cout << dp[i][j] << " ";
		}
		std::cout << std::endl;
	}*/
	std::cout << dp[n][capacity] <<" is the maximum value from CPU\n" ;
	for (int i = 0; i <=n; i++)
		delete[] dp[i];
	delete[] dp;
}


// two types are shwon below







int main()
{
__cu2cl_Init();

	int val[] = { 60, 100, 120 };
	int wt[] = { 10, 20, 30 };
	int capacity = 50;
	int n = sizeof(val) / sizeof(val[0]);
	//knapSack(val, wt, capacity, n);

	cl_mem d_value;
cl_mem d_weight;
	*(void**)&d_value = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, n * sizeof(int), NULL, &err);
//printf("clCreateBuffer for device variable *(void**)&d_value is: %s\n", getErrorString(err));
	*(void**)&d_weight = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, n * sizeof(int), NULL, &err);
//printf("clCreateBuffer for device variable *(void**)&d_weight is: %s\n", getErrorString(err));
	cl_mem dp;
	*(void**)&dp = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, (n + 1) * (capacity + 1) * sizeof(int), NULL, &err);
//printf("clCreateBuffer for device variable *(void**)&dp is: %s\n", getErrorString(err));

	err = clEnqueueWriteBuffer(__cu2cl_CommandQueue, d_value, CL_TRUE, 0, (n) * sizeof(int), val, 0, NULL, NULL);
//printf("Memory copy from host variable val to device variable d_value: %s\n", getErrorString(err));
	err = clEnqueueWriteBuffer(__cu2cl_CommandQueue, d_weight, CL_TRUE, 0, (n) * sizeof(int), wt, 0, NULL, NULL);
//printf("Memory copy from host variable wt to device variable d_weight: %s\n", getErrorString(err));


	//dim3 block((capacity / N),1,1);
	//dim3 thread(N, 1, 1);
	int block = 1 + (capacity+1) / N;
	int thread = N;
	
	clock_t start, end;
	start = clock();
	knapSack(val, wt, capacity, n);
	end = clock();
	double time = ((double)end - (double)start) / CLOCKS_PER_SEC;

	printf("%f is the time taken by the CPU\n",time);



	start = clock();
	err = clSetKernelArg(__cu2cl_Kernel_knapsackGPU2, 0, sizeof(cl_mem), &dp);
/*printf("clSetKernelArg for argument 0 of kernel __cu2cl_Kernel_knapsackGPU2 is: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_knapsackGPU2, 1, sizeof(cl_mem), &d_value);
/*printf("clSetKernelArg for argument 1 of kernel __cu2cl_Kernel_knapsackGPU2 is: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_knapsackGPU2, 2, sizeof(cl_mem), &d_weight);
/*printf("clSetKernelArg for argument 2 of kernel __cu2cl_Kernel_knapsackGPU2 is: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_knapsackGPU2, 3, sizeof(int), &capacity);
/*printf("clSetKernelArg for argument 3 of kernel __cu2cl_Kernel_knapsackGPU2 is: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_knapsackGPU2, 4, sizeof(int), &n);
/*printf("clSetKernelArg for argument 4 of kernel __cu2cl_Kernel_knapsackGPU2 is: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
localWorkSize[0] = thread;
globalWorkSize[0] = (block)*localWorkSize[0];
err = clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_knapsackGPU2, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
//printf("clEnqueueNDRangeKernel for the kernel __cu2cl_Kernel_knapsackGPU2: %s\n", getErrorString(err));
	end = clock();
	time = ((double)end - (double)start) / CLOCKS_PER_SEC;

	printf("%f is the time taken by the GPU\n", time);
	// this was one method
	/*for (int i = 0;i <= n;i++)
	{
		knapsackGPU <<<block,thread>>> (dp, i, d_value, d_weight,capacity);
		cudaDeviceSynchronize();
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			printf("CUDA Error: %s\n", cudaGetErrorString(err));
			exit(1);
		}
	}*/

	int *h_dp;
	h_dp = (int*)malloc(sizeof(int)*(n+1)*(capacity+1));
	err = clEnqueueReadBuffer(__cu2cl_CommandQueue, dp, CL_TRUE, 0, (n + 1) * (capacity + 1) * sizeof(int), h_dp, 0, NULL, NULL);
//printf("Memory copy from device variable h_dp to host variable dp: %s\n", getErrorString(err));

	printf("%d is the maximum value\n", h_dp[capacity + n*(capacity+1)]);
	
	/*for (int i = 0; i <= n; i++)
	{
		for (int j = 0; j <= capacity; j++)
		{
			std::cout << h_dp[j+i*(capacity+1)] << " ";
		}
		std::cout << std::endl;
	}*/

	return 0;
__cu2cl_Cleanup();
}

