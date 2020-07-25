#include <vector>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include "cu2cl_util.h"




cl_kernel __cu2cl_Kernel_knapsackGPU;
cl_kernel __cu2cl_Kernel_knapsackGPU2;
cl_program __cu2cl_Program_knapsack_cu;
cl_int err;
extern const char *progSrc;
extern size_t progLen;

extern cl_platform_id __cu2cl_Platform;
extern cl_device_id __cu2cl_Device;
extern cl_context __cu2cl_Context;
extern cl_command_queue __cu2cl_CommandQueue;

extern size_t globalWorkSize[3];
extern size_t localWorkSize[3];
void __cu2cl_Cleanup_knapsack_cu() {
    clReleaseKernel(__cu2cl_Kernel_knapsackGPU);
    clReleaseKernel(__cu2cl_Kernel_knapsackGPU2);
    clReleaseProgram(__cu2cl_Program_knapsack_cu);
}
void __cu2cl_Init_knapsack_cu() {
    #ifdef WITH_ALTERA
    progLen = __cu2cl_LoadProgramSource("knapsack_cu_cl.aocx", &progSrc);
    __cu2cl_Program_knapsack_cu = clCreateProgramWithBinary(__cu2cl_Context, 1, &__cu2cl_Device, &progLen, (const unsigned char **)&progSrc, NULL, &err);
    //printf("clCreateProgramWithBinary for knapsack.cu-cl.cl: %s\n", getErrorString(err));
    #else
    progLen = __cu2cl_LoadProgramSource("knapsack.cu-cl.cl", &progSrc);
    __cu2cl_Program_knapsack_cu = clCreateProgramWithSource(__cu2cl_Context, 1, &progSrc, &progLen, &err);
    //printf("clCreateProgramWithSource for knapsack.cu-cl.cl: %s\n", getErrorString(err));
    #endif
    free((void *) progSrc);
    err = clBuildProgram(__cu2cl_Program_knapsack_cu, 1, &__cu2cl_Device, "-I . ", NULL, NULL);
    /*printf("clBuildProgram : %s\n", getErrorString(err)); //Uncomment this line to access the error string of the error code returned by clBuildProgram*/
    if(err != CL_SUCCESS){
        std::vector<char> buildLog;
        size_t logSize;
        err = clGetProgramBuildInfo(__cu2cl_Program_knapsack_cu, __cu2cl_Device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        printf("clGetProgramBuildInfo : %s\n", getErrorString(err));
        buildLog.resize(logSize);
        clGetProgramBuildInfo(__cu2cl_Program_knapsack_cu, __cu2cl_Device, CL_PROGRAM_BUILD_LOG, logSize, &buildLog[0], NULL);
        printf("%s\n", &buildLog[0]);
    }
    __cu2cl_Kernel_knapsackGPU = clCreateKernel(__cu2cl_Program_knapsack_cu, "knapsackGPU", &err);
    /*printf("__cu2cl_Kernel_knapsackGPU creation: %s\n", getErrorString(err)); // Uncomment this line to get error string for the error code returned by clCreateKernel while creating the Kernel: knapsackGPU*/
    __cu2cl_Kernel_knapsackGPU2 = clCreateKernel(__cu2cl_Program_knapsack_cu, "knapsackGPU2", &err);
    /*printf("__cu2cl_Kernel_knapsackGPU2 creation: %s\n", getErrorString(err)); // Uncomment this line to get error string for the error code returned by clCreateKernel while creating the Kernel: knapsackGPU2*/
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


// two types are shown below







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
//printf("clCreateBuffer for device variable *(void**)&d_value: %s\n", getErrorString(err));
	*(void**)&d_weight = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, n * sizeof(int), NULL, &err);
//printf("clCreateBuffer for device variable *(void**)&d_weight: %s\n", getErrorString(err));
	cl_mem dp;
	*(void**)&dp = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, (n + 1) * (capacity + 1) * sizeof(int), NULL, &err);
//printf("clCreateBuffer for device variable *(void**)&dp: %s\n", getErrorString(err));

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
/*printf("clSetKernelArg for argument 0 of kernel __cu2cl_Kernel_knapsackGPU2: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_knapsackGPU2, 1, sizeof(cl_mem), &d_value);
/*printf("clSetKernelArg for argument 1 of kernel __cu2cl_Kernel_knapsackGPU2: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_knapsackGPU2, 2, sizeof(cl_mem), &d_weight);
/*printf("clSetKernelArg for argument 2 of kernel __cu2cl_Kernel_knapsackGPU2: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_knapsackGPU2, 3, sizeof(int), &capacity);
/*printf("clSetKernelArg for argument 3 of kernel __cu2cl_Kernel_knapsackGPU2: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_knapsackGPU2, 4, sizeof(int), &n);
/*printf("clSetKernelArg for argument 4 of kernel __cu2cl_Kernel_knapsackGPU2: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
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
