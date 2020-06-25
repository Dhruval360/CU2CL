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
    __cu2cl_Program_knapsack_cu = clCreateProgramWithBinary(__cu2cl_Context, 1, &__cu2cl_Device, &progLen, (const unsigned char **)&progSrc, NULL, NULL);
    #else
    progLen = __cu2cl_LoadProgramSource("knapsack.cu-cl.cl", &progSrc);
    __cu2cl_Program_knapsack_cu = clCreateProgramWithSource(__cu2cl_Context, 1, &progSrc, &progLen, NULL);
    #endif
    free((void *) progSrc);
    clBuildProgram(__cu2cl_Program_knapsack_cu, 1, &__cu2cl_Device, "-I . ", NULL, NULL);
    __cu2cl_Kernel_knapsackGPU = clCreateKernel(__cu2cl_Program_knapsack_cu, "knapsackGPU", NULL);
    __cu2cl_Kernel_knapsackGPU2 = clCreateKernel(__cu2cl_Program_knapsack_cu, "knapsackGPU2", NULL);
}



#include "device_launch_parameters.h-cl.h"
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
	*(void**)&d_value = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, n * sizeof(int), NULL, NULL);
	*(void**)&d_weight = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, n * sizeof(int), NULL, NULL);
	cl_mem dp;
	*(void**)&dp = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, (n + 1) * (capacity + 1) * sizeof(int), NULL, NULL);

	clEnqueueWriteBuffer(__cu2cl_CommandQueue, d_value, CL_TRUE, 0, (n) * sizeof(int), val, 0, NULL, NULL);
	clEnqueueWriteBuffer(__cu2cl_CommandQueue, d_weight, CL_TRUE, 0, (n) * sizeof(int), wt, 0, NULL, NULL);


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
	knapsackGPU2 << <block,thread>> > (dp,d_value,d_weight,capacity,n);
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
	clEnqueueReadBuffer(__cu2cl_CommandQueue, dp, CL_TRUE, 0, (n + 1) * (capacity + 1) * sizeof(int), h_dp, 0, NULL, NULL);

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
