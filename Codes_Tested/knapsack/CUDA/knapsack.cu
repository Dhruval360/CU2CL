#include "cuda_runtime.h"
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

__global__ void knapsackGPU(int* dp, int row, int* d_value, int* d_weight,int capacity)
{
	int in = threadIdx.x + (blockDim.x * blockIdx.x);
	if (row != 0)
	{
		int ind = in + (row * (capacity+1));
		if (in <= (capacity+1) && in > 0)
		{
			if (in >= d_weight[row - 1])
			{
				dp[ind] = dp[ind - (capacity+1)]> (d_value[row - 1] + dp[ind - (capacity + 1) - d_weight[row - 1]]) ? dp[ind - (capacity + 1)] : (d_value[row - 1] + dp[ind - (capacity + 1) - d_weight[row - 1]]);
			}
			else
				dp[ind] = dp[ind - (capacity+1)];
		}
		if (in == 0)
		{
			dp[ind] = 0;
		}
	}
	else
	{
		dp[in] = 0;
	}
}


__global__ void knapsackGPU2(int* dp, int* d_value, int* d_weight, int capacity,int n)
{
	int in = threadIdx.x + (blockDim.x * blockIdx.x);
	for (int row = 0;row <= n;row++)
	{
		if (row != 0)
		{
			int ind = in + (row * (capacity + 1));
			if (in <= (capacity + 1) && in > 0)
			{
				if (in >= d_weight[row - 1])
				{
					dp[ind] = dp[ind - (capacity + 1)] > (d_value[row - 1] + dp[ind - (capacity + 1) - d_weight[row - 1]]) ? dp[ind - (capacity + 1)] : (d_value[row - 1] + dp[ind - (capacity + 1) - d_weight[row - 1]]);
				}
				else
					dp[ind] = dp[ind - (capacity + 1)];
			}
			if (in == 0)
			{
				dp[ind] = 0;
			}
		}
		else
		{
			dp[in] = 0;
		}
	}
	
}


int main()
{
	int val[] = { 60, 100, 120 };
	int wt[] = { 10, 20, 30 };
	int capacity = 50;
	int n = sizeof(val) / sizeof(val[0]);
	//knapSack(val, wt, capacity, n);

	int* d_value, * d_weight;
	cudaMalloc((void**)&d_value, n * sizeof(int));
	cudaMalloc((void**)&d_weight, n * sizeof(int));
	int* dp;
	cudaMalloc((void**)&dp, (n + 1) * (capacity + 1) * sizeof(int));

	cudaMemcpy(d_value, val, (n) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_weight, wt, (n) * sizeof(int), cudaMemcpyHostToDevice);


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
	knapsackGPU2 <<<block,thread>>> (dp,d_value,d_weight,capacity,n);
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
	cudaMemcpy(h_dp, dp, (n + 1) * (capacity + 1) * sizeof(int), cudaMemcpyDeviceToHost);

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
}
