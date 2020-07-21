



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




// two types are shwon below

__kernel void knapsackGPU(__global int* dp, int row, __global int* d_value, __global int* d_weight,int capacity)
{
	int in = get_local_id(0) + (get_local_size(0) * get_group_id(0));
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


__kernel void knapsackGPU2(__global int* dp, __global int* d_value, __global int* d_weight, int capacity,int n)
{
	int in = get_local_id(0) + (get_local_size(0) * get_group_id(0));
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




