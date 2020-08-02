



#define imin(a,b) (a<b?a:b)
#define  N  33 * 1024
#define threadsPerBlock  256
#define blocksPerGrid  imin(32, (N+threadsPerBlock-1) / threadsPerBlock)

//const int N = 33 * 1024;
//const int threadsPerBlock = 256;
//const int blocksPerGrid = imin(32, (N+threadsPerBlock-1) / threadsPerBlock);

__kernel void dotProd(__global float* a, __global float* b, __global float* c) {
	__local float cache[threadsPerBlock];
	int tid = get_local_id(0) + get_group_id(0) * get_local_size(0);
	int cacheIndex = get_local_id(0);
	
	float temp = 0;
	while (tid < N){
		temp += a[tid] * b[tid];
		tid += get_local_size(0) * get_num_groups(0);
	}
	
	// set the cache values
	cache[cacheIndex] = temp;
	
	// synchronize threads in this block
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	
	// for reductions, threadsPerBlock must be a power of 2
	// because of the following code
	int i = get_local_size(0)/2;
	while (i != 0){
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i];
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
		i /= 2;
	}
	
	if (cacheIndex == 0)
		c[get_group_id(0)] = cache[0];
}



