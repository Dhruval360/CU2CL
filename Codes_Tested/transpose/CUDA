#include "cuda.h"
#include <stdio.h>

const int N = 1024;		// The matrix used will be of size (N,N)
const int K = 32;		// This is for the tile size (tile size is (K,K))

// To compare the outputs obtained from the GPU with that of the CPU
int compare_matrices(float* gpu, float* ref)
{
	for (int i = 0; i < N * N; i++)
		if (gpu[i] != ref[i])
			return 1;
	return 0;
}

// To fill a matrix with sequential numbers in the range 0..N-1
void fill_matrix(float* mat)
{
	for (int j = 0; j < N * N; j++)
		mat[j] = (float)j;
}

// CPU code for the serial execution of matrix transpose
void transpose_CPU(float in[], float out[])
{
	for (int j = 0; j < N; j++)
		for (int i = 0; i < N; i++)
			out[j + i * N] = in[i + j * N]; 
}

// Serial function to be launched on a single thread of the GPU
__global__ void transpose_serial(float in[], float out[])
{
	for (int j = 0; j < N; j++)
		for (int i = 0; i < N; i++)
			out[j + i * N] = in[i + j * N]; 
}

// One thread per row of output matrix (unrolling by N)
__global__ void transpose_parallel_per_row(float in[], float out[])
{
	int i = threadIdx.x;

	for (int j = 0; j < N; j++)
		out[j + i * N] = in[i + j * N]; 
}

// One thread per element, in KxK threadblocks, thread (x,y) in grid writes element (i,j) of output matrix 
__global__ void transpose_parallel_per_element(float in[], float out[])
{
	int i = blockIdx.x * K + threadIdx.x;
	int j = blockIdx.y * K + threadIdx.y;

	out[j + i * N] = in[i + j * N]; 
}

// One thread per element, in (tilesize)x(tilesize) threadblocks
// thread blocks read & write tiles, in coalesced fashion
// adjacent threads read adjacent input elements, write adjacent output elements
__global__ void transpose_parallel_per_element_tiled(float in[], float out[])
{
	// (i,j) locations of the tile corners for input & output matrices:
	int in_corner_i = blockIdx.x * K, in_corner_j = blockIdx.y * K;
	int out_corner_i = blockIdx.y * K, out_corner_j = blockIdx.x * K;

	int x = threadIdx.x, y = threadIdx.y;

	__shared__ float tile[K][K];

	// Coalesced read from global memory and transposed write into shared memory
	tile[y][x] = in[(in_corner_i + x) + (in_corner_j + y) * N];
	__syncthreads();
	// Read from shared memory and coalesced write to global memory
	out[(out_corner_i + x) + (out_corner_j + y) * N] = tile[x][y];
}

// One thread per element, in (tilesize)x(tilesize) threadblocks
// thread blocks read & write tiles, in coalesced fashion
// adjacent threads read adjacent input elements, write adjacent output elmts
__global__ void transpose_parallel_per_element_tiled16(float in[], float out[])
{
	// (i,j) locations of the tile corners for input & output matrices:
	int in_corner_i = blockIdx.x * 16, in_corner_j = blockIdx.y * 16;
	int out_corner_i = blockIdx.y * 16, out_corner_j = blockIdx.x * 16;

	int x = threadIdx.x, y = threadIdx.y;

	__shared__ float tile[16][16];

	// Coalesced read from global memory and transposed write into shared memory
	tile[y][x] = in[(in_corner_i + x) + (in_corner_j + y) * N];
	__syncthreads();
	// Read from shared memory and coalesced write to global memory
	out[(out_corner_i + x) + (out_corner_j + y) * N] = tile[x][y];
}

// One thread per element, in KxK threadblocks
// thread blocks read & write tiles, in coalesced fashion
// shared memory array has been padded to avoid bank conflicts
__global__ void transpose_parallel_per_element_tiled_padded(float in[], float out[])
{
	// (i,j) locations of the tile corners for input & output matrices:
	int in_corner_i = blockIdx.x * K, in_corner_j = blockIdx.y * K;
	int out_corner_i = blockIdx.y * K, out_corner_j = blockIdx.x * K;

	int x = threadIdx.x, y = threadIdx.y;

	__shared__ float tile[K][K + 1];

	// Coalesced read from global memory and transposed write into shared memory
	tile[y][x] = in[(in_corner_i + x) + (in_corner_j + y) * N];
	__syncthreads();
	// Read from shared memory and coalesced write to global memory
	out[(out_corner_i + x) + (out_corner_j + y) * N] = tile[x][y];
}

// One thread per element, in 16x16 threadblocks
// thread blocks read & write tiles, in coalesced fashion
// shared memory array has been padded to avoid bank conflicts
__global__ void transpose_parallel_per_element_tiled_padded16(float in[], float out[])
{
	// (i,j) locations of the tile corners for input & output matrices:
	int in_corner_i = blockIdx.x * 16, in_corner_j = blockIdx.y * 16;
	int out_corner_i = blockIdx.y * 16, out_corner_j = blockIdx.x * 16;

	int x = threadIdx.x, y = threadIdx.y;

	__shared__ float tile[16][16 + 1];

	// Coalesced read from global memory and transposed write into shared memory
	tile[y][x] = in[(in_corner_i + x) + (in_corner_j + y) * N];
	__syncthreads();
	// Read from shared memory and coalesced write to global memory
	out[(out_corner_i + x) + (out_corner_j + y) * N] = tile[x][y];
}

int main()
{
	int numbytes = N * N * sizeof(float);

	float *in = (float*)malloc(numbytes);
	float *out = (float*)malloc(numbytes);
	float *reference_output = (float*)malloc(numbytes);

	fill_matrix(in); 

	printf("The matrix used is of size (%d, %d)\n", N, N);	

	clock_t start, end;
	double cpu_time_used, gpu_time_used;
     
     	start = clock();
	transpose_CPU(in, reference_output); // This creates the reference matrix (the expected output)
	end = clock();
	cpu_time_used = ((double) (end - start) * 1000) / CLOCKS_PER_SEC;
	printf("transpose_CPU                                                : Finished in %lf ms\n", cpu_time_used);

	float* d_in, * d_out;

	cudaMalloc(&d_in, numbytes);
	cudaMalloc(&d_out, numbytes);
	cudaMemcpy(d_in, in, numbytes, cudaMemcpyHostToDevice);

	// Transpose with each thread taking care of one row in the matrix
     	start = clock();
	transpose_parallel_per_row <<<1, N>>> (d_in, d_out);
	end = clock();
	gpu_time_used = ((double) (end - start) * 1000) / CLOCKS_PER_SEC;
	cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);	
	printf("transpose_parallel_per_row                                   : Finished in %lf ms and the output obtained is: %s\n", gpu_time_used, compare_matrices(out, reference_output) ? "Incorrect" : "Correct");

	
     	start = clock();
	transpose_serial <<<1, 1>>> (d_in, d_out);
	end = clock();
	gpu_time_used = ((double) (end - start) * 1000) / CLOCKS_PER_SEC;

	cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
	
	printf("transpose_serial                                             : Finished in %lf ms and the output obtained is: %s\n", gpu_time_used, compare_matrices(out, reference_output) ? "Incorrect" : "Correct");



	dim3 blocks(N / K, N / K); // blocks per grid
	dim3 threads(K, K);	// threads per block

	// Transpose with a thread for each element in the matrix
     	start = clock();
	transpose_parallel_per_element <<<blocks, threads>>> (d_in, d_out);
	end = clock();
	gpu_time_used = ((double) (end - start) * 1000) / CLOCKS_PER_SEC;
	cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
	printf("transpose_parallel_per_element                               : Finished in %lf ms and the output obtained is: %s\n", gpu_time_used, compare_matrices(out, reference_output) ? "Incorrect" : "Correct");


	// Tiled transpose with a thread for each element in the tile
     	start = clock();
	transpose_parallel_per_element_tiled <<<blocks, threads>>> (d_in, d_out);
	end = clock();
	gpu_time_used = ((double) (end - start) * 1000) / CLOCKS_PER_SEC;
	cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
	printf("transpose_parallel_per_element_tiled (32X32)                 : Finished in %lf ms and the output obtained is: %s\n", gpu_time_used, compare_matrices(out, reference_output) ? "Incorrect" : "Correct");



	dim3 blocks16x16(N / 16, N / 16); // blocks per grid
	dim3 threads16x16(16, 16);	 // threads per block


	// Tiled transpose with each element of the tile given to a thread
     	start = clock();
	transpose_parallel_per_element_tiled16 <<<blocks16x16, threads16x16>>> (d_in, d_out);
	end = clock();
	gpu_time_used = ((double) (end - start) * 1000) / CLOCKS_PER_SEC;
	cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
	printf("transpose_parallel_per_element_tiled (16X16)                 : Finished in %lf ms and the output obtained is: %s\n", gpu_time_used, compare_matrices(out, reference_output) ? "Incorrect" : "Correct");


	// Tiled transpose with padding
     	start = clock();
	transpose_parallel_per_element_tiled_padded16 <<<blocks16x16, threads16x16>>> (d_in, d_out);
	end = clock();
	gpu_time_used = ((double) (end - start) * 1000) / CLOCKS_PER_SEC;
	cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);	
	printf("transpose_parallel_per_element_tiled_padded (16X16)          : Finished in %lf ms and the output obtained is: %s\n", gpu_time_used, compare_matrices(out, reference_output) ? "Incorrect" : "Correct");


	cudaFree(d_in);
	cudaFree(d_out);
}

