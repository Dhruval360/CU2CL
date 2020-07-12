


		// The matrix used will be of size (N,N)
		// This is for the tile size (tile size is (K,K))

// To compare the outputs obtained from the GPU with that of the CPU


// To fill a matrix with sequential numbers in the range 0..N-1


// CPU code for the serial execution of matrix transpose


// Serial function to be launched on a single thread of the GPU
__kernel void transpose_serial(float in[], float out[])
{
	for (int j = 0; j < N; j++)
		for (int i = 0; i < N; i++)
			out[j + i * N] = in[i + j * N]; 
}

// One thread per row of output matrix (unrolling by N)
__kernel void transpose_parallel_per_row(float in[], float out[])
{
	int i = get_local_id(0);

	for (int j = 0; j < N; j++)
		out[j + i * N] = in[i + j * N]; 
}

// One thread per element, in KxK threadblocks, thread (x,y) in grid writes element (i,j) of output matrix 
__kernel void transpose_parallel_per_element(float in[], float out[])
{
	int i = get_group_id(0) * K + get_local_id(0);
	int j = get_group_id(1) * K + get_local_id(1);

	out[j + i * N] = in[i + j * N]; 
}

// One thread per element, in (tilesize)x(tilesize) threadblocks
// thread blocks read & write tiles, in coalesced fashion
// adjacent threads read adjacent input elements, write adjacent output elements
__kernel void transpose_parallel_per_element_tiled(float in[], float out[])
{
	// (i,j) locations of the tile corners for input & output matrices:
	int in_corner_i = get_group_id(0) * K, in_corner_j = get_group_id(1) * K;
	int out_corner_i = get_group_id(1) * K, out_corner_j = get_group_id(0) * K;

	int x = get_local_id(0), y = get_local_id(1);

	__local float tile[K][K];

	// Coalesced read from global memory and transposed write into shared memory
	tile[y][x] = in[(in_corner_i + x) + (in_corner_j + y) * N];
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	// Read from shared memory and coalesced write to global memory
	out[(out_corner_i + x) + (out_corner_j + y) * N] = tile[x][y];
}

// One thread per element, in (tilesize)x(tilesize) threadblocks
// thread blocks read & write tiles, in coalesced fashion
// adjacent threads read adjacent input elements, write adjacent output elmts
__kernel void transpose_parallel_per_element_tiled16(float in[], float out[])
{
	// (i,j) locations of the tile corners for input & output matrices:
	int in_corner_i = get_group_id(0) * 16, in_corner_j = get_group_id(1) * 16;
	int out_corner_i = get_group_id(1) * 16, out_corner_j = get_group_id(0) * 16;

	int x = get_local_id(0), y = get_local_id(1);

	__local float tile[16][16];

	// Coalesced read from global memory and transposed write into shared memory
	tile[y][x] = in[(in_corner_i + x) + (in_corner_j + y) * N];
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	// Read from shared memory and coalesced write to global memory
	out[(out_corner_i + x) + (out_corner_j + y) * N] = tile[x][y];
}

// One thread per element, in KxK threadblocks
// thread blocks read & write tiles, in coalesced fashion
// shared memory array has been padded to avoid bank conflicts
__kernel void transpose_parallel_per_element_tiled_padded(float in[], float out[])
{
	// (i,j) locations of the tile corners for input & output matrices:
	int in_corner_i = get_group_id(0) * K, in_corner_j = get_group_id(1) * K;
	int out_corner_i = get_group_id(1) * K, out_corner_j = get_group_id(0) * K;

	int x = get_local_id(0), y = get_local_id(1);

	__local float tile[K][K + 1];

	// Coalesced read from global memory and transposed write into shared memory
	tile[y][x] = in[(in_corner_i + x) + (in_corner_j + y) * N];
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	// Read from shared memory and coalesced write to global memory
	out[(out_corner_i + x) + (out_corner_j + y) * N] = tile[x][y];
}

// One thread per element, in 16x16 threadblocks
// thread blocks read & write tiles, in coalesced fashion
// shared memory array has been padded to avoid bank conflicts
__kernel void transpose_parallel_per_element_tiled_padded16(float in[], float out[])
{
	// (i,j) locations of the tile corners for input & output matrices:
	int in_corner_i = get_group_id(0) * 16, in_corner_j = get_group_id(1) * 16;
	int out_corner_i = get_group_id(1) * 16, out_corner_j = get_group_id(0) * 16;

	int x = get_local_id(0), y = get_local_id(1);

	__local float tile[16][16 + 1];

	// Coalesced read from global memory and transposed write into shared memory
	tile[y][x] = in[(in_corner_i + x) + (in_corner_j + y) * N];
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	// Read from shared memory and coalesced write to global memory
	out[(out_corner_i + x) + (out_corner_j + y) * N] = tile[x][y];
}



