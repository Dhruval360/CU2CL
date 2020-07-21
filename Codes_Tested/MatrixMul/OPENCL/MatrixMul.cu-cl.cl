



#define BLOCK_SIZE 16

/*
*********************************************************************
function name: gpu_matrix_mult
description: dot product of two matrix (not only square)
parameters: 
            &a GPU device pointer to a m X n matrix (A)
            &b GPU device pointer to a n X k matrix (B)
            &c GPU device output purpose pointer to a m X k matrix (C) 
            to store the result
Note:
    grid and block should be configured as:
        dim3 dimGrid((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    further sppedup can be obtained by using shared memory to decrease global memory access times
return: none
*********************************************************************
*/
__kernel void gpu_matrix_mult(__global int *a,__global int *b, __global int *c, int m, int n, int k)
{ 
    int row = get_group_id(1) * get_local_size(1) + get_local_id(1); 
    int col = get_group_id(0) * get_local_size(0) + get_local_id(0);
    int sum = 0;
    if( col < k && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
} 

/*
*********************************************************************
function name: gpu_square_matrix_mult
description: dot product of two matrix (not only square) in GPU
parameters: 
            &a GPU device pointer to a n X n matrix (A)
            &b GPU device pointer to a n X n matrix (B)
            &c GPU device output purpose pointer to a n X n matrix (C) 
            to store the result
Note:
    grid and block should be configured as:
        dim3 dim_grid((n - 1) / BLOCK_SIZE + 1, (n - 1) / BLOCK_SIZE + 1, 1);
        dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);
return: none
*********************************************************************
*/
__kernel void gpu_square_matrix_mult(__global int *d_a, __global int *d_b, __global int *d_result, int n) 
{
    __local int tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __local int tile_b[BLOCK_SIZE][BLOCK_SIZE];

    int row = get_group_id(1) * BLOCK_SIZE + get_local_id(1);
    int col = get_group_id(0) * BLOCK_SIZE + get_local_id(0);
    int tmp = 0;
    int idx;

    for (int sub = 0; sub < get_num_groups(0); ++sub) 
    {
        idx = row * n + sub * BLOCK_SIZE + get_local_id(0);
        if(idx >= n*n)
        {
            // n may not divisible by BLOCK_SIZE
            tile_a[get_local_id(1)][get_local_id(0)] = 0;
        }
        else
        {
            tile_a[get_local_id(1)][get_local_id(0)] = d_a[idx];
        }

        idx = (sub * BLOCK_SIZE + get_local_id(1)) * n + col;
        if(idx >= n*n)
        {
            tile_b[get_local_id(1)][get_local_id(0)] = 0;
        }  
        else
        {
            tile_b[get_local_id(1)][get_local_id(0)] = d_b[idx];
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

        for (int k = 0; k < BLOCK_SIZE; ++k) 
        {
            tmp += tile_a[get_local_id(1)][k] * tile_b[k][get_local_id(0)];
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
    if(row < n && col < n)
    {
        d_result[row * n + col] = tmp;
    }
}

/*
*********************************************************************
function name: gpu_matrix_transpose
description: matrix transpose
parameters: 
            &mat_in GPU device pointer to a rows X cols matrix
            &mat_out GPU device output purpose pointer to a cols X rows matrix 
            to store the result
Note:
    grid and block should be configured as:
        dim3 dim_grid((n - 1) / BLOCK_SIZE + 1, (n - 1) / BLOCK_SIZE + 1, 1);
        dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);
return: none
*********************************************************************
*/
__kernel void gpu_matrix_transpose(__global int* mat_in, __global int* mat_out, unsigned int rows, unsigned int cols) 
{
    unsigned int idx = get_group_id(0) * get_local_size(0) + get_local_id(0);
    unsigned int idy = get_group_id(1) * get_local_size(1) + get_local_id(1);

    if (idx < cols && idy < rows) 
    {
        unsigned int pos = idy * cols + idx;
        unsigned int trans_pos = idx * rows + idy;
        mat_out[trans_pos] = mat_in[pos];
    }
}
/*
*********************************************************************
function name: cpu_matrix_mult
description: dot product of two matrix (not only square) in CPU, 
             for validating GPU results
parameters: 
            &a CPU host pointer to a m X n matrix (A)
            &b CPU host pointer to a n X k matrix (B)
            &c CPU host output purpose pointer to a m X k matrix (C) 
            to store the result
return: none
*********************************************************************
*/


/*
*********************************************************************
function name: main
description: test and compare
parameters: 
            none
return: none
*********************************************************************
*/
