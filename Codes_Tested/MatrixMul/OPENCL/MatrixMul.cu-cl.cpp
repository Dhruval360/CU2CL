#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include "cu2cl_util.h"


cl_mem __cu2cl_Mem_h_a;
cl_mem __cu2cl_Mem_h_b;
cl_mem __cu2cl_Mem_h_c;
cl_mem __cu2cl_Mem_h_cc;


cl_kernel __cu2cl_Kernel_gpu_matrix_mult;
cl_kernel __cu2cl_Kernel_gpu_square_matrix_mult;
cl_kernel __cu2cl_Kernel_gpu_matrix_transpose;
cl_program __cu2cl_Program_MatrixMul_cu;
cl_int err;
extern const char *progSrc;
extern size_t progLen;

extern cl_platform_id __cu2cl_Platform;
extern cl_device_id __cu2cl_Device;
extern cl_context __cu2cl_Context;
extern cl_command_queue __cu2cl_CommandQueue;

extern size_t globalWorkSize[3];
extern size_t localWorkSize[3];
void __cu2cl_Cleanup_MatrixMul_cu() {
    clReleaseKernel(__cu2cl_Kernel_gpu_matrix_mult);
    clReleaseKernel(__cu2cl_Kernel_gpu_square_matrix_mult);
    clReleaseKernel(__cu2cl_Kernel_gpu_matrix_transpose);
    clReleaseProgram(__cu2cl_Program_MatrixMul_cu);
}
void __cu2cl_Init_MatrixMul_cu() {
    #ifdef WITH_ALTERA
    progLen = __cu2cl_LoadProgramSource("MatrixMul_cu_cl.aocx", &progSrc);
    __cu2cl_Program_MatrixMul_cu = clCreateProgramWithBinary(__cu2cl_Context, 1, &__cu2cl_Device, &progLen, (const unsigned char **)&progSrc, NULL, &err);
    //printf("clCreateProgramWithBinary for MatrixMul.cu-cl.cl: %s\n", getErrorString(err));
    #else
    progLen = __cu2cl_LoadProgramSource("MatrixMul.cu-cl.cl", &progSrc);
    __cu2cl_Program_MatrixMul_cu = clCreateProgramWithSource(__cu2cl_Context, 1, &progSrc, &progLen, &err);
    //printf("clCreateProgramWithSource for MatrixMul.cu-cl.cl: %s\n", getErrorString(err));
    #endif
    free((void *) progSrc);
    err = clBuildProgram(__cu2cl_Program_MatrixMul_cu, 1, &__cu2cl_Device, "-I . ", NULL, NULL);
    /*printf("clBuildProgram : %s\n", getErrorString(err)); //Uncomment this line to access the error string of the error code returned by clBuildProgram*/
    if(err != CL_SUCCESS){
        std::vector<char> buildLog;
        size_t logSize;
        err = clGetProgramBuildInfo(__cu2cl_Program_MatrixMul_cu, __cu2cl_Device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        printf("clGetProgramBuildInfo : %s\n", getErrorString(err));
        buildLog.resize(logSize);
        clGetProgramBuildInfo(__cu2cl_Program_MatrixMul_cu, __cu2cl_Device, CL_PROGRAM_BUILD_LOG, logSize, &buildLog[0], NULL);
        printf("%s\n", &buildLog[0]);
    }
    __cu2cl_Kernel_gpu_matrix_mult = clCreateKernel(__cu2cl_Program_MatrixMul_cu, "gpu_matrix_mult", &err);
    /*printf("__cu2cl_Kernel_gpu_matrix_mult creation: %s\n", getErrorString(err)); // Uncomment this line to get error string for the error code returned by clCreateKernel while creating the Kernel: gpu_matrix_mult*/
    __cu2cl_Kernel_gpu_square_matrix_mult = clCreateKernel(__cu2cl_Program_MatrixMul_cu, "gpu_square_matrix_mult", &err);
    /*printf("__cu2cl_Kernel_gpu_square_matrix_mult creation: %s\n", getErrorString(err)); // Uncomment this line to get error string for the error code returned by clCreateKernel while creating the Kernel: gpu_square_matrix_mult*/
    __cu2cl_Kernel_gpu_matrix_transpose = clCreateKernel(__cu2cl_Program_MatrixMul_cu, "gpu_matrix_transpose", &err);
    /*printf("__cu2cl_Kernel_gpu_matrix_transpose creation: %s\n", getErrorString(err)); // Uncomment this line to get error string for the error code returned by clCreateKernel while creating the Kernel: gpu_matrix_transpose*/
}

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

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
void cpu_matrix_mult(int *h_a, int *h_b, int *h_result, int m, int n, int k) {
    for (int i = 0; i < m; ++i) 
    {
        for (int j = 0; j < k; ++j) 
        {
            int tmp = 0.0;
            for (int h = 0; h < n; ++h) 
            {
                tmp += h_a[i * n + h] * h_b[h * k + j];
            }
            h_result[i * k + j] = tmp;
        }
    }
}

/*
*********************************************************************
function name: main
description: test and compare
parameters: 
            none
return: none
*********************************************************************
*/
int main(int argc, char const *argv[])
{
__cu2cl_Init();

    int m, n, k;
    /* Fixed seed for illustration */
    srand(3333);
    printf("please type in m n and k\n");
    scanf("%d %d %d", &m, &n, &k);

    // allocate memory in host RAM, h_cc is used to store CPU result
    int *h_a, *h_b, *h_c, *h_cc;
    __cu2cl_MallocHost((void **) &h_a, sizeof(int)*m*n, &__cu2cl_Mem_h_a);
    __cu2cl_MallocHost((void **) &h_b, sizeof(int)*n*k, &__cu2cl_Mem_h_b);
    __cu2cl_MallocHost((void **) &h_c, sizeof(int)*m*k, &__cu2cl_Mem_h_c);
    __cu2cl_MallocHost((void **) &h_cc, sizeof(int)*m*k, &__cu2cl_Mem_h_cc);

    // random initialize matrix A
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            h_a[i * n + j] = rand() % 1024;
        }
    }

    // random initialize matrix B
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            h_b[i * k + j] = rand() % 1024;
        }
    }

    float gpu_elapsed_time_ms, cpu_elapsed_time_ms;

    // some events to count the execution time
    cl_event start, stop;
    start = clCreateUserEvent(__cu2cl_Context, &err);
//printf("clCreateUserEvent for the event start: %s\n", getErrorString(err));
    stop = clCreateUserEvent(__cu2cl_Context, &err);
//printf("clCreateUserEvent for the event stop: %s\n", getErrorString(err));

    // start to count execution time of GPU version
    err = clEnqueueMarkerWithWaitList(__cu2cl_CommandQueue, 0, 0, &start);
//printf("clEnqueMarkerWithWaitList for the event start: %s\n", getErrorString(err));
    // Allocate memory space on the device 
    cl_mem d_a;
cl_mem d_b;
cl_mem d_c;
    *(void **) &d_a = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(int)*m*n, NULL, &err);
//printf("clCreateBuffer for device variable *(void **) &d_a: %s\n", getErrorString(err));
    *(void **) &d_b = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(int)*n*k, NULL, &err);
//printf("clCreateBuffer for device variable *(void **) &d_b: %s\n", getErrorString(err));
    *(void **) &d_c = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(int)*m*k, NULL, &err);
//printf("clCreateBuffer for device variable *(void **) &d_c: %s\n", getErrorString(err));

    // copy matrix A and B from host to device memory
    err = clEnqueueWriteBuffer(__cu2cl_CommandQueue, d_a, CL_TRUE, 0, sizeof(int)*m*n, h_a, 0, NULL, NULL);
//printf("Memory copy from host variable h_a to device variable d_a: %s\n", getErrorString(err));
    err = clEnqueueWriteBuffer(__cu2cl_CommandQueue, d_b, CL_TRUE, 0, sizeof(int)*n*k, h_b, 0, NULL, NULL);
//printf("Memory copy from host variable h_b to device variable d_b: %s\n", getErrorString(err));

    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t dimGrid[3] = {grid_cols, grid_rows, 1};
    size_t dimBlock[3] = {BLOCK_SIZE, BLOCK_SIZE, 1};
   
    // Launch kernel 
    if(m == n && n == k)
    {
        err = clSetKernelArg(__cu2cl_Kernel_gpu_square_matrix_mult, 0, sizeof(cl_mem), &d_a);
/*printf("clSetKernelArg for argument 0 of kernel __cu2cl_Kernel_gpu_square_matrix_mult: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_gpu_square_matrix_mult, 1, sizeof(cl_mem), &d_b);
/*printf("clSetKernelArg for argument 1 of kernel __cu2cl_Kernel_gpu_square_matrix_mult: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_gpu_square_matrix_mult, 2, sizeof(cl_mem), &d_c);
/*printf("clSetKernelArg for argument 2 of kernel __cu2cl_Kernel_gpu_square_matrix_mult: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_gpu_square_matrix_mult, 3, sizeof(int), &n);
/*printf("clSetKernelArg for argument 3 of kernel __cu2cl_Kernel_gpu_square_matrix_mult: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
localWorkSize[0] = dimBlock[0];
localWorkSize[1] = dimBlock[1];
localWorkSize[2] = dimBlock[2];
globalWorkSize[0] = dimGrid[0]*localWorkSize[0];
globalWorkSize[1] = dimGrid[1]*localWorkSize[1];
globalWorkSize[2] = dimGrid[2]*localWorkSize[2];
err = clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_gpu_square_matrix_mult, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
//printf("clEnqueueNDRangeKernel for the kernel __cu2cl_Kernel_gpu_square_matrix_mult: %s\n", getErrorString(err));    
    }
    else
    {
        err = clSetKernelArg(__cu2cl_Kernel_gpu_matrix_mult, 0, sizeof(cl_mem), &d_a);
/*printf("clSetKernelArg for argument 0 of kernel __cu2cl_Kernel_gpu_matrix_mult: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_gpu_matrix_mult, 1, sizeof(cl_mem), &d_b);
/*printf("clSetKernelArg for argument 1 of kernel __cu2cl_Kernel_gpu_matrix_mult: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_gpu_matrix_mult, 2, sizeof(cl_mem), &d_c);
/*printf("clSetKernelArg for argument 2 of kernel __cu2cl_Kernel_gpu_matrix_mult: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_gpu_matrix_mult, 3, sizeof(int), &m);
/*printf("clSetKernelArg for argument 3 of kernel __cu2cl_Kernel_gpu_matrix_mult: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_gpu_matrix_mult, 4, sizeof(int), &n);
/*printf("clSetKernelArg for argument 4 of kernel __cu2cl_Kernel_gpu_matrix_mult: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_gpu_matrix_mult, 5, sizeof(int), &k);
/*printf("clSetKernelArg for argument 5 of kernel __cu2cl_Kernel_gpu_matrix_mult: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
localWorkSize[0] = dimBlock[0];
localWorkSize[1] = dimBlock[1];
localWorkSize[2] = dimBlock[2];
globalWorkSize[0] = dimGrid[0]*localWorkSize[0];
globalWorkSize[1] = dimGrid[1]*localWorkSize[1];
globalWorkSize[2] = dimGrid[2]*localWorkSize[2];
err = clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_gpu_matrix_mult, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
//printf("clEnqueueNDRangeKernel for the kernel __cu2cl_Kernel_gpu_matrix_mult: %s\n", getErrorString(err));    
    }
    // Transefr results from device to host 
    err = clEnqueueReadBuffer(__cu2cl_CommandQueue, d_c, CL_TRUE, 0, sizeof(int)*m*k, h_c, 0, NULL, NULL);
//printf("Memory copy from device variable h_c to host variable d_c: %s\n", getErrorString(err));
    err = clFinish(__cu2cl_CommandQueue);
//printf("clFinish return message = %s\n", getErrorString(err));
    // time counting terminate
    err = clEnqueueMarkerWithWaitList(__cu2cl_CommandQueue, 0, 0, &stop);
//printf("clEnqueMarkerWithWaitList for the event stop: %s\n", getErrorString(err));
    clWaitForEvents(1, &stop);

    // compute time elapse on GPU computing
    __cu2cl_EventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on GPU: %f ms.\n\n", m, n, n, k, gpu_elapsed_time_ms);

    // start the CPU version
    err = clEnqueueMarkerWithWaitList(__cu2cl_CommandQueue, 0, 0, &start);
//printf("clEnqueMarkerWithWaitList for the event start: %s\n", getErrorString(err));

    cpu_matrix_mult(h_a, h_b, h_cc, m, n, k);

    err = clEnqueueMarkerWithWaitList(__cu2cl_CommandQueue, 0, 0, &stop);
//printf("clEnqueMarkerWithWaitList for the event stop: %s\n", getErrorString(err));
    clWaitForEvents(1, &stop);
    __cu2cl_EventElapsedTime(&cpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on CPU: %f ms.\n\n", m, n, n, k, cpu_elapsed_time_ms);

    // validate results computed by GPU
    int all_ok = 1;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            //printf("[%d][%d]:%d == [%d][%d]:%d, ", i, j, h_cc[i*k + j], i, j, h_c[i*k + j]);
            if(h_cc[i*k + j] != h_c[i*k + j])
            {
                all_ok = 0;
            }
        }
        //printf("\n");
    }

    // roughly compute speedup
    if(all_ok)
    {
        printf("all results are correct!!!, speedup = %f\n", cpu_elapsed_time_ms / gpu_elapsed_time_ms);
    }
    else
    {
        printf("incorrect results\n");
    }

    // free memory
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    __cu2cl_FreeHost(h_a, __cu2cl_Mem_h_a);
    __cu2cl_FreeHost(h_b, __cu2cl_Mem_h_b);
    __cu2cl_FreeHost(h_c, __cu2cl_Mem_h_c);
    __cu2cl_FreeHost(h_cc, __cu2cl_Mem_h_cc);
    return 0;
__cu2cl_Cleanup();
}