////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////


// System includes



// CUDA runtime


// helper functions and utilities to work with CUDA



#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif


/* Add two vectors on the GPU */
__kernel void vectorAddGPU(__global float *a, __global float *b, __global float *c, int N)
{
    int idx = get_group_id(0)*get_local_size(0) + get_local_id(0);

    if (idx < N)
    {
        c[idx] = a[idx] + b[idx];
    }
}

// Allocate generic memory with malloc() and pin it laster instead of using cudaHostAlloc()


// Macro to aligned up to the memory size in question
#define MEMORY_ALIGNMENT  4096
#define ALIGN_UP(x,size) ( ((size_t)x+(size-1))&(~(size-1)) )


