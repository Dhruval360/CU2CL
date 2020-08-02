/* (C) 2010-2017 Virginia Polytechnic Institute & State University (also known as "Virginia Tech"). All Rights Reserved.
/* This software is provided as-is.  Neither the authors, Virginia Tech nor Virginia Tech Intellectual Properties, Inc. assert, warrant, or guarantee that the software is fit for any purpose whatsoever, nor do they collectively or individually accept any responsibility or liability for any action or activity that results from the use of this software.  The entire risk as to the quality and performance of the software rests with the user, and no remedies shall be provided by the authors, Virginia Tech or Virginia Tech Intellectual Properties, Inc.
*
*    This library is free software; you can redistribute it and/or modify it under the terms of the attached GNU Lesser General Public License v2.1 as published by the Free Software Foundation.
*
*    This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
*
*   You should have received a copy of the GNU Lesser General Public License along with this library; if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA 
*/
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif
void __cu2cl_Init();

void __cu2cl_Cleanup();
size_t __cu2cl_LoadProgramSource(const char *filename, const char **progSrc);

struct __cu2cl_DeviceProp {
    char name[256];
    cl_ulong totalGlobalMem;
    cl_ulong sharedMemPerBlock;
    cl_uint regsPerBlock;
    cl_uint warpSize;
    size_t memPitch; //Unsupported!
    size_t maxThreadsPerBlock;
    size_t maxThreadsDim[3];
    int maxGridSize[3]; //Unsupported!
    cl_uint clockRate;
    size_t totalConstMem; //Unsupported!
    cl_uint major;
    cl_uint minor;
    size_t textureAlignment; //Unsupported!
    cl_bool deviceOverlap;
    cl_uint multiProcessorCount;
    cl_bool kernelExecTimeoutEnabled;
    cl_bool integrated;
    int canMapHostMemory; //Unsupported!
    int computeMode; //Unsupported!
    int maxTexture1D; //Unsupported!
    int maxTexture2D[2]; //Unsupported!
    int maxTexture3D[3]; //Unsupported!
    int maxTexture2DArray[3]; //Unsupported!
    size_t surfaceAlignment; //Unsupported!
    int concurrentKernels; //Unsupported!
    cl_bool ECCEnabled;
    int pciBusID; //Unsupported!
    int pciDeviceID; //Unsupported!
    int tccDriver; //Unsupported!
    //int __cudaReserved[21];
};


cl_int __cu2cl_GetDeviceProperties(struct __cu2cl_DeviceProp * prop, cl_device_id device);

const char *getErrorString(cl_int error);

void __cu2cl_Init_simpleZeroCopy_cu();

void __cu2cl_Cleanup_simpleZeroCopy_cu();


#ifdef __cplusplus
}
#endif
#include <vector>

#define char1 cl_char
#define uchar1 cl_uchar
#define short1 cl_short
#define ushort1 cl_ushort
#define int1 cl_int
#define uint1 cl_uint
#define long1 cl_long
#define ulong1 cl_ulong
#define float1 cl_float
#define char2 cl_char2
#define uchar2 cl_uchar2
#define short2 cl_short2
#define ushort2 cl_ushort2
#define int2 cl_int2
#define uint2 cl_uint2
#define long2 cl_long2
#define ulong2 cl_ulong2
#define float2 cl_float2
#define char3 cl_char3
#define uchar3 cl_uchar3
#define short3 cl_short3
#define ushort3 cl_ushort3
#define int3 cl_int3
#define uint3 cl_uint3
#define long3 cl_long3
#define ulong3 cl_ulong3
#define float3 cl_float3
#define char4 cl_char4
#define uchar4 cl_uchar4
#define short4 cl_short4
#define ushort4 cl_ushort4
#define int4 cl_int4
#define uint4 cl_uint4
#define long4 cl_long4
#define ulong4 cl_ulong4
#define float4 cl_float4
#define longlong1 cl_long
#define ulonglong1 cl_ulong
#define double1 cl_double
#define longlong2 cl_long2
#define ulonglong2 cl_ulong2
#define double2 cl_double2
