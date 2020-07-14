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

void __cu2cl_Init_simpleZeroCopy_cu();

void __cu2cl_Cleanup_simpleZeroCopy_cu();


#ifdef __cplusplus
}
#endif
