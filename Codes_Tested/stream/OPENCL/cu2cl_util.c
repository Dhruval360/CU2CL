/* (C) 2010-2017 Virginia Polytechnic Institute & State University (also known as "Virginia Tech"). All Rights Reserved.
/* This software is provided as-is.  Neither the authors, Virginia Tech nor Virginia Tech Intellectual Properties, Inc. assert, warrant, or guarantee that the software is fit for any purpose whatsoever, nor do they collectively or individually accept any responsibility or liability for any action or activity that results from the use of this software.  The entire risk as to the quality and performance of the software rests with the user, and no remedies shall be provided by the authors, Virginia Tech or Virginia Tech Intellectual Properties, Inc.
*
*    This library is free software; you can redistribute it and/or modify it under the terms of the attached GNU Lesser General Public License v2.1 as published by the Free Software Foundation.
*
*    This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
*
*   You should have received a copy of the GNU Lesser General Public License along with this library; if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA 
*/
#include "cu2cl_util.h"
extern cl_kernel __cu2cl_Kernel_kernel;
extern cl_program __cu2cl_Program_stream_cu;
const char *progSrc;
size_t progLen;

cl_platform_id __cu2cl_Platform;
cl_device_id __cu2cl_Device;
cl_context __cu2cl_Context;
cl_command_queue __cu2cl_CommandQueue;

size_t globalWorkSize[3];
size_t localWorkSize[3];
size_t __cu2cl_LoadProgramSource(const char *filename, const char **progSrc) {
    FILE *f = fopen(filename, "r");
    fseek(f, 0, SEEK_END);
    size_t len = (size_t) ftell(f);
    *progSrc = (const char *) malloc(sizeof(char)*len);
    rewind(f);
    fread((void *) *progSrc, len, 1, f);
    fclose(f);
    return len;
}


void __cu2cl_Init() {
    clGetPlatformIDs(1, &__cu2cl_Platform, NULL);
    clGetDeviceIDs(__cu2cl_Platform, CL_DEVICE_TYPE_ALL, 1, &__cu2cl_Device, NULL);
    __cu2cl_Context = clCreateContext(NULL, 1, &__cu2cl_Device, NULL, NULL, NULL);
    __cu2cl_CommandQueue = clCreateCommandQueue(__cu2cl_Context, __cu2cl_Device, CL_QUEUE_PROFILING_ENABLE, NULL);
    __cu2cl_Init_stream_cu();
}

void __cu2cl_Cleanup() {
    __cu2cl_Cleanup_stream_cu();
    clReleaseCommandQueue(__cu2cl_CommandQueue);
    clReleaseContext(__cu2cl_Context);
}
