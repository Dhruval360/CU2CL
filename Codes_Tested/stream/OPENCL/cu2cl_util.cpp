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
extern cl_int err;
const char *progSrc;
size_t progLen;

cl_device_id * __cu2cl_AllDevices;
cl_uint __cu2cl_AllDevices_curr_idx;
cl_uint __cu2cl_AllDevices_size;
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


void __cu2cl_ScanDevices() {
   int i;
   cl_uint num_platforms = 0;
   cl_uint num_devices = 0;
   cl_uint p_dev_count, d_idx;

   //allocate space for platforms
   clGetPlatformIDs(0, 0, &num_platforms);
   cl_platform_id * platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id) * num_platforms);

   //get all platforms
   clGetPlatformIDs(num_platforms, &platforms[0], 0);

   //count devices over all platforms
   for (i = 0; i < num_platforms; i++) {
       p_dev_count = 0;
       clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, 0, &p_dev_count);
       num_devices += p_dev_count;
   }

   //allocate space for devices
   __cu2cl_AllDevices = (cl_device_id *) malloc(sizeof(cl_device_id) * num_devices);

   //get all devices
   d_idx = 0;
   for ( i = 0; i < num_platforms; i++) {
       clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices-d_idx, &__cu2cl_AllDevices[d_idx], &p_dev_count);
       d_idx += p_dev_count;
       p_dev_count = 0;
   }

   __cu2cl_AllDevices_size = d_idx;
   free(platforms);
}


void  __cu2cl_ResetDevice() {
       clReleaseCommandQueue(__cu2cl_CommandQueue);
       clReleaseContext(__cu2cl_Context);
}


const char *getErrorString(cl_int error){
	switch(error){
// run-time and JIT compiler errors
	case 0: return "CL_SUCCESS";
	case -1: return "CL_DEVICE_NOT_FOUND";
	case -2: return "CL_DEVICE_NOT_AVAILABLE";
	case -3: return "CL_COMPILER_NOT_AVAILABLE";
	case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	case -5: return "CL_OUT_OF_RESOURCES";
	case -6: return "CL_OUT_OF_HOST_MEMORY";
	case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
	case -8: return "CL_MEM_COPY_OVERLAP";
	case -9: return "CL_IMAGE_FORMAT_MISMATCH";
	case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case -11: return "CL_BUILD_PROGRAM_FAILURE";
	case -12: return "CL_MAP_FAILURE";
	case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
	case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
	case -15: return "CL_COMPILE_PROGRAM_FAILURE";
	case -16: return "CL_LINKER_NOT_AVAILABLE";
	case -17: return "CL_LINK_PROGRAM_FAILURE";
	case -18: return "CL_DEVICE_PARTITION_FAILED";
	case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
	// compile-time errors
	case -30: return "CL_INVALID_VALUE";
	case -31: return "CL_INVALID_DEVICE_TYPE";
	case -32: return "CL_INVALID_PLATFORM";
	case -33: return "CL_INVALID_DEVICE";
	case -34: return "CL_INVALID_CONTEXT";
	case -35: return "CL_INVALID_QUEUE_PROPERTIES";
	case -36: return "CL_INVALID_COMMAND_QUEUE";
	case -37: return "CL_INVALID_HOST_PTR";
	case -38: return "CL_INVALID_MEM_OBJECT";
	case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	case -40: return "CL_INVALID_IMAGE_SIZE";
	case -41: return "CL_INVALID_SAMPLER";
	case -42: return "CL_INVALID_BINARY";
	case -43: return "CL_INVALID_BUILD_OPTIONS";
	case -44: return "CL_INVALID_PROGRAM";
	case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
	case -46: return "CL_INVALID_KERNEL_NAME";
	case -47: return "CL_INVALID_KERNEL_DEFINITION";
	case -48: return "CL_INVALID_KERNEL";
	case -49: return "CL_INVALID_ARG_INDEX";
	case -50: return "CL_INVALID_ARG_VALUE";
	case -51: return "CL_INVALID_ARG_SIZE";
	case -52: return "CL_INVALID_KERNEL_ARGS";
	case -53: return "CL_INVALID_WORK_DIMENSION";
	case -54: return "CL_INVALID_WORK_GROUP_SIZE";
	case -55: return "CL_INVALID_WORK_ITEM_SIZE";
	case -56: return "CL_INVALID_GLOBAL_OFFSET";
	case -57: return "CL_INVALID_EVENT_WAIT_LIST";
	case -58: return "CL_INVALID_EVENT";
	case -59: return "CL_INVALID_OPERATION";
	case -60: return "CL_INVALID_GL_OBJECT";
	case -61: return "CL_INVALID_BUFFER_SIZE";
	case -62: return "CL_INVALID_MIP_LEVEL";
	case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
	case -64: return "CL_INVALID_PROPERTY";
	case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
	case -66: return "CL_INVALID_COMPILER_OPTIONS";
	case -67: return "CL_INVALID_LINKER_OPTIONS";
	case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
	// extension errors
	case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
	case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
	case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
	case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
	case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
	case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
	default: return "Unknown OpenCL error";
	}
}

void __cu2cl_Init() {
    cl_int err;
    clGetPlatformIDs(1, &__cu2cl_Platform, NULL);
    clGetDeviceIDs(__cu2cl_Platform, CL_DEVICE_TYPE_ALL, 1, &__cu2cl_Device, NULL);
    __cu2cl_Context = clCreateContext(NULL, 1, &__cu2cl_Device, NULL, NULL, NULL);
    __cu2cl_CommandQueue = clCreateCommandQueue(__cu2cl_Context, __cu2cl_Device, CL_QUEUE_PROFILING_ENABLE, &err);
//printf("Creation of main command queue: %s\n", getErrorString(err));
    __cu2cl_Init_stream_cu();
}

void __cu2cl_Cleanup() {
    __cu2cl_Cleanup_stream_cu();
    clReleaseCommandQueue(__cu2cl_CommandQueue);
    clReleaseContext(__cu2cl_Context);
}
