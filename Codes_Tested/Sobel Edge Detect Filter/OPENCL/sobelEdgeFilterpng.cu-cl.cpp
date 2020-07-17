#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include "cu2cl_util.h"




cl_kernel __cu2cl_Kernel_sobelGpu;
cl_program __cu2cl_Program_sobelEdgeFilterpng_cu;
extern const char *progSrc;
extern size_t progLen;

extern cl_kernel __cu2cl_Kernel___cu2cl_Memset;
extern cl_program __cu2cl_Util_Program;
extern cl_platform_id __cu2cl_Platform;
extern cl_device_id __cu2cl_Device;
extern cl_context __cu2cl_Context;
extern cl_command_queue __cu2cl_CommandQueue;

extern size_t globalWorkSize[3];
extern size_t localWorkSize[3];
void __cu2cl_Cleanup_sobelEdgeFilterpng_cu() {
    clReleaseKernel(__cu2cl_Kernel_sobelGpu);
    clReleaseProgram(__cu2cl_Program_sobelEdgeFilterpng_cu);
}
void __cu2cl_Init_sobelEdgeFilterpng_cu() {
    #ifdef WITH_ALTERA
    progLen = __cu2cl_LoadProgramSource("sobelEdgeFilterpng_cu_cl.aocx", &progSrc);
    __cu2cl_Program_sobelEdgeFilterpng_cu = clCreateProgramWithBinary(__cu2cl_Context, 1, &__cu2cl_Device, &progLen, (const unsigned char **)&progSrc, NULL, NULL);
    #else
    progLen = __cu2cl_LoadProgramSource("sobelEdgeFilterpng.cu-cl.cl", &progSrc);
    __cu2cl_Program_sobelEdgeFilterpng_cu = clCreateProgramWithSource(__cu2cl_Context, 1, &progSrc, &progLen, NULL);
    #endif
    free((void *) progSrc);
    clBuildProgram(__cu2cl_Program_sobelEdgeFilterpng_cu, 1, &__cu2cl_Device, "-I . ", NULL, NULL);
    __cu2cl_Kernel_sobelGpu = clCreateKernel(__cu2cl_Program_sobelEdgeFilterpng_cu, "sobelGpu", NULL);
}

cl_int err;
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

/*#include<iostream>
#include<stdio.h>
#include "math.h"
// #include "lodepng.h"
#include <string>
#include <sys/time.h>
#define N 20.0
// #include <helper_functions.h>
#include <cuda_runtime.h>
// #include <helper_cuda.h>
// nvcc sobelEdgeFilterpng.cu lodepng.cpp -arch sm_20
// Time taken by CPU :  0.95400 ms
// Time taken by GPU :  0.09100 ms


typedef unsigned char byte;

// making a structure for the image with pixels, width and height
struct imgData {
  imgData(byte* pix = NULL, unsigned int w = 0, unsigned int h = 0) : pixels(pix), width(w), height(h) {};
  byte* pixels;
  unsigned int width;
  unsigned int height;
};

// imgData loadImage(char* filename) {
//   unsigned int width, height;
//   byte* rgb;
//   unsigned error = lodepng_decode_file(&rgb, &width, &height, filename, LCT_RGBA, 8);
//   if (error) {
//     printf("LodePNG had an error during file processing. Exiting program.\n");
//     // printf("Error code: %u: %s\n", error, lodepng_error_text(error));
//     exit(2);
//   }
//   // we convert to grayscale here
//   byte* grayscale = new byte[width * height];
//   // pixels stored as rgba
//   byte* img = rgb;
//   for (int i = 0; i < width * height; ++i) {
//     int r = *img++;
//     int g = *img++;
//     int b = *img++;
//     int a = *img++;
//     grayscale[i] = 0.3 * r + 0.6 * g + 0.1 * b + 0.5;
//   }
//   free(rgb);
//   return imgData(grayscale, width, height);
// }


// void writeImage(char* filename, imgData img) {
//   std::string newName = filename;
//   // checking image coordinates
//   // printf("starting to write %d is width and %d is height\n", img.width, img.height);
//   unsigned error = lodepng_encode_file(newName.c_str(), img.pixels, img.width, img.height, LCT_GREY, 8);
//   if (error) {
//     printf("LodePNG had an error during file writing. Exiting program.\n");
//     exit(3);
//   }
//   else
//   {
//     printf("No error while writing?\n");
//   }
//   delete [] img.pixels;
// }



void sobelCpu(byte* orig, byte* op, const unsigned int width, const unsigned int height) {
  for (int y = 1; y < height - 1; y++) {
    for (int x = 1; x < width - 1; x++) {
      // we find dx and dy
      int dx = (-1 * orig[(y - 1) * width + (x - 1)]) + (-2 * orig[y * width + (x - 1)]) + (-1 * orig[(y + 1) * width + (x - 1)]) +
      (orig[(y - 1) * width + (x + 1)]) + (2 * orig[y * width + (x + 1)]) + (orig[(y + 1) * width + (x + 1)]);
      int dy = (orig[(y - 1) * width + (x - 1)]) + (2 * orig[(y - 1) * width + x]) + (orig[(y - 1) * width + (x + 1)]) +
      (-1 * orig[(y + 1) * width + (x - 1)]) + (-2 * orig[(y + 1) * width + x]) + (-1 * orig[(y + 1) * width + (x + 1)]);
      // sqrt(dx^2 + dy^2) gives the new pixel value
      op[y * width + x] = sqrt((float)((dx * dx) + (dy * dy)));
    }
  }
}

__global__ void sobelGpu(const byte* orig, byte* gpu, const unsigned int width, const unsigned int height) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  float dx, dy;
  if( threadIdx.x > 0 && threadIdx.y > 0 && threadIdx.x < (width-1) && threadIdx.y < (height-1)) {
    dx = (-1* orig[(y-1)*width + (x-1)]) + (-2*orig[y*width+(x-1)]) + (-1*orig[(y+1)*width+(x-1)]) +
    (    orig[(y-1)*width + (x+1)]) + ( 2*orig[y*width+(x+1)]) + (   orig[(y+1)*width+(x+1)]);
    dy = (    orig[(y-1)*width + (x-1)]) + ( 2*orig[(y-1)*width+x]) + (   orig[(y-1)*width+(x+1)]) +
    (-1* orig[(y+1)*width + (x-1)]) + (-2*orig[(y+1)*width+x]) + (-1*orig[(y+1)*width+(x+1)]);
    gpu[y*width + x] = sqrt( (dx*dx) + (dy*dy) );
  }
}



int main()
{
	byte *d_in,*d_out;

  // imgData in = loadImage("person.png");
  
  int w = in.width;int h=in.height;
  
  // checking dimensions
  // printf("width is %d, height is %d\n",w,h);
  
  imgData op(new byte[w*h],w,h);
  imgData op2(new byte[w*h],w,h);

  // debuggin to check about the 
  // writeImage("personcameas.png", in);

  cudaMalloc((void **)&d_in,(w*h));
  cudaMalloc((void **)&d_out,(w*h));

  cudaMemcpy(d_in,in.pixels,w*h,cudaMemcpyHostToDevice);
  cudaMemset(d_out, 0, w*h);

  dim3 threadsPerBlock(N,N,1);
  dim3 numBlocks(ceil(w/N),ceil(h/N),1);


  struct timeval t1, t2;
  gettimeofday(&t1, 0);
  sobelCpu(in.pixels,op.pixels,w,h);
  gettimeofday(&t2, 0);

  double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;

  printf("Time taken by CPU :  %3.5f ms \n", time);
  
  // printf("before writing to cpuimg\n");
  // writeImage("fromcpu.png", op);

  // printf("after writing to cpuimg\n");
  // writeImage("aftercpu.png", in);
  
  gettimeofday(&t1, 0);
  sobelGpu<<<numBlocks,threadsPerBlock>>>(d_in,d_out,w,h);
  
  // checking if any errors occured
  cudaError_t cudaerror = cudaDeviceSynchronize(); // waits for completion, returns error code
  if ( cudaerror != cudaSuccess ) fprintf( stderr, "Cuda failed to synchronize:\n"); // if error, output error
  gettimeofday(&t2, 0);
  
  // printf("survived kernel\n");
  time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
  printf("Time taken by GPU :  %3.5f ms \n", time);

  // writeImage("aftergpucpupersoncameas.png", in);
  checkCudaError(cudaMemcpy(op2.pixels,d_out,(w*h),cudaMemcpyDeviceToHost));
  op2.width=w;
  op2.height=h;
  
  // writeImage("fromgpu.png", op2);
  
}
*/
#include<iostream>
#include<stdio.h>
#include "math.h-cl.h"
// #include "lodepng.h"
#include <string>
#include <sys/time.h>
#define N 20.0

// nvcc sobelEdgeFilterpng.cu lodepng.cpp -arch sm_20
// Time taken by CPU :  0.95400 ms
// Time taken by GPU :  0.09100 ms


typedef unsigned char byte;

// making a structure for the image with pixels, width and height
struct imgData {
  imgData(byte* pix = NULL, unsigned int w = 0, unsigned int h = 0) : pixels(pix), width(w), height(h) {};
  byte* pixels;
  unsigned int width;
  unsigned int height;
};

imgData loadImage(char* filename) {
  unsigned int width, height;
  byte* rgb;
  unsigned error = lodepng_decode_file(&rgb, &width, &height, filename, LCT_RGBA, 8);
  if (error) {
    printf("LodePNG had an error during file processing. Exiting program.\n");
    // printf("Error code: %u: %s\n", error, lodepng_error_text(error));
    exit(2);
  }
  // we convert to grayscale here
  byte* grayscale = new byte[width * height];
  // pixels stored as rgba
  byte* img = rgb;
  for (int i = 0; i < width * height; ++i) {
    int r = *img++;
    int g = *img++;
    int b = *img++;
    int a = *img++;
    grayscale[i] = 0.3 * r + 0.6 * g + 0.1 * b + 0.5;
  }
  free(rgb);
  return imgData(grayscale, width, height);
}


void writeImage(char* filename, imgData img) {
  std::string newName = filename;
  // checking image coordinates
  // printf("starting to write %d is width and %d is height\n", img.width, img.height);
  unsigned error = lodepng_encode_file(newName.c_str(), img.pixels, img.width, img.height, LCT_GREY, 8);
  if (error) {
    printf("LodePNG had an error during file writing. Exiting program.\n");
    exit(3);
  }
  else
  {
    printf("No error while writing?\n");
  }
  delete [] img.pixels;
}



void sobelCpu(byte* orig, byte* op, const unsigned int width, const unsigned int height) {
  for (int y = 1; y < height - 1; y++) {
    for (int x = 1; x < width - 1; x++) {
      // we find dx and dy
      int dx = (-1 * orig[(y - 1) * width + (x - 1)]) + (-2 * orig[y * width + (x - 1)]) + (-1 * orig[(y + 1) * width + (x - 1)]) +
      (orig[(y - 1) * width + (x + 1)]) + (2 * orig[y * width + (x + 1)]) + (orig[(y + 1) * width + (x + 1)]);
      int dy = (orig[(y - 1) * width + (x - 1)]) + (2 * orig[(y - 1) * width + x]) + (orig[(y - 1) * width + (x + 1)]) +
      (-1 * orig[(y + 1) * width + (x - 1)]) + (-2 * orig[(y + 1) * width + x]) + (-1 * orig[(y + 1) * width + (x + 1)]);
      // sqrt(dx^2 + dy^2) gives the new pixel value
      op[y * width + x] = sqrt((float)((dx * dx) + (dy * dy)));
    }
  }
}





int main()
{
__cu2cl_Init();

  cl_mem d_in;
cl_mem d_out;

  imgData in = loadImage("person.png");
  
  int w = in.width;int h=in.height;
  
  // checking dimensions
  // printf("width is %d, height is %d\n",w,h);
  
  imgData op(new byte[w*h],w,h);
  imgData op2(new byte[w*h],w,h);

  // debuggin to check about the 
  // writeImage("personcameas.png", in);

  *(void **)&d_in = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, (w*h), NULL, NULL);
  *(void **)&d_out = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, (w*h), NULL, NULL);

  clEnqueueWriteBuffer(__cu2cl_CommandQueue, d_in, CL_TRUE, 0, w*h, in.pixels, 0, NULL, NULL);
  __cu2cl_Memset(d_out, 0, w*h);

  size_t threadsPerBlock[3] = {N, N, 1};
  size_t numBlocks[3] = {ceil(w/N), ceil(h/N), 1};


  struct timeval t1, t2;
  gettimeofday(&t1, 0);
  sobelCpu(in.pixels,op.pixels,w,h);
  gettimeofday(&t2, 0);

  double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;

  printf("Time taken by CPU :  %3.5f ms \n", time);
  
  // printf("before writing to cpuimg\n");
  writeImage("fromcpu.png", op);

  // printf("after writing to cpuimg\n");
  // writeImage("aftercpu.png", in);
  
  gettimeofday(&t1, 0);
  clSetKernelArg(__cu2cl_Kernel_sobelGpu, 0, sizeof(cl_mem), &d_in);
clSetKernelArg(__cu2cl_Kernel_sobelGpu, 1, sizeof(cl_mem), &d_out);
clSetKernelArg(__cu2cl_Kernel_sobelGpu, 2, sizeof(unsigned int), &w);
clSetKernelArg(__cu2cl_Kernel_sobelGpu, 3, sizeof(unsigned int), &h);
localWorkSize[0] = threadsPerBlock[0];
localWorkSize[1] = threadsPerBlock[1];
localWorkSize[2] = threadsPerBlock[2];
globalWorkSize[0] = numBlocks[0]*localWorkSize[0];
globalWorkSize[1] = numBlocks[1]*localWorkSize[1];
globalWorkSize[2] = numBlocks[2]*localWorkSize[2];
clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_sobelGpu, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
  
  // checking if any errors occured
  cl_int cudaerror = clFinish(__cu2cl_CommandQueue); // waits for completion, returns error code
  if ( clFinish(__cu2cl_CommandQueue) == CL_SUCCESS ) 
  {
    printf("works? \n");
  // fprintf( stderr, "Cuda failed to synchronize:\n"); // if error, output error
  }
  gettimeofday(&t2, 0);
  
  // printf("survived kernel\n");
  time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
  printf("Time taken by GPU :  %3.5f ms \n", time);

  // writeImage("aftergpucpupersoncameas.png", in);
  clEnqueueReadBuffer(__cu2cl_CommandQueue, d_out, CL_TRUE, 0, (w*h), op2.pixels, 0, NULL, NULL);
  op2.width=w;
  op2.height=h;
  
  writeImage("fromgpu.png", op2);
  
__cu2cl_Cleanup();
}
