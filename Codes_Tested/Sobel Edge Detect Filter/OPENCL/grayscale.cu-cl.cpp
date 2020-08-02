#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include "cu2cl_util.h"




cl_kernel __cu2cl_Kernel_rgba_to_greyscale;
cl_program __cu2cl_Program_grayscale_cu;
cl_int err;
extern cl_kernel __cu2cl_Kernel_sobelGpu;
extern cl_program __cu2cl_Program_sobelEdgeFilterpng_cu;
extern cl_int err;
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
void __cu2cl_Cleanup_grayscale_cu() {
    clReleaseKernel(__cu2cl_Kernel_rgba_to_greyscale);
    clReleaseProgram(__cu2cl_Program_grayscale_cu);
}
void __cu2cl_Init_grayscale_cu() {
    #ifdef WITH_ALTERA
    progLen = __cu2cl_LoadProgramSource("grayscale_cu_cl.aocx", &progSrc);
    __cu2cl_Program_grayscale_cu = clCreateProgramWithBinary(__cu2cl_Context, 1, &__cu2cl_Device, &progLen, (const unsigned char **)&progSrc, NULL, &err);
    //printf("clCreateProgramWithBinary for grayscale.cu-cl.cl: %s\n", getErrorString(err));
    #else
    progLen = __cu2cl_LoadProgramSource("grayscale.cu-cl.cl", &progSrc);
    __cu2cl_Program_grayscale_cu = clCreateProgramWithSource(__cu2cl_Context, 1, &progSrc, &progLen, &err);
    //printf("clCreateProgramWithSource for grayscale.cu-cl.cl: %s\n", getErrorString(err));
    #endif
    free((void *) progSrc);
    err = clBuildProgram(__cu2cl_Program_grayscale_cu, 1, &__cu2cl_Device, "-I . ", NULL, NULL);
    /*printf("clBuildProgram : %s\n", getErrorString(err)); //Uncomment this line to access the error string of the error code returned by clBuildProgram*/
    if(err != CL_SUCCESS){
        std::vector<char> buildLog;
        size_t logSize;
        err = clGetProgramBuildInfo(__cu2cl_Program_grayscale_cu, __cu2cl_Device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        printf("clGetProgramBuildInfo : %s\n", getErrorString(err));
        buildLog.resize(logSize);
        clGetProgramBuildInfo(__cu2cl_Program_grayscale_cu, __cu2cl_Device, CL_PROGRAM_BUILD_LOG, logSize, &buildLog[0], NULL);
        printf("%s\n", &buildLog[0]);
    }
    __cu2cl_Kernel_rgba_to_greyscale = clCreateKernel(__cu2cl_Program_grayscale_cu, "rgba_to_greyscale", &err);
    /*printf("__cu2cl_Kernel_rgba_to_greyscale creation: %s\n", getErrorString(err)); // Uncomment this line to get error string for the error code returned by clCreateKernel while creating the Kernel: rgba_to_greyscale*/
}

#include <iostream>
#include <string>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>


#include <string>

cv::Mat imageRGBA;
cv::Mat imageGrey;

cl_uchar4        *d_rgbaImage__;
unsigned char *d_greyImage__;

size_t numRows() { return imageRGBA.rows; }
size_t numCols() { return imageRGBA.cols; }

//return types are void since any internal error will be handled by quitting
//no point in returning error codes...
//returns a pointer to an RGBA version of the input image
//and a pointer to the single channel grey-scale output
//on both the host and device
void preProcess(uchar4 **inputImage, unsigned char **greyImage,
                cl_mem * d_rgbaImage, cl_mem * d_greyImage,
                const std::string &filename) {
  //make sure the context initializes ok
  // checkCudaErrors(cudaFree(0));

  cv::Mat image;
  image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
  if (image.empty()) {
    std::cerr << "Couldn't open file: " << filename << std::endl;
    exit(1);
  }

  cv::cvtColor(image, imageRGBA, CV_BGR2RGBA);

  //allocate memory for the output
  imageGrey.create(image.rows, image.cols, CV_8UC1);

  //This shouldn't ever happen given the way the images are created
  //at least based upon my limited understanding of OpenCV, but better to check
  if (!imageRGBA.isContinuous() || !imageGrey.isContinuous()) {
    std::cerr << "Images aren't continuous!! Exiting." << std::endl;
    exit(1);
  }

  *inputImage = (uchar4 *)imageRGBA.ptr<unsigned char>(0);
  *greyImage  = imageGrey.ptr<unsigned char>(0);

  const size_t numPixels = numRows() * numCols();
  //allocate memory on the device for both input and output
/*CU2CL Note -- Rewriting single decl*/
  *d_rgbaImage = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(cl_uchar4) * numPixels, NULL, &err);
//printf("clCreateBuffer for device variable *d_rgbaImage: %s\n", getErrorString(err));
/*CU2CL Note -- Rewriting single decl*/
  *d_greyImage = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(unsigned char) * numPixels, NULL, &err);
//printf("clCreateBuffer for device variable *d_greyImage: %s\n", getErrorString(err));
  __cu2cl_Memset(*d_greyImage, 0, numPixels * sizeof(unsigned char)); //make sure no memory is left laying around

  //copy input array to the GPU
  err = clEnqueueWriteBuffer(__cu2cl_CommandQueue, *d_rgbaImage, CL_TRUE, 0, sizeof(cl_uchar4) * numPixels, *inputImage, 0, NULL, NULL);
//printf("Memory copy from host variable *inputImage to device variable *d_rgbaImage: %s\n", getErrorString(err));

  d_rgbaImage__ = *d_rgbaImage;
  d_greyImage__ = *d_greyImage;
}

void postProcess(const std::string& output_file) {
  const int numPixels = numRows() * numCols();
  //copy the output back to the host
  err = clEnqueueReadBuffer(__cu2cl_CommandQueue, d_greyImage__, CL_TRUE, 0, sizeof(unsigned char) * numPixels, imageGrey.ptr<unsigned char>(0), 0, NULL, NULL);
//printf("Memory copy from device variable imageGrey.ptr<unsigned char>(0) to host variable d_greyImage__: %s\n", getErrorString(err));

  //output the image
  cv::imwrite(output_file.c_str(), imageGrey);

  //cleanup
  clReleaseMemObject(d_rgbaImage__);
  clReleaseMemObject(d_greyImage__);
}

//include the definitions of the above functions


void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
                            unsigned char* const d_greyImage, size_t numRows, size_t numCols)
{
  const int thread = 16;
  const size_t blockSize[3] = {thread, thread, 1};
  const size_t gridSize[3] = {ceil(numRows / (float)thread), ceil(numCols / (float)thread), 1};
/*CU2CL Note -- Fast-tracked dim3 type without cast*/
  err = clSetKernelArg(__cu2cl_Kernel_rgba_to_greyscale, 0, sizeof(const uchar4 *), &d_rgbaImage);
/*printf("clSetKernelArg for argument 0 of kernel __cu2cl_Kernel_rgba_to_greyscale: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_rgba_to_greyscale, 1, sizeof(unsigned char *), &d_greyImage);
/*printf("clSetKernelArg for argument 1 of kernel __cu2cl_Kernel_rgba_to_greyscale: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_rgba_to_greyscale, 2, sizeof(int), &numRows);
/*printf("clSetKernelArg for argument 2 of kernel __cu2cl_Kernel_rgba_to_greyscale: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_rgba_to_greyscale, 3, sizeof(int), &numCols);
/*printf("clSetKernelArg for argument 3 of kernel __cu2cl_Kernel_rgba_to_greyscale: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
localWorkSize[0] = blockSize[0];
localWorkSize[1] = blockSize[1];
localWorkSize[2] = blockSize[2];
globalWorkSize[0] = gridSize[0]*localWorkSize[0];
globalWorkSize[1] = gridSize[1]*localWorkSize[1];
globalWorkSize[2] = gridSize[2]*localWorkSize[2];
err = clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_rgba_to_greyscale, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
//printf("clEnqueueNDRangeKernel for the kernel __cu2cl_Kernel_rgba_to_greyscale: %s\n", getErrorString(err));

  err = clFinish(__cu2cl_CommandQueue);
//printf("clFinish return message = %s\n", getErrorString(err));
}

int main() {
__cu2cl_Init();

  cl_uchar4        *h_rgbaImage, *d_rgbaImage;
  unsigned char *h_greyImage, *d_greyImage;

  std::string input_file;
  std::string output_file;

  input_file  = "person.png";
  output_file = "GPU.png";


  //load the image and give us our input and output pointers
  preProcess(&h_rgbaImage, &h_greyImage, &d_rgbaImage, &d_greyImage, input_file);

  // GpuTimer timer;
  // timer.Start();
  //call the grayscale code
  your_rgba_to_greyscale(h_rgbaImage, d_rgbaImage, d_greyImage, numRows(), numCols());
  // timer.Stop();
  err = clFinish(__cu2cl_CommandQueue);
//printf("clFinish return message = %s\n", getErrorString(err));
  printf("\n");
  // int err = printf("%f msecs.\n", timer.Elapsed());

  // if (err < 0) {
  //   //Couldn't print! Probably the student closed stdout - bad news
  //   std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
  //   exit(1);
  // }

  //check results and output the grey image
  postProcess(output_file);

  return 0;
__cu2cl_Cleanup();
}
