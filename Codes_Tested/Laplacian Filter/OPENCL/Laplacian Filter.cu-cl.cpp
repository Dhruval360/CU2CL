#include <vector>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include "cu2cl_util.h"




cl_kernel __cu2cl_Kernel_laplacianFilter;
cl_program __cu2cl_Program_Laplacian_Filter_cu;
cl_int err;
extern const char *progSrc;
extern size_t progLen;

extern cl_platform_id __cu2cl_Platform;
extern cl_device_id __cu2cl_Device;
extern cl_context __cu2cl_Context;
extern cl_command_queue __cu2cl_CommandQueue;

extern size_t globalWorkSize[3];
extern size_t localWorkSize[3];
void __cu2cl_Cleanup_Laplacian_Filter_cu() {
    clReleaseKernel(__cu2cl_Kernel_laplacianFilter);
    clReleaseProgram(__cu2cl_Program_Laplacian_Filter_cu);
}
void __cu2cl_Init_Laplacian_Filter_cu() {
    #ifdef WITH_ALTERA
    progLen = __cu2cl_LoadProgramSource("Laplacian_Filter_cu_cl.aocx", &progSrc);
    __cu2cl_Program_Laplacian_Filter_cu = clCreateProgramWithBinary(__cu2cl_Context, 1, &__cu2cl_Device, &progLen, (const unsigned char **)&progSrc, NULL, &err);
    //printf("clCreateProgramWithBinary for Laplacian Filter.cu-cl.cl: %s\n", getErrorString(err));
    #else
    progLen = __cu2cl_LoadProgramSource("Laplacian Filter.cu-cl.cl", &progSrc);
    __cu2cl_Program_Laplacian_Filter_cu = clCreateProgramWithSource(__cu2cl_Context, 1, &progSrc, &progLen, &err);
    //printf("clCreateProgramWithSource for Laplacian Filter.cu-cl.cl: %s\n", getErrorString(err));
    #endif
    free((void *) progSrc);
    err = clBuildProgram(__cu2cl_Program_Laplacian_Filter_cu, 1, &__cu2cl_Device, "-I . ", NULL, NULL);
    /*printf("clBuildProgram : %s\n", getErrorString(err)); //Uncomment this line to access the error string of the error code returned by clBuildProgram*/
    if(err != CL_SUCCESS){
        std::vector<char> buildLog;
        size_t logSize;
        err = clGetProgramBuildInfo(__cu2cl_Program_Laplacian_Filter_cu, __cu2cl_Device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        printf("clGetProgramBuildInfo : %s\n", getErrorString(err));
        buildLog.resize(logSize);
        clGetProgramBuildInfo(__cu2cl_Program_Laplacian_Filter_cu, __cu2cl_Device, CL_PROGRAM_BUILD_LOG, logSize, &buildLog[0], NULL);
        printf("%s\n", &buildLog[0]);
    }
    __cu2cl_Kernel_laplacianFilter = clCreateKernel(__cu2cl_Program_Laplacian_Filter_cu, "laplacianFilter", &err);
    /*printf("__cu2cl_Kernel_laplacianFilter creation: %s\n", getErrorString(err)); // Uncomment this line to get error string for the error code returned by clCreateKernel while creating the Kernel: laplacianFilter*/
}


#include "opencv2/imgproc/imgproc.hpp-cl.h"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <stdio.h>



#define BLOCK_SIZE      16
#define FILTER_WIDTH    3       
#define FILTER_HEIGHT   3       

using namespace std;

// Run Laplacian Filter on GPU




void laplacianFilter_GPU_wrapper(const cv::Mat& input, cv::Mat& output);
  

using namespace std;


// Program main
int main() {
__cu2cl_Init();


   // name of image
   string image_name = "sample";

   // input & output file names
   string input_file =  image_name+".jpeg";
   string output_file_gpu = image_name+"_gpu.jpeg";

   // Read input image 
   cv::Mat srcImage = cv::imread(input_file ,CV_LOAD_IMAGE_UNCHANGED);
   if(srcImage.empty())
   {
      std::cout<<"Image Not Found: "<< input_file << std::endl;
      return -1;
   }
   cout <<"\ninput image size: "<<srcImage.cols<<" "<<srcImage.rows<<" "<<srcImage.channels()<<"\n";

   // convert RGB to gray scale
   cv::cvtColor(srcImage, srcImage, CV_BGR2GRAY);

   // Declare the output image  
   cv::Mat dstImage (srcImage.size(), srcImage.type());

   // run laplacian filter on GPU  
   laplacianFilter_GPU_wrapper(srcImage, dstImage);
   // normalization to 0-255
   dstImage.convertTo(dstImage, CV_32F, 1.0 / 255, 0);
   dstImage*=255;
   // Output image
   imwrite(output_file_gpu, dstImage);
      
   return 0;
__cu2cl_Cleanup();
}














