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

extern cl_device_id * __cu2cl_AllDevices;
extern cl_uint __cu2cl_AllDevices_curr_idx;
extern cl_uint __cu2cl_AllDevices_size;
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
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <stdio.h>




#define BLOCK_SIZE      16
#define FILTER_WIDTH    3       
#define FILTER_HEIGHT   3       

using namespace std;

void laplacianFilter_GPU_wrapper(const cv::Mat& input, cv::Mat& output);
// Run Laplacian Filter on GPU





  

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
/*CU2CL Warning -- CU2CL Identified cudaDeviceReset usage*/
	__cu2cl_ResetDevice();      
   return 0;
__cu2cl_Cleanup();
}




void laplacianFilter_GPU_wrapper(const cv::Mat& input, cv::Mat& output)
{
        // Use cuda event to catch time
        cl_event start, stop;
        start = clCreateUserEvent(__cu2cl_Context, &err);
//printf("clCreateUserEvent for the event start: %s\n", getErrorString(err));
        stop = clCreateUserEvent(__cu2cl_Context, &err);
//printf("clCreateUserEvent for the event stop: %s\n", getErrorString(err));

        // Calculate number of input & output bytes in each block
        const int inputSize = input.cols * input.rows;
        const int outputSize = output.cols * output.rows;
        cl_mem d_input;
cl_mem d_output;
        
        // Allocate device memory
        d_input = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, inputSize, NULL, &err);
//printf("clCreateBuffer for device variable d_input: %s\n", getErrorString(err));
        d_output = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, outputSize, NULL, &err);
//printf("clCreateBuffer for device variable d_output: %s\n", getErrorString(err));

        // Copy data from OpenCV input image to device memory
        err = clEnqueueWriteBuffer(__cu2cl_CommandQueue, d_input, CL_TRUE, 0, inputSize, input.ptr(), 0, NULL, NULL);
//printf("Memory copy from host variable input.ptr() to device variable d_input: %s\n", getErrorString(err));

        // Specify block size
        const size_t block[3] = {BLOCK_SIZE, BLOCK_SIZE, 1};

        // Calculate grid size to cover the whole image
        const size_t grid[3] = {(output.cols + block[0] - 1)/block[0], (output.rows + block[1] - 1)/block[1], 1};

        // Start time
        err = clEnqueueMarkerWithWaitList(__cu2cl_CommandQueue, 0, 0, &start);
//printf("clEnqueMarker for the event start: %s\n", getErrorString(err));

        // Run BoxFilter kernel on CUDA 
/*CU2CL Note -- Fast-tracked dim3 type without cast*/
        err = clSetKernelArg(__cu2cl_Kernel_laplacianFilter, 0, sizeof(cl_mem), &d_input);
/*printf("clSetKernelArg for argument 0 of kernel __cu2cl_Kernel_laplacianFilter: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_laplacianFilter, 1, sizeof(cl_mem), &d_output);
/*printf("clSetKernelArg for argument 1 of kernel __cu2cl_Kernel_laplacianFilter: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_laplacianFilter, 2, sizeof(unsigned int), &output.cols);
/*printf("clSetKernelArg for argument 2 of kernel __cu2cl_Kernel_laplacianFilter: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_laplacianFilter, 3, sizeof(unsigned int), &output.rows);
/*printf("clSetKernelArg for argument 3 of kernel __cu2cl_Kernel_laplacianFilter: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
localWorkSize[0] = block[0];
localWorkSize[1] = block[1];
localWorkSize[2] = block[2];
globalWorkSize[0] = grid[0]*localWorkSize[0];
globalWorkSize[1] = grid[1]*localWorkSize[1];
globalWorkSize[2] = grid[2]*localWorkSize[2];
err = clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_laplacianFilter, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
//printf("clEnqueueNDRangeKernel for the kernel __cu2cl_Kernel_laplacianFilter: %s\n", getErrorString(err));

        // Stop time
        err = clEnqueueMarkerWithWaitList(__cu2cl_CommandQueue, 0, 0, &stop);
//printf("clEnqueMarker for the event stop: %s\n", getErrorString(err));

        //Copy data from device memory to output image
        err = clEnqueueReadBuffer(__cu2cl_CommandQueue, d_output, CL_TRUE, 0, outputSize, output.ptr(), 0, NULL, NULL);
//printf("Memory copy from device variable output.ptr() to host variable d_output: %s\n", getErrorString(err));

        //Free the device memory
        clReleaseMemObject(d_input);
        clReleaseMemObject(d_output);

        clWaitForEvents(1, &stop);
        float milliseconds = 0;
        
        // Calculate elapsed time in milisecond  
        __cu2cl_EventElapsedTime(&milliseconds, start, stop);
        cout<< "\nProcessing time for GPU (ms): " << milliseconds << "\n";
}
