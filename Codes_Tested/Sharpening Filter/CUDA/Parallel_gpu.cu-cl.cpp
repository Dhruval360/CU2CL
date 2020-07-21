#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include "cu2cl_util.h"




cl_kernel __cu2cl_Kernel_sharpeningFilter;
cl_program __cu2cl_Program_Parallel_gpu_cu;
extern const char *progSrc;
extern size_t progLen;

extern cl_platform_id __cu2cl_Platform;
extern cl_device_id __cu2cl_Device;
extern cl_context __cu2cl_Context;
extern cl_command_queue __cu2cl_CommandQueue;

extern size_t globalWorkSize[3];
extern size_t localWorkSize[3];
void __cu2cl_Cleanup_Parallel_gpu_cu() {
    clReleaseKernel(__cu2cl_Kernel_sharpeningFilter);
    clReleaseProgram(__cu2cl_Program_Parallel_gpu_cu);
}
void __cu2cl_Init_Parallel_gpu_cu() {
    #ifdef WITH_ALTERA
    progLen = __cu2cl_LoadProgramSource("Parallel_gpu_cu_cl.aocx", &progSrc);
    __cu2cl_Program_Parallel_gpu_cu = clCreateProgramWithBinary(__cu2cl_Context, 1, &__cu2cl_Device, &progLen, (const unsigned char **)&progSrc, NULL, &err);
    #else
    progLen = __cu2cl_LoadProgramSource("Parallel_gpu.cu-cl.cl", &progSrc);
    __cu2cl_Program_Parallel_gpu_cu = clCreateProgramWithSource(__cu2cl_Context, 1, &progSrc, &progLen, &err);
    //printf("clCreateProgramWithSource for Parallel_gpu.cu-cl.cl: %s\n", getErrorString(err));
    #endif
    free((void *) progSrc);
    err = clBuildProgram(__cu2cl_Program_Parallel_gpu_cu, 1, &__cu2cl_Device, "-I . ", NULL, NULL);
    //printf("clBuildProgram : %s\n", getErrorString(err)); //Uncomment this line to access the error string of the error code returned by clBuildProgram
    if(err != CL_SUCCESS){
        std::vector<char> buildLog;
        size_t logSize;
        err = clGetProgramBuildInfo(__cu2cl_Program_Parallel_gpu_cu, &__cu2cl_Device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        printf("clGetProgramBuildInfo : %s\n", getErrorString(err));
        buildLog.resize(logSize)
;        clGetProgramBuildInfo(__cu2cl_Program_Parallel_gpu_cu, __cu2cl_Device, CL_PROGRAM_BUILD_LOG, logSize, &buildLog[0], nullptr);
        std::cout << &buildLog[0] << '
';
    }

    __cu2cl_Kernel_sharpeningFilter = clCreateKernel(__cu2cl_Program_Parallel_gpu_cu, "sharpeningFilter", &err);
    //printf("__cu2cl_Kernel_sharpeningFilter creation: %s\n", getErrorString(err)); // Uncomment this line to get error string for the error code returned by clCreateKernel while creating the Kernel: sharpeningFilter
}

#include "opencv2/imgproc/imgproc.hpp-cl.h"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <stdio.h>


//
#define BLOCK_SIZE      16
#define FILTER_WIDTH    3       
#define FILTER_HEIGHT   3       

using namespace std;
using namespace cv;



void sharpeningFilter_GPU_wrapper(const Mat& input, Mat& output);



int main(int argc, char** argv) {
__cu2cl_Init();



    string image_name = "sample";


    string input_file = image_name + ".jpg";

    string output_file_gpu = image_name + "_gpu.jpg";

    // Read input image 
    Mat srcImage = imread(input_file);
    if (srcImage.empty())
    {
        cout << "Image Not Found: " << input_file << endl;
        return -1;
    }
    cout << "\ninput image size: " << srcImage.cols << " " << srcImage.rows << " " << srcImage.channels() << "\n";

    // Declare the output image  
    Mat dstImage(srcImage.size(), srcImage.type());

    // run median filter on GPU  
    sharpeningFilter_GPU_wrapper(srcImage, dstImage);
    // Output image
    imwrite(output_file_gpu, dstImage);
    Mat image = imread("sample_gpu.jpg");
    imshow("image", image);
    waitKey();


    return 0;
__cu2cl_Cleanup();
}
// Run Sharpening Filter on GPU



// The wrapper is used to call sharpening filter 
void sharpeningFilter_GPU_wrapper(const cv::Mat& input, cv::Mat& output)
{
    // Use cuda event to catch time
    cl_event start, stop;
    start;
    stop;

    // Calculate number of image channels
    int channel = input.step / input.cols;

    // Calculate number of input & output bytes in each block
    const int inputSize = input.cols * input.rows * channel;
    const int outputSize = output.cols * output.rows * channel;
    cl_mem d_input;
cl_mem d_output;

    // Allocate device memory
    d_input = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, inputSize, NULL, &err);
//printf("clCreateBuffer for device variable d_input is: %s\n", getErrorString(err));
    d_output = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, outputSize, NULL, &err);
//printf("clCreateBuffer for device variable d_output is: %s\n", getErrorString(err));

    // Copy data from OpenCV input image to device memory
    clEnqueueWriteBuffer(__cu2cl_CommandQueue, d_input, CL_TRUE, 0, inputSize, input.ptr(), 0, NULL, NULL);

    // Specify block size
    const size_t block[3] = {BLOCK_SIZE, BLOCK_SIZE, 1};

    // Calculate grid size to cover the whole image
    const size_t grid[3] = {(output.cols + block[0] - 1) / block[0], (output.rows + block[1] - 1) / block[1], 1};

    // Start time
    clEnqueueMarker(__cu2cl_CommandQueue, &start);

    // Run BoxFilter kernel on CUDA 
/*CU2CL Note -- Fast-tracked dim3 type without cast*/
    err = clSetKernelArg(__cu2cl_Kernel_sharpeningFilter, 0, sizeof(cl_mem), &d_input);
//printf("clSetKernelArg for argument 0 of kernel __cu2cl_Kernel_sharpeningFilter is: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg
err = clSetKernelArg(__cu2cl_Kernel_sharpeningFilter, 1, sizeof(cl_mem), &d_output);
//printf("clSetKernelArg for argument 1 of kernel __cu2cl_Kernel_sharpeningFilter is: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg
err = clSetKernelArg(__cu2cl_Kernel_sharpeningFilter, 2, sizeof(unsigned int), &output.cols);
//printf("clSetKernelArg for argument 2 of kernel __cu2cl_Kernel_sharpeningFilter is: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg
err = clSetKernelArg(__cu2cl_Kernel_sharpeningFilter, 3, sizeof(unsigned int), &output.rows);
//printf("clSetKernelArg for argument 3 of kernel __cu2cl_Kernel_sharpeningFilter is: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg
err = clSetKernelArg(__cu2cl_Kernel_sharpeningFilter, 4, sizeof(int), &channel);
//printf("clSetKernelArg for argument 4 of kernel __cu2cl_Kernel_sharpeningFilter is: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg
localWorkSize[0] = block[0];
localWorkSize[1] = block[1];
localWorkSize[2] = block[2];
globalWorkSize[0] = grid[0]*localWorkSize[0];
globalWorkSize[1] = grid[1]*localWorkSize[1];
globalWorkSize[2] = grid[2]*localWorkSize[2];
clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_sharpeningFilter, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

    // Stop time
    clEnqueueMarker(__cu2cl_CommandQueue, &stop);

    //Copy data from device memory to output image
    clEnqueueReadBuffer(__cu2cl_CommandQueue, d_output, CL_TRUE, 0, outputSize, output.ptr(), 0, NULL, NULL);

    //Free the device memory
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);

    clWaitForEvents(1, &stop);
    float milliseconds = 0;

    // Calculate elapsed time in milisecond  
    __cu2cl_EventElapsedTime(&milliseconds, start, stop);
    cout << "\nProcessing time on GPU (ms): " << milliseconds << "\n";
}
