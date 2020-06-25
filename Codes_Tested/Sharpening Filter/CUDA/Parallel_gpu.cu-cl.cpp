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
    __cu2cl_Program_Parallel_gpu_cu = clCreateProgramWithBinary(__cu2cl_Context, 1, &__cu2cl_Device, &progLen, (const unsigned char **)&progSrc, NULL, NULL);
    #else
    progLen = __cu2cl_LoadProgramSource("Parallel_gpu.cu-cl.cl", &progSrc);
    __cu2cl_Program_Parallel_gpu_cu = clCreateProgramWithSource(__cu2cl_Context, 1, &progSrc, &progLen, NULL);
    #endif
    free((void *) progSrc);
    clBuildProgram(__cu2cl_Program_Parallel_gpu_cu, 1, &__cu2cl_Device, "-I . ", NULL, NULL);
    __cu2cl_Kernel_sharpeningFilter = clCreateKernel(__cu2cl_Program_Parallel_gpu_cu, "sharpeningFilter", NULL);
}



#include "device_launch_parameters.h-cl.h"


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
/*CU2CL Unsupported -- Unsupported CUDA call: cudaEventCreate*/
    cudaEventCreate(&start);
/*CU2CL Unsupported -- Unsupported CUDA call: cudaEventCreate*/
    cudaEventCreate(&stop);

    // Calculate number of image channels
    int channel = input.step / input.cols;

    // Calculate number of input & output bytes in each block
    const int inputSize = input.cols * input.rows * channel;
    const int outputSize = output.cols * output.rows * channel;
    cl_mem d_input;
cl_mem d_output;

    // Allocate device memory
    *&d_input = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, inputSize, NULL, NULL);
    *&d_output = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, outputSize, NULL, NULL);

    // Copy data from OpenCV input image to device memory
    clEnqueueWriteBuffer(__cu2cl_CommandQueue, d_input, CL_TRUE, 0, inputSize, input.ptr(), 0, NULL, NULL);

    // Specify block size
    const size_t block[3] = {BLOCK_SIZE, BLOCK_SIZE, 1};

    // Calculate grid size to cover the whole image
    const size_t grid[3] = {(output.cols + block.x - 1) / block.x, (output.rows + block.y - 1) / block.y, 1};

    // Start time
    clEnqueueMarker(, &start);

    // Run BoxFilter kernel on CUDA 
    sharpeningFilter << <grid, block >> > (d_input, d_output, output.cols, output.rows, channel);

    // Stop time
    clEnqueueMarker(, &stop);

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
