#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include "cu2cl_util.h"




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

void __cu2cl_Init_Parallel_gpu_cu() {
    #ifdef WITH_ALTERA
    progLen = __cu2cl_LoadProgramSource("Parallel_gpu_cu_cl.aocx", &progSrc);
    __cu2cl_Program_Parallel_gpu_cu = clCreateProgramWithBinary(__cu2cl_Context, 1, &__cu2cl_Device, &progLen, (const unsigned char **)&progSrc, NULL, &err);
    #else
    progLen = __cu2cl_LoadProgramSource("Parallel_gpu.cu-cl.cl", &progSrc);
    __cu2cl_Program_Parallel_gpu_cu = clCreateProgramWithSource(__cu2cl_Context, 1, &progSrc, &progLen, &err);
    //printf("clCreateProgramWithSource for Parallel_gpu.cu-cl.cl: %s\n", getErrorString(err));    #endif
    free((void *) progSrc);
    err = clBuildProgram(__cu2cl_Program_Parallel_gpu_cu, 1, &__cu2cl_Device, "-I . ", NULL, NULL);
    /*printf("clBuildProgram : %s\n", getErrorString(err)); //Uncomment this line to access the error string of the error code returned by clBuildProgram*/
    if(err != CL_SUCCESS){
        std::vector<char> buildLog;
        size_t logSize;
        err = clGetProgramBuildInfo(__cu2cl_Program_Parallel_gpu_cu, &__cu2cl_Device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        printf("clGetProgramBuildInfo : %s\n", getErrorString(err));
        buildLog.resize(logSize);
        clGetProgramBuildInfo(__cu2cl_Program_Parallel_gpu_cu, __cu2cl_Device, CL_PROGRAM_BUILD_LOG, logSize, &buildLog[0], nullptr);
        std::cout << &buildLog[0] << '\n';
    }
    __cu2cl_Kernel_sharpeningFilter = clCreateKernel(__cu2cl_Program_Parallel_gpu_cu, "sharpeningFilter", &err);
    /*printf("__cu2cl_Kernel_sharpeningFilter creation: %s
", getErrorString(err)); // Uncomment this line to get error string for the error code returned by clCreateKernel while creating the Kernel: sharpeningFilter*/
}

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
    err = clEnqueueWriteBuffer(__cu2cl_CommandQueue, d_input, CL_TRUE, 0, inputSize, input.ptr(), 0, NULL, NULL);
//printf("Memory copy from host variable input.ptr() to device variable d_input: %s\n", getErrorString(err));

    // Specify block size
    const size_t block[3] = {BLOCK_SIZE, BLOCK_SIZE, 1};

    // Calculate grid size to cover the whole image
    const size_t grid[3] = {(output.cols + block[0] - 1) / block[0], (output.rows + block[1] - 1) / block[1], 1};

    // Start time
    err = clEnqueueMarker(__cu2cl_CommandQueue, &start);
//printf("clEnqueMarker for the event start: %s\n", getErrorString(err));

    // Run BoxFilter kernel on CUDA 
/*CU2CL Note -- Fast-tracked dim3 type without cast*/
    err = clSetKernelArg(__cu2cl_Kernel_sharpeningFilter, 0, sizeof(cl_mem), &d_input);
/*printf("clSetKernelArg for argument 0 of kernel __cu2cl_Kernel_sharpeningFilter is: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_sharpeningFilter, 1, sizeof(cl_mem), &d_output);
/*printf("clSetKernelArg for argument 1 of kernel __cu2cl_Kernel_sharpeningFilter is: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_sharpeningFilter, 2, sizeof(unsigned int), &output.cols);
/*printf("clSetKernelArg for argument 2 of kernel __cu2cl_Kernel_sharpeningFilter is: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_sharpeningFilter, 3, sizeof(unsigned int), &output.rows);
/*printf("clSetKernelArg for argument 3 of kernel __cu2cl_Kernel_sharpeningFilter is: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_sharpeningFilter, 4, sizeof(int), &channel);
/*printf("clSetKernelArg for argument 4 of kernel __cu2cl_Kernel_sharpeningFilter is: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
localWorkSize[0] = block[0];
localWorkSize[1] = block[1];
localWorkSize[2] = block[2];
globalWorkSize[0] = grid[0]*localWorkSize[0];
globalWorkSize[1] = grid[1]*localWorkSize[1];
globalWorkSize[2] = grid[2]*localWorkSize[2];
err = clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_sharpeningFilter, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
//printf("clEnqueueNDRangeKernel for the kernel __cu2cl_Kernel_sharpeningFilter: %s\n", getErrorString(err));

    // Stop time
    err = clEnqueueMarker(__cu2cl_CommandQueue, &stop);
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
    cout << "\nProcessing time on GPU (ms): " << milliseconds << "\n";
}
