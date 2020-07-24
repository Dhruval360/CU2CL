#include <vector>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include "cu2cl_util.h"




cl_kernel __cu2cl_Kernel_naive_normalized_cross_correlation;
cl_kernel __cu2cl_Kernel_remove_redness_from_coordinates;
cl_kernel __cu2cl_Kernel_histogram_kernel;
cl_kernel __cu2cl_Kernel_exclusive_scan_kernel;
cl_kernel __cu2cl_Kernel_move_kernel;
cl_program __cu2cl_Program_redEyeGPU_cu;
cl_int err;
extern cl_kernel __cu2cl_Kernel_naive_normalized_cross_correlation;
extern cl_kernel __cu2cl_Kernel_remove_redness_from_coordinates;
extern cl_program __cu2cl_Program_redEYECPU_cu;
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
void __cu2cl_Cleanup_redEyeGPU_cu() {
    clReleaseKernel(__cu2cl_Kernel_naive_normalized_cross_correlation);
    clReleaseKernel(__cu2cl_Kernel_remove_redness_from_coordinates);
    clReleaseKernel(__cu2cl_Kernel_histogram_kernel);
    clReleaseKernel(__cu2cl_Kernel_exclusive_scan_kernel);
    clReleaseKernel(__cu2cl_Kernel_move_kernel);
    clReleaseProgram(__cu2cl_Program_redEyeGPU_cu);
}
void __cu2cl_Init_redEyeGPU_cu() {
    #ifdef WITH_ALTERA
    progLen = __cu2cl_LoadProgramSource("redEyeGPU_cu_cl.aocx", &progSrc);
    __cu2cl_Program_redEyeGPU_cu = clCreateProgramWithBinary(__cu2cl_Context, 1, &__cu2cl_Device, &progLen, (const unsigned char **)&progSrc, NULL, &err);
    //printf("clCreateProgramWithBinary for redEyeGPU.cu-cl.cl: %s\n", getErrorString(err));
    #else
    progLen = __cu2cl_LoadProgramSource("redEyeGPU.cu-cl.cl", &progSrc);
    __cu2cl_Program_redEyeGPU_cu = clCreateProgramWithSource(__cu2cl_Context, 1, &progSrc, &progLen, &err);
    //printf("clCreateProgramWithSource for redEyeGPU.cu-cl.cl: %s\n", getErrorString(err));
    #endif
    free((void *) progSrc);
    err = clBuildProgram(__cu2cl_Program_redEyeGPU_cu, 1, &__cu2cl_Device, "-I . ", NULL, NULL);
    /*printf("clBuildProgram : %s\n", getErrorString(err)); //Uncomment this line to access the error string of the error code returned by clBuildProgram*/
    if(err != CL_SUCCESS){
        std::vector<char> buildLog;
        size_t logSize;
        err = clGetProgramBuildInfo(__cu2cl_Program_redEyeGPU_cu, __cu2cl_Device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        printf("clGetProgramBuildInfo : %s\n", getErrorString(err));
        buildLog.resize(logSize);
        clGetProgramBuildInfo(__cu2cl_Program_redEyeGPU_cu, __cu2cl_Device, CL_PROGRAM_BUILD_LOG, logSize, &buildLog[0], NULL);
        printf("%s\n", &buildLog[0]);
    }
    __cu2cl_Kernel_naive_normalized_cross_correlation = clCreateKernel(__cu2cl_Program_redEyeGPU_cu, "naive_normalized_cross_correlation", &err);
    /*printf("__cu2cl_Kernel_naive_normalized_cross_correlation creation: %s\n", getErrorString(err)); // Uncomment this line to get error string for the error code returned by clCreateKernel while creating the Kernel: naive_normalized_cross_correlation*/
    __cu2cl_Kernel_remove_redness_from_coordinates = clCreateKernel(__cu2cl_Program_redEyeGPU_cu, "remove_redness_from_coordinates", &err);
    /*printf("__cu2cl_Kernel_remove_redness_from_coordinates creation: %s\n", getErrorString(err)); // Uncomment this line to get error string for the error code returned by clCreateKernel while creating the Kernel: remove_redness_from_coordinates*/
    __cu2cl_Kernel_histogram_kernel = clCreateKernel(__cu2cl_Program_redEyeGPU_cu, "histogram_kernel", &err);
    /*printf("__cu2cl_Kernel_histogram_kernel creation: %s\n", getErrorString(err)); // Uncomment this line to get error string for the error code returned by clCreateKernel while creating the Kernel: histogram_kernel*/
    __cu2cl_Kernel_exclusive_scan_kernel = clCreateKernel(__cu2cl_Program_redEyeGPU_cu, "exclusive_scan_kernel", &err);
    /*printf("__cu2cl_Kernel_exclusive_scan_kernel creation: %s\n", getErrorString(err)); // Uncomment this line to get error string for the error code returned by clCreateKernel while creating the Kernel: exclusive_scan_kernel*/
    __cu2cl_Kernel_move_kernel = clCreateKernel(__cu2cl_Program_redEyeGPU_cu, "move_kernel", &err);
    /*printf("__cu2cl_Kernel_move_kernel creation: %s\n", getErrorString(err)); // Uncomment this line to get error string for the error code returned by clCreateKernel while creating the Kernel: move_kernel*/
}

#include <float.h>
#include <math.h>
#include <stdio.h>

// #include "utils.h"

#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>


// #include "loadSaveImage.h"











static size_t numRowsImg;
static size_t numColsImg;
static size_t templateHalfWidth;
static size_t templateHalfHeight;

static cl_uchar4* inImg;
static cl_uchar4* eyeTemplate;



void loadImageHDR(const std::string& filename,
    float** imagePtr,
    size_t* numRows, size_t* numCols)
{
    cv::Mat image = cv::imread(filename.c_str(), cv::IMREAD_COLOR | cv::IMREAD_ANYDEPTH);
    if (image.empty()) {
        std::cerr << "Couldn't open file: " << filename << std::endl;
        exit(1);
    }

    if (image.channels() != 3) {
        std::cerr << "Image must be color!" << std::endl;
        exit(1);
    }

    if (!image.isContinuous()) {
        std::cerr << "Image isn't continuous!" << std::endl;
        exit(1);
    }

    *imagePtr = new float[image.rows * image.cols * image.channels()];

    float* cvPtr = image.ptr<float>(0);
    for (size_t i = 0; i < image.rows * image.cols * image.channels(); ++i)
        (*imagePtr)[i] = cvPtr[i];

    *numRows = image.rows;
    *numCols = image.cols;
}

void loadImageRGBA(const std::string& filename,
    uchar4** imagePtr,
    size_t* numRows, size_t* numCols)
{
    cv::Mat image = cv::imread(filename);
    if (image.empty()) {
        std::cerr << "Couldn't open file: " << filename << std::endl;
        exit(1);
    }

    if (image.channels() != 3) {
        std::cerr << "Image must be color!" << std::endl;
        exit(1);
    }

    if (!image.isContinuous()) {
        std::cerr << "Image isn't continuous!" << std::endl;
        exit(1);
    }

    cv::Mat imageRGBA;
    cv::cvtColor(image, imageRGBA, cv::COLOR_BGR2RGBA);

    *imagePtr = new uchar4[image.rows * image.cols];

    unsigned char* cvPtr = imageRGBA.ptr<unsigned char>(0);
    for (size_t i = 0; i < image.rows * image.cols; ++i) {
        (*imagePtr)[i].x = cvPtr[4 * i + 0];
        (*imagePtr)[i].y = cvPtr[4 * i + 1];
        (*imagePtr)[i].z = cvPtr[4 * i + 2];
        (*imagePtr)[i].w = cvPtr[4 * i + 3];
    }

    *numRows = image.rows;
    *numCols = image.cols;
}

void saveImageRGBA(const uchar4* const image,
    const size_t numRows, const size_t numCols,
    const std::string& output_file)
{
    int sizes[2];
    sizes[0] = numRows;
    sizes[1] = numCols;
    cv::Mat imageRGBA(2, sizes, CV_8UC4, (void*)image);
    cv::Mat imageOutputBGR;
    cv::cvtColor(imageRGBA, imageOutputBGR, cv::COLOR_RGBA2BGR);
    //output the image
    cv::imwrite(output_file.c_str(), imageOutputBGR);
}

//output an exr file
//assumed to already be BGR
void saveImageHDR(const float* const image,
    const size_t numRows, const size_t numCols,
    const std::string& output_file)
{
    int sizes[2];
    sizes[0] = numRows;
    sizes[1] = numCols;

    cv::Mat imageHDR(2, sizes, CV_32FC3, (void*)image);

    imageHDR = imageHDR * 255;

    cv::imwrite(output_file.c_str(), imageHDR);
}




// we will run 1 exclusive scan, but then when we
// do the move, for zero vals, we iwll take mid - val of scan there




// max size for n/d better one
int get_max_size(int n, int d) {
  return (int)ceil( (float)n / (float)d ) + 1;
}


// host function for radix sort
void radix_sort(cl_mem d_inputVals,
               cl_mem d_inputPos,
               cl_mem d_outputVals,
               cl_mem d_outputPos,
               const size_t numElems)
{
  cl_mem d_bins;
  unsigned int  h_bins[2];
  cl_mem d_scanned;
  cl_mem d_moved;
  const size_t histo_size = 2 * sizeof(unsigned int);
  const size_t arr_size   = numElems * sizeof(unsigned int);

  d_bins = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, histo_size, NULL, &err);
//printf("clCreateBuffer for device variable d_bins: %s\n", getErrorString(err));
  d_scanned = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, arr_size, NULL, &err);
//printf("clCreateBuffer for device variable d_scanned: %s\n", getErrorString(err));
  d_moved = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, arr_size, NULL, &err);
//printf("clCreateBuffer for device variable d_moved: %s\n", getErrorString(err));


  // for histogram kernel defined here
  size_t thread_dim[3] = {1024, 1, 1};
  size_t hist_block_dim[3] = {get_max_size(numElems, thread_dim[0]), 1, 1};


  for (unsigned int pass = 0; pass < 32; pass++) {
    unsigned int one = 1;
    __cu2cl_Memset(d_bins, 0, histo_size);
    __cu2cl_Memset(d_scanned, 0, arr_size);
    __cu2cl_Memset(d_outputVals, 0, arr_size);
    __cu2cl_Memset(d_outputPos, 0, arr_size);

    err = clSetKernelArg(__cu2cl_Kernel_histogram_kernel, 0, sizeof(unsigned int), &pass);
/*printf("clSetKernelArg for argument 0 of kernel __cu2cl_Kernel_histogram_kernel: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_histogram_kernel, 1, sizeof(cl_mem), &d_bins);
/*printf("clSetKernelArg for argument 1 of kernel __cu2cl_Kernel_histogram_kernel: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_histogram_kernel, 2, sizeof(unsigned int *), &d_inputVals);
/*printf("clSetKernelArg for argument 2 of kernel __cu2cl_Kernel_histogram_kernel: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_histogram_kernel, 3, sizeof(int), &numElems);
/*printf("clSetKernelArg for argument 3 of kernel __cu2cl_Kernel_histogram_kernel: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
localWorkSize[0] = thread_dim[0];
localWorkSize[1] = thread_dim[1];
localWorkSize[2] = thread_dim[2];
globalWorkSize[0] = hist_block_dim[0]*localWorkSize[0];
globalWorkSize[1] = hist_block_dim[1]*localWorkSize[1];
globalWorkSize[2] = hist_block_dim[2]*localWorkSize[2];
err = clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_histogram_kernel, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
//printf("clEnqueueNDRangeKernel for the kernel __cu2cl_Kernel_histogram_kernel: %s\n", getErrorString(err));
    err = clFinish(__cu2cl_CommandQueue);
//printf("clFinish return message = %s\n", getErrorString(err));
    // checkCudaErrors(cudaGetLastError());

    err = clEnqueueReadBuffer(__cu2cl_CommandQueue, d_bins, CL_TRUE, 0, histo_size, &h_bins, 0, NULL, NULL);
//printf("Memory copy from device variable &h_bins to host variable d_bins: %s\n", getErrorString(err));

    // printf("debugging %d %d %d %d %d \n", h_bins[0], h_bins[1], h_bins[0] + h_bins[1], numElems, (one << pass));

    for (int i = 0; i < get_max_size(numElems, thread_dim[0]); i++) {
      err = clSetKernelArg(__cu2cl_Kernel_exclusive_scan_kernel, 0, sizeof(unsigned int), &pass);
/*printf("clSetKernelArg for argument 0 of kernel __cu2cl_Kernel_exclusive_scan_kernel: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_exclusive_scan_kernel, 1, sizeof(const unsigned int *), &d_inputVals);
/*printf("clSetKernelArg for argument 1 of kernel __cu2cl_Kernel_exclusive_scan_kernel: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_exclusive_scan_kernel, 2, sizeof(cl_mem), &d_scanned);
/*printf("clSetKernelArg for argument 2 of kernel __cu2cl_Kernel_exclusive_scan_kernel: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_exclusive_scan_kernel, 3, sizeof(int), &numElems);
/*printf("clSetKernelArg for argument 3 of kernel __cu2cl_Kernel_exclusive_scan_kernel: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_exclusive_scan_kernel, 4, sizeof(unsigned int), &i);
/*printf("clSetKernelArg for argument 4 of kernel __cu2cl_Kernel_exclusive_scan_kernel: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_exclusive_scan_kernel, 5, sizeof(unsigned int), &thread_dim[0]);
/*printf("clSetKernelArg for argument 5 of kernel __cu2cl_Kernel_exclusive_scan_kernel: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
localWorkSize[0] = thread_dim[0];
localWorkSize[1] = thread_dim[1];
localWorkSize[2] = thread_dim[2];
globalWorkSize[0] = ( = {1, 1, 1})*localWorkSize[0];
err = clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_exclusive_scan_kernel, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
//printf("clEnqueueNDRangeKernel for the kernel __cu2cl_Kernel_exclusive_scan_kernel: %s\n", getErrorString(err));
      err = clFinish(__cu2cl_CommandQueue);
//printf("clFinish return message = %s\n", getErrorString(err));
      // checkCudaErrors(cudaGetLastError());
    }
    // calculate the move positions
    err = clSetKernelArg(__cu2cl_Kernel_move_kernel, 0, sizeof(unsigned int), &pass);
/*printf("clSetKernelArg for argument 0 of kernel __cu2cl_Kernel_move_kernel: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_move_kernel, 1, sizeof(unsigned int *), &d_inputVals);
/*printf("clSetKernelArg for argument 1 of kernel __cu2cl_Kernel_move_kernel: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_move_kernel, 2, sizeof(unsigned int *), &d_inputPos);
/*printf("clSetKernelArg for argument 2 of kernel __cu2cl_Kernel_move_kernel: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_move_kernel, 3, sizeof(unsigned int *), &d_outputVals);
/*printf("clSetKernelArg for argument 3 of kernel __cu2cl_Kernel_move_kernel: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_move_kernel, 4, sizeof(unsigned int *), &d_outputPos);
/*printf("clSetKernelArg for argument 4 of kernel __cu2cl_Kernel_move_kernel: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_move_kernel, 5, sizeof(cl_mem), &d_moved);
/*printf("clSetKernelArg for argument 5 of kernel __cu2cl_Kernel_move_kernel: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_move_kernel, 6, sizeof(cl_mem), &d_scanned);
/*printf("clSetKernelArg for argument 6 of kernel __cu2cl_Kernel_move_kernel: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_move_kernel, 7, sizeof(unsigned int), &h_bins[0]);
/*printf("clSetKernelArg for argument 7 of kernel __cu2cl_Kernel_move_kernel: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_move_kernel, 8, sizeof(size_t), &numElems);
/*printf("clSetKernelArg for argument 8 of kernel __cu2cl_Kernel_move_kernel: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
localWorkSize[0] = thread_dim[0];
localWorkSize[1] = thread_dim[1];
localWorkSize[2] = thread_dim[2];
globalWorkSize[0] = hist_block_dim[0]*localWorkSize[0];
globalWorkSize[1] = hist_block_dim[1]*localWorkSize[1];
globalWorkSize[2] = hist_block_dim[2]*localWorkSize[2];
err = clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_move_kernel, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
//printf("clEnqueueNDRangeKernel for the kernel __cu2cl_Kernel_move_kernel: %s\n", getErrorString(err));
    err = clFinish(__cu2cl_CommandQueue);
//printf("clFinish return message = %s\n", getErrorString(err)); 
    // checkCudaErrors(cudaGetLastError());
    err = clEnqueueCopyBuffer(__cu2cl_CommandQueue, d_outputVals, d_inputVals, 0, 0, arr_size, 0, NULL, NULL);
//printf("Memory copy from device variable d_outputVals to device variable d_inputVals: %s\n", getErrorString(err));
    err = clEnqueueCopyBuffer(__cu2cl_CommandQueue, d_outputPos, d_inputPos, 0, 0, arr_size, 0, NULL, NULL);
//printf("Memory copy from device variable d_outputPos to device variable d_inputPos: %s\n", getErrorString(err));
    err = clFinish(__cu2cl_CommandQueue);
//printf("clFinish return message = %s\n", getErrorString(err)); 
    // checkCudaErrors(cudaGetLastError());
  }
  clReleaseMemObject(d_moved);
  clReleaseMemObject(d_scanned);
  clReleaseMemObject(d_bins);
}

int main() {
__cu2cl_Init();

    unsigned int* inputVals;
    unsigned int* inputPos;
    unsigned int* outputVals;
    unsigned int* outputPos;

    size_t numElems;


    std::string input_file = "red_eye_effect_5.jpg";
    std::string template_file = "red_eye_effect_template_5.jpg";
    std::string output_file = "fromGPU.jpg";
    std::string reference_file = "fromCPU.jpg";

    // thrust::device_vector<unsigned char> d_red;
    // thrust::device_vector<unsigned char> d_blue;
    // thrust::device_vector<unsigned char> d_green;


    size_t numRowsTemplate, numColsTemplate, numRowsImg, nowColsImg;

    loadImageRGBA(input_file, &inImg, &numRowsImg, &numColsImg);
    loadImageRGBA(template_file, &eyeTemplate, &numRowsTemplate, &numColsTemplate);

    templateHalfWidth = (numColsTemplate - 1) / 2;
    templateHalfHeight = (numRowsTemplate - 1) / 2;

    //we need to split each image into its separate channels
    numElems = numRowsImg * numColsImg;
    size_t templateSize = numRowsTemplate * numColsTemplate;



    uchar* r = new uchar[numElems];
    uchar* g = new uchar[numElems];
    uchar* b = new uchar[numElems];

    cl_mem d_r;
cl_mem d_b;
cl_mem d_g;

    cl_mem d_op_r;

    *(void**)&d_r = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(uchar) * numElems, NULL, &err);
//printf("clCreateBuffer for device variable *(void**)&d_r: %s\n", getErrorString(err));
    *(void**)&d_op_r = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(uchar) * numElems, NULL, &err);
//printf("clCreateBuffer for device variable *(void**)&d_op_r: %s\n", getErrorString(err));
    *(void**)&d_g = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(uchar) * numElems, NULL, &err);
//printf("clCreateBuffer for device variable *(void**)&d_g: %s\n", getErrorString(err));
    *(void**)&d_b = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(uchar) * numElems, NULL, &err);
//printf("clCreateBuffer for device variable *(void**)&d_b: %s\n", getErrorString(err));
    for (size_t i = 0; i < numRowsImg * numColsImg; ++i)
    {
        r[i] = (inImg[i].x);
        g[i] = (inImg[i].y);
        b[i] = (inImg[i].z);
    }
    err = clEnqueueWriteBuffer(__cu2cl_CommandQueue, d_r, CL_TRUE, 0, sizeof(uchar) * numElems, r, 0, NULL, NULL);
//printf("Memory copy from host variable r to device variable d_r: %s\n", getErrorString(err));
    err = clEnqueueWriteBuffer(__cu2cl_CommandQueue, d_op_r, CL_TRUE, 0, sizeof(uchar) * numElems, b, 0, NULL, NULL);
//printf("Memory copy from host variable b to device variable d_op_r: %s\n", getErrorString(err));
    err = clEnqueueWriteBuffer(__cu2cl_CommandQueue, d_b, CL_TRUE, 0, sizeof(uchar) * numElems, b, 0, NULL, NULL);
//printf("Memory copy from host variable b to device variable d_b: %s\n", getErrorString(err));
    err = clEnqueueWriteBuffer(__cu2cl_CommandQueue, d_g, CL_TRUE, 0, sizeof(uchar) * numElems, g, 0, NULL, NULL);
//printf("Memory copy from host variable g to device variable d_g: %s\n", getErrorString(err));
    uchar* rt = new uchar[templateSize];
    uchar* gt = new uchar[templateSize];
    uchar* bt = new uchar[templateSize];
    //cudaMalloc((void**)&r,numElems);
    //cudaMalloc((void**)&g, numElems);
    //cudaMalloc((void**)&b, numElems);
    for (size_t i = 0; i < templateSize; ++i)
    {
        rt[i] = (eyeTemplate[i].x);
        gt[i] = (eyeTemplate[i].y);
        bt[i] = (eyeTemplate[i].z);
    }
    cl_mem d_rt;
cl_mem d_bt;
cl_mem d_gt;

    *(void**)&d_rt = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(uchar) * templateSize, NULL, &err);
//printf("clCreateBuffer for device variable *(void**)&d_rt: %s\n", getErrorString(err));
    *(void**)&d_gt = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(uchar) * templateSize, NULL, &err);
//printf("clCreateBuffer for device variable *(void**)&d_gt: %s\n", getErrorString(err));
    *(void**)&d_bt = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(uchar) * templateSize, NULL, &err);
//printf("clCreateBuffer for device variable *(void**)&d_bt: %s\n", getErrorString(err));
    err = clEnqueueWriteBuffer(__cu2cl_CommandQueue, d_rt, CL_TRUE, 0, sizeof(uchar) * templateSize, r, 0, NULL, NULL);
//printf("Memory copy from host variable r to device variable d_rt: %s\n", getErrorString(err));
    err = clEnqueueWriteBuffer(__cu2cl_CommandQueue, d_bt, CL_TRUE, 0, sizeof(uchar) * templateSize, b, 0, NULL, NULL);
//printf("Memory copy from host variable b to device variable d_bt: %s\n", getErrorString(err));
    err = clEnqueueWriteBuffer(__cu2cl_CommandQueue, d_gt, CL_TRUE, 0, sizeof(uchar) * templateSize, g, 0, NULL, NULL);
//printf("Memory copy from host variable g to device variable d_gt: %s\n", getErrorString(err));

    unsigned int r_sum, b_sum, g_sum;
    r_sum = 0;b_sum = 0;g_sum = 0;
    for (int i = 0;i < numElems;i++)
    {
        r_sum += r[i];
        b_sum += b[i];
        g_sum += g[i];
    }
    unsigned int rt_sum, bt_sum, gt_sum;
    rt_sum = 0;bt_sum = 0;gt_sum = 0;
    for (int i = 0;i < templateSize;i++)
    {
        rt_sum += rt[i];
        bt_sum += bt[i];
        gt_sum += gt[i];
    }

    float r_mean = (double)rt_sum / templateSize;
    float b_mean = (double)bt_sum / templateSize;
    float g_mean = (double)gt_sum / templateSize;

    //printf("this is rmean\n", r_mean);
    //printf("It came through\n");


    const size_t blockSize[3] = {32, 8, 1};
    const size_t gridSize[3] = {(numColsImg + blockSize[0] - 1) / blockSize[0], (numRowsImg + blockSize[1] - 1) / blockSize[1], 1};

    //now compute the cross-correlations for each channel
    cl_mem red_data;
    *(void**)&red_data = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(float) * numElems, NULL, &err);
//printf("clCreateBuffer for device variable *(void**)&red_data: %s\n", getErrorString(err));
    cl_mem blue_data;
    *(void**)&blue_data = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(float) * numElems, NULL, &err);
//printf("clCreateBuffer for device variable *(void**)&blue_data: %s\n", getErrorString(err));
    cl_mem green_data;
    *(void**)&green_data = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(float) * numElems, NULL, &err);
//printf("clCreateBuffer for device variable *(void**)&green_data: %s\n", getErrorString(err));

    naive_normalized_cross_correlation <<<gridSize, blockSize >> > (red_data,
        d_r,
        d_rt,
        numRowsImg, numColsImg,
        templateHalfHeight, numRowsTemplate,
        templateHalfWidth, numColsTemplate,
        numRowsTemplate * numColsTemplate, r_mean);

    err = clFinish(__cu2cl_CommandQueue);
//printf("clFinish return message = %s\n", getErrorString(err));
    // checkCudaErrors(cudaGetLastError());

    naive_normalized_cross_correlation << <gridSize, blockSize >> > (blue_data,
        d_b,
        d_bt,
        numRowsImg, numColsImg,
        templateHalfHeight, numRowsTemplate,
        templateHalfWidth, numColsTemplate,
        numRowsTemplate * numColsTemplate, b_mean);
    err = clFinish(__cu2cl_CommandQueue);
//printf("clFinish return message = %s\n", getErrorString(err)); 
    // checkCudaErrors(cudaGetLastError());

    naive_normalized_cross_correlation << <gridSize, blockSize >> > (green_data,
        d_g,
        d_gt,
        numRowsImg, numColsImg,
        templateHalfHeight, numRowsTemplate,
        templateHalfWidth, numColsTemplate,
        numRowsTemplate * numColsTemplate, g_mean);

    err = clFinish(__cu2cl_CommandQueue);
//printf("clFinish return message = %s\n", getErrorString(err));
     // checkCudaErrors(cudaGetLastError());

    float* h_red_data, * h_blue_data, * h_green_data;
    h_red_data = new float[numElems];
    h_green_data = new float[numElems];
    h_blue_data = new float[numElems];
    err = clEnqueueReadBuffer(__cu2cl_CommandQueue, red_data, CL_TRUE, 0, sizeof(float) * numElems, h_red_data, 0, NULL, NULL);
//printf("Memory copy from device variable h_red_data to host variable red_data: %s\n", getErrorString(err));
    err = clEnqueueReadBuffer(__cu2cl_CommandQueue, blue_data, CL_TRUE, 0, sizeof(float) * numElems, h_blue_data, 0, NULL, NULL);
//printf("Memory copy from device variable h_blue_data to host variable blue_data: %s\n", getErrorString(err));
    err = clEnqueueReadBuffer(__cu2cl_CommandQueue, green_data, CL_TRUE, 0, sizeof(float) * numElems, h_green_data, 0, NULL, NULL);
//printf("Memory copy from device variable h_green_data to host variable green_data: %s\n", getErrorString(err));
    err = clFinish(__cu2cl_CommandQueue);
//printf("clFinish return message = %s\n", getErrorString(err)); 
    // checkCudaErrors(cudaGetLastError());
    float* combined = new float[numElems];
    float mini = 0;




    for (int i = 0;i < numElems;i++)
    {
        combined[i] = h_red_data[i] * h_blue_data[i] * h_green_data[i];
        if (mini > combined[i])
        {
            mini = combined[i];
            //printf("%f is mini", mini);
        }
    }
    printf("%f is mini", mini);
    // find min and add bias so NO VALUE is negetive, easier to deal with


    inputVals = new unsigned int[numElems];
    for (int i = 0;i < numElems;i++)
    {
        //printf("combined val: %d \t", combined[i]);
        combined[i] = (double)combined[i] + (double)(-1 * mini);
        //printf("combined val: %f \t", combined[i]);
        inputVals[i] = combined[i];
    }


    inputPos = new unsigned int[numElems];

    for (int i = 0;i < numElems;i++)
    {
        inputPos[i] = i;
    }

    cl_mem d_inputVals;
    *(void**)&d_inputVals = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(unsigned int)* numElems, NULL, &err);
//printf("clCreateBuffer for device variable *(void**)&d_inputVals: %s\n", getErrorString(err));
    err = clEnqueueWriteBuffer(__cu2cl_CommandQueue, d_inputVals, CL_TRUE, 0, sizeof(unsigned int)* numElems, inputVals, 0, NULL, NULL);
//printf("Memory copy from host variable inputVals to device variable d_inputVals: %s\n", getErrorString(err));

    cl_mem d_inputPos;
    *(void **)&d_inputPos = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(unsigned int)*numElems, NULL, &err);
//printf("clCreateBuffer for device variable *(void **)&d_inputPos: %s\n", getErrorString(err));
    err = clEnqueueWriteBuffer(__cu2cl_CommandQueue, d_inputPos, CL_TRUE, 0, sizeof(unsigned int) * numElems, inputPos, 0, NULL, NULL);
//printf("Memory copy from host variable inputPos to device variable d_inputPos: %s\n", getErrorString(err));

    outputVals = new unsigned int[numElems];
    outputPos = new unsigned int[numElems];

    cl_mem d_outputPos;
cl_mem d_outputVals;
    *(void**)&d_outputPos = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(unsigned int) * numElems, NULL, &err);
//printf("clCreateBuffer for device variable *(void**)&d_outputPos: %s\n", getErrorString(err));
    *(void**)&d_outputVals = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(unsigned int) * numElems, NULL, &err);
//printf("clCreateBuffer for device variable *(void**)&d_outputVals: %s\n", getErrorString(err));
    // printf("before radix");

    radix_sort(d_inputVals,d_inputPos,d_outputVals,d_outputPos,numElems);

    // printf("after radix");

    const size_t block2Size[3] = {256, 1, 1};
    const size_t grid2Size[3] = {(40 + blockSize[0] - 1) / blockSize[0], 1, 1};
    /*for (int i = 0;i < 20;i++)
    {
        printf("i:%d and outputPos: %d\n",i,outputPos[i]);
    }*/

/*CU2CL Note -- Fast-tracked dim3 type without cast*/
/*CU2CL Note -- Inserted temporary variable for kernel literal argument 5!*/
/*CU2CL Note -- Inserted temporary variable for kernel literal argument 8!*/
/*CU2CL Note -- Inserted temporary variable for kernel literal argument 9!*/
    err = clSetKernelArg(__cu2cl_Kernel_remove_redness_from_coordinates, 0, sizeof(cl_mem), &d_outputPos);
/*printf("clSetKernelArg for argument 0 of kernel __cu2cl_Kernel_remove_redness_from_coordinates: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_remove_redness_from_coordinates, 1, sizeof(cl_mem), &d_r);
/*printf("clSetKernelArg for argument 1 of kernel __cu2cl_Kernel_remove_redness_from_coordinates: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_remove_redness_from_coordinates, 2, sizeof(cl_mem), &d_b);
/*printf("clSetKernelArg for argument 2 of kernel __cu2cl_Kernel_remove_redness_from_coordinates: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_remove_redness_from_coordinates, 3, sizeof(cl_mem), &d_g);
/*printf("clSetKernelArg for argument 3 of kernel __cu2cl_Kernel_remove_redness_from_coordinates: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_remove_redness_from_coordinates, 4, sizeof(cl_mem), &d_op_r);
/*printf("clSetKernelArg for argument 4 of kernel __cu2cl_Kernel_remove_redness_from_coordinates: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
int __cu2cl_Kernel_remove_redness_from_coordinates_temp_arg_5 = 40;
err = clSetKernelArg(__cu2cl_Kernel_remove_redness_from_coordinates, 5, sizeof(int), &__cu2cl_Kernel_remove_redness_from_coordinates_temp_arg_5);
/*printf("clSetKernelArg for argument 5 of kernel __cu2cl_Kernel_remove_redness_from_coordinates: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_remove_redness_from_coordinates, 6, sizeof(int), &numRowsImg);
/*printf("clSetKernelArg for argument 6 of kernel __cu2cl_Kernel_remove_redness_from_coordinates: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
err = clSetKernelArg(__cu2cl_Kernel_remove_redness_from_coordinates, 7, sizeof(int), &numColsImg);
/*printf("clSetKernelArg for argument 7 of kernel __cu2cl_Kernel_remove_redness_from_coordinates: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
int __cu2cl_Kernel_remove_redness_from_coordinates_temp_arg_8 = 9;
err = clSetKernelArg(__cu2cl_Kernel_remove_redness_from_coordinates, 8, sizeof(int), &__cu2cl_Kernel_remove_redness_from_coordinates_temp_arg_8);
/*printf("clSetKernelArg for argument 8 of kernel __cu2cl_Kernel_remove_redness_from_coordinates: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
int __cu2cl_Kernel_remove_redness_from_coordinates_temp_arg_9 = 9;
err = clSetKernelArg(__cu2cl_Kernel_remove_redness_from_coordinates, 9, sizeof(int), &__cu2cl_Kernel_remove_redness_from_coordinates_temp_arg_9);
/*printf("clSetKernelArg for argument 9 of kernel __cu2cl_Kernel_remove_redness_from_coordinates: %s\n", getErrorString(err));//Uncomment this for getting error string of the error code returned by clSetKernelArg*/
localWorkSize[0] = block2Size[0];
localWorkSize[1] = block2Size[1];
localWorkSize[2] = block2Size[2];
globalWorkSize[0] = grid2Size[0]*localWorkSize[0];
globalWorkSize[1] = grid2Size[1]*localWorkSize[1];
globalWorkSize[2] = grid2Size[2]*localWorkSize[2];
err = clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_remove_redness_from_coordinates, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
//printf("clEnqueueNDRangeKernel for the kernel __cu2cl_Kernel_remove_redness_from_coordinates: %s\n", getErrorString(err));
    err = clFinish(__cu2cl_CommandQueue);
//printf("clFinish return message = %s\n", getErrorString(err)); 
    // checkCudaErrors(cudaGetLastError());

    uchar* h_op_r = new uchar[numElems];
    err = clEnqueueReadBuffer(__cu2cl_CommandQueue, d_op_r, CL_TRUE, 0, sizeof(uchar) * numElems, h_op_r, 0, NULL, NULL);
//printf("Memory copy from device variable h_op_r to host variable d_op_r: %s\n", getErrorString(err));
    printf("after the kernel\n");

    // combine channels
    cl_uchar4* outputImg = new uchar4[numElems];
    for (int i = 0;i < numElems;i++)
    {
        outputImg[i].x = h_op_r[i];
        outputImg[i].y = g[i];
        outputImg[i].z = b[i];
        outputImg[i].w = 255;
    }


    saveImageRGBA(outputImg, numRowsImg, numColsImg, output_file);

    return 0;
__cu2cl_Cleanup();
}
