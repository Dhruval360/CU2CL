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
cl_program __cu2cl_Program_redGPU_cu;
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
void __cu2cl_Cleanup_redGPU_cu() {
    clReleaseKernel(__cu2cl_Kernel_naive_normalized_cross_correlation);
    clReleaseKernel(__cu2cl_Kernel_remove_redness_from_coordinates);
    clReleaseKernel(__cu2cl_Kernel_histogram_kernel);
    clReleaseKernel(__cu2cl_Kernel_exclusive_scan_kernel);
    clReleaseKernel(__cu2cl_Kernel_move_kernel);
    clReleaseProgram(__cu2cl_Program_redGPU_cu);
}
void __cu2cl_Init_redGPU_cu() {
    #ifdef WITH_ALTERA
    progLen = __cu2cl_LoadProgramSource("redGPU_cu_cl.aocx", &progSrc);
    __cu2cl_Program_redGPU_cu = clCreateProgramWithBinary(__cu2cl_Context, 1, &__cu2cl_Device, &progLen, (const unsigned char **)&progSrc, NULL, NULL);
    #else
    progLen = __cu2cl_LoadProgramSource("redGPU.cu-cl.cl", &progSrc);
    __cu2cl_Program_redGPU_cu = clCreateProgramWithSource(__cu2cl_Context, 1, &progSrc, &progLen, NULL);
    #endif
    free((void *) progSrc);
    clBuildProgram(__cu2cl_Program_redGPU_cu, 1, &__cu2cl_Device, "-I . ", NULL, NULL);
    __cu2cl_Kernel_naive_normalized_cross_correlation = clCreateKernel(__cu2cl_Program_redGPU_cu, "naive_normalized_cross_correlation", NULL);
    __cu2cl_Kernel_remove_redness_from_coordinates = clCreateKernel(__cu2cl_Program_redGPU_cu, "remove_redness_from_coordinates", NULL);
    __cu2cl_Kernel_histogram_kernel = clCreateKernel(__cu2cl_Program_redGPU_cu, "histogram_kernel", NULL);
    __cu2cl_Kernel_exclusive_scan_kernel = clCreateKernel(__cu2cl_Program_redGPU_cu, "exclusive_scan_kernel", NULL);
    __cu2cl_Kernel_move_kernel = clCreateKernel(__cu2cl_Program_redGPU_cu, "move_kernel", NULL);
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
#include <stdio.h>










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
    cl_uchar4** imagePtr,
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

    *imagePtr = new cl_uchar4[image.rows * image.cols];

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

void saveImageRGBA(const cl_uchar4* const image,
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

  *&d_bins = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, histo_size, NULL, NULL);
  *&d_scanned = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, arr_size, NULL, NULL);
  *&d_moved = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, arr_size, NULL, NULL);


  // for histogram kernel defined here
  size_t thread_dim[3] = {1024, 1, 1};
  size_t hist_block_dim[3] = {get_max_size(numElems, thread_dim[0]), 1, 1};


  for (unsigned int pass = 0; pass < 32; pass++) {
    unsigned int one = 1;
    __cu2cl_Memset(d_bins, 0, histo_size);
    __cu2cl_Memset(d_scanned, 0, arr_size);
    __cu2cl_Memset(d_outputVals, 0, arr_size);
    __cu2cl_Memset(d_outputPos, 0, arr_size);

    clSetKernelArg(__cu2cl_Kernel_histogram_kernel, 0, sizeof(unsigned int), &pass);
clSetKernelArg(__cu2cl_Kernel_histogram_kernel, 1, sizeof(cl_mem), &d_bins);
clSetKernelArg(__cu2cl_Kernel_histogram_kernel, 2, sizeof(unsigned int *), &d_inputVals);
clSetKernelArg(__cu2cl_Kernel_histogram_kernel, 3, sizeof(int), &numElems);
localWorkSize[0] = thread_dim[0];
localWorkSize[1] = thread_dim[1];
localWorkSize[2] = thread_dim[2];
globalWorkSize[0] = hist_block_dim[0]*localWorkSize[0];
globalWorkSize[1] = hist_block_dim[1]*localWorkSize[1];
globalWorkSize[2] = hist_block_dim[2]*localWorkSize[2];
clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_histogram_kernel, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    clFinish(__cu2cl_CommandQueue);
    // checkCudaErrors(cudaGetLastError());

    clEnqueueReadBuffer(__cu2cl_CommandQueue, d_bins, CL_TRUE, 0, histo_size, &h_bins, 0, NULL, NULL);

    // printf("debugging %d %d %d %d %d \n", h_bins[0], h_bins[1], h_bins[0] + h_bins[1], numElems, (one << pass));

    for (int i = 0; i < get_max_size(numElems, thread_dim[0]); i++) {
      clSetKernelArg(__cu2cl_Kernel_exclusive_scan_kernel, 0, sizeof(unsigned int), &pass);
clSetKernelArg(__cu2cl_Kernel_exclusive_scan_kernel, 1, sizeof(const unsigned int *), &d_inputVals);
clSetKernelArg(__cu2cl_Kernel_exclusive_scan_kernel, 2, sizeof(cl_mem), &d_scanned);
clSetKernelArg(__cu2cl_Kernel_exclusive_scan_kernel, 3, sizeof(int), &numElems);
clSetKernelArg(__cu2cl_Kernel_exclusive_scan_kernel, 4, sizeof(unsigned int), &i);
clSetKernelArg(__cu2cl_Kernel_exclusive_scan_kernel, 5, sizeof(unsigned int), &thread_dim[0]);
localWorkSize[0] = thread_dim[0];
localWorkSize[1] = thread_dim[1];
localWorkSize[2] = thread_dim[2];
globalWorkSize[0] = hist_block_dim[0];
globalWorkSize[1] = hist_block_dim[1];
globalWorkSize[2] = hist_block_dim[2];
clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_exclusive_scan_kernel, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
      clFinish(__cu2cl_CommandQueue);
      // checkCudaErrors(cudaGetLastError());
    }
    // calculate the move positions
    clSetKernelArg(__cu2cl_Kernel_move_kernel, 0, sizeof(unsigned int), &pass);
clSetKernelArg(__cu2cl_Kernel_move_kernel, 1, sizeof(unsigned int *), &d_inputVals);
clSetKernelArg(__cu2cl_Kernel_move_kernel, 2, sizeof(unsigned int *), &d_inputPos);
clSetKernelArg(__cu2cl_Kernel_move_kernel, 3, sizeof(unsigned int *), &d_outputVals);
clSetKernelArg(__cu2cl_Kernel_move_kernel, 4, sizeof(unsigned int *), &d_outputPos);
clSetKernelArg(__cu2cl_Kernel_move_kernel, 5, sizeof(cl_mem), &d_moved);
clSetKernelArg(__cu2cl_Kernel_move_kernel, 6, sizeof(cl_mem), &d_scanned);
clSetKernelArg(__cu2cl_Kernel_move_kernel, 7, sizeof(unsigned int), &h_bins[0]);
clSetKernelArg(__cu2cl_Kernel_move_kernel, 8, sizeof(size_t), &numElems);
localWorkSize[0] = thread_dim[0];
localWorkSize[1] = thread_dim[1];
localWorkSize[2] = thread_dim[2];
globalWorkSize[0] = hist_block_dim[0]*localWorkSize[0];
globalWorkSize[1] = hist_block_dim[1]*localWorkSize[1];
globalWorkSize[2] = hist_block_dim[2]*localWorkSize[2];
clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_move_kernel, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    clFinish(__cu2cl_CommandQueue); 
    // checkCudaErrors(cudaGetLastError());
    clEnqueueCopyBuffer(__cu2cl_CommandQueue, d_outputVals, d_inputVals, 0, 0, arr_size, 0, NULL, NULL);
    clEnqueueCopyBuffer(__cu2cl_CommandQueue, d_outputPos, d_inputPos, 0, 0, arr_size, 0, NULL, NULL);
    clFinish(__cu2cl_CommandQueue); 
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

    *(void**)&d_r = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(uchar) * numElems, NULL, NULL);
    *(void**)&d_op_r = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(uchar) * numElems, NULL, NULL);
    *(void**)&d_g = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(uchar) * numElems, NULL, NULL);
    *(void**)&d_b = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(uchar) * numElems, NULL, NULL);
    for (size_t i = 0; i < numRowsImg * numColsImg; ++i)
    {
        r[i] = (inImg[i].x);
        g[i] = (inImg[i].y);
        b[i] = (inImg[i].z);
    }
    clEnqueueWriteBuffer(__cu2cl_CommandQueue, d_r, CL_TRUE, 0, sizeof(uchar) * numElems, r, 0, NULL, NULL);
    clEnqueueWriteBuffer(__cu2cl_CommandQueue, d_op_r, CL_TRUE, 0, sizeof(uchar) * numElems, b, 0, NULL, NULL);
    clEnqueueWriteBuffer(__cu2cl_CommandQueue, d_b, CL_TRUE, 0, sizeof(uchar) * numElems, b, 0, NULL, NULL);
    clEnqueueWriteBuffer(__cu2cl_CommandQueue, d_g, CL_TRUE, 0, sizeof(uchar) * numElems, g, 0, NULL, NULL);
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

    *(void**)&d_rt = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(uchar) * templateSize, NULL, NULL);
    *(void**)&d_gt = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(uchar) * templateSize, NULL, NULL);
    *(void**)&d_bt = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(uchar) * templateSize, NULL, NULL);
    clEnqueueWriteBuffer(__cu2cl_CommandQueue, d_rt, CL_TRUE, 0, sizeof(uchar) * templateSize, r, 0, NULL, NULL);
    clEnqueueWriteBuffer(__cu2cl_CommandQueue, d_bt, CL_TRUE, 0, sizeof(uchar) * templateSize, b, 0, NULL, NULL);
    clEnqueueWriteBuffer(__cu2cl_CommandQueue, d_gt, CL_TRUE, 0, sizeof(uchar) * templateSize, g, 0, NULL, NULL);

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
    *(void**)&red_data = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(float) * numElems, NULL, NULL);
    cl_mem blue_data;
    *(void**)&blue_data = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(float) * numElems, NULL, NULL);
    cl_mem green_data;
    *(void**)&green_data = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(float) * numElems, NULL, NULL);

/*CU2CL Note -- Fast-tracked dim3 type without cast*/
/*CU2CL Note -- Inserted temporary variable for kernel literal argument 9!*/
    clSetKernelArg(__cu2cl_Kernel_naive_normalized_cross_correlation, 0, sizeof(cl_mem), &red_data);
clSetKernelArg(__cu2cl_Kernel_naive_normalized_cross_correlation, 1, sizeof(cl_mem), &d_r);
clSetKernelArg(__cu2cl_Kernel_naive_normalized_cross_correlation, 2, sizeof(cl_mem), &d_rt);
clSetKernelArg(__cu2cl_Kernel_naive_normalized_cross_correlation, 3, sizeof(int), &numRowsImg);
clSetKernelArg(__cu2cl_Kernel_naive_normalized_cross_correlation, 4, sizeof(int), &numColsImg);
clSetKernelArg(__cu2cl_Kernel_naive_normalized_cross_correlation, 5, sizeof(int), &templateHalfHeight);
clSetKernelArg(__cu2cl_Kernel_naive_normalized_cross_correlation, 6, sizeof(int), &numRowsTemplate);
clSetKernelArg(__cu2cl_Kernel_naive_normalized_cross_correlation, 7, sizeof(int), &templateHalfWidth);
clSetKernelArg(__cu2cl_Kernel_naive_normalized_cross_correlation, 8, sizeof(int), &numColsTemplate);
int __cu2cl_Kernel_naive_normalized_cross_correlation_temp_arg_9 = numRowsTemplate * numColsTemplate;
clSetKernelArg(__cu2cl_Kernel_naive_normalized_cross_correlation, 9, sizeof(int), &__cu2cl_Kernel_naive_normalized_cross_correlation_temp_arg_9);
clSetKernelArg(__cu2cl_Kernel_naive_normalized_cross_correlation, 10, sizeof(float), &r_mean);
localWorkSize[0] = blockSize[0];
localWorkSize[1] = blockSize[1];
localWorkSize[2] = blockSize[2];
globalWorkSize[0] =gridSize[0];
globalWorkSize[1] = gridSize[1];
globalWorkSize[2] = gridSize[2];
clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_naive_normalized_cross_correlation, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

    clFinish(__cu2cl_CommandQueue);
    // checkCudaErrors(cudaGetLastError());

/*CU2CL Note -- Fast-tracked dim3 type without cast*/
/*CU2CL Note -- Inserted temporary variable for kernel literal argument 9!*/
    clSetKernelArg(__cu2cl_Kernel_naive_normalized_cross_correlation, 0, sizeof(cl_mem), &blue_data);
clSetKernelArg(__cu2cl_Kernel_naive_normalized_cross_correlation, 1, sizeof(cl_mem), &d_b);
clSetKernelArg(__cu2cl_Kernel_naive_normalized_cross_correlation, 2, sizeof(cl_mem), &d_bt);
clSetKernelArg(__cu2cl_Kernel_naive_normalized_cross_correlation, 3, sizeof(int), &numRowsImg);
clSetKernelArg(__cu2cl_Kernel_naive_normalized_cross_correlation, 4, sizeof(int), &numColsImg);
clSetKernelArg(__cu2cl_Kernel_naive_normalized_cross_correlation, 5, sizeof(int), &templateHalfHeight);
clSetKernelArg(__cu2cl_Kernel_naive_normalized_cross_correlation, 6, sizeof(int), &numRowsTemplate);
clSetKernelArg(__cu2cl_Kernel_naive_normalized_cross_correlation, 7, sizeof(int), &templateHalfWidth);
clSetKernelArg(__cu2cl_Kernel_naive_normalized_cross_correlation, 8, sizeof(int), &numColsTemplate);
 __cu2cl_Kernel_naive_normalized_cross_correlation_temp_arg_9 = numRowsTemplate * numColsTemplate;
clSetKernelArg(__cu2cl_Kernel_naive_normalized_cross_correlation, 9, sizeof(int), &__cu2cl_Kernel_naive_normalized_cross_correlation_temp_arg_9);
clSetKernelArg(__cu2cl_Kernel_naive_normalized_cross_correlation, 10, sizeof(float), &b_mean);
localWorkSize[0] = blockSize[0];
localWorkSize[1] = blockSize[1];
localWorkSize[2] = blockSize[2];
globalWorkSize[0] =gridSize[0];
globalWorkSize[1] = gridSize[1];
globalWorkSize[2] = gridSize[2];

clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_naive_normalized_cross_correlation, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    clFinish(__cu2cl_CommandQueue); 
    // checkCudaErrors(cudaGetLastError());

/*CU2CL Note -- Fast-tracked dim3 type without cast*/
/*CU2CL Note -- Inserted temporary variable for kernel literal argument 9!*/
    clSetKernelArg(__cu2cl_Kernel_naive_normalized_cross_correlation, 0, sizeof(cl_mem), &green_data);
clSetKernelArg(__cu2cl_Kernel_naive_normalized_cross_correlation, 1, sizeof(cl_mem), &d_g);
clSetKernelArg(__cu2cl_Kernel_naive_normalized_cross_correlation, 2, sizeof(cl_mem), &d_gt);
clSetKernelArg(__cu2cl_Kernel_naive_normalized_cross_correlation, 3, sizeof(int), &numRowsImg);
clSetKernelArg(__cu2cl_Kernel_naive_normalized_cross_correlation, 4, sizeof(int), &numColsImg);
clSetKernelArg(__cu2cl_Kernel_naive_normalized_cross_correlation, 5, sizeof(int), &templateHalfHeight);
clSetKernelArg(__cu2cl_Kernel_naive_normalized_cross_correlation, 6, sizeof(int), &numRowsTemplate);
clSetKernelArg(__cu2cl_Kernel_naive_normalized_cross_correlation, 7, sizeof(int), &templateHalfWidth);
clSetKernelArg(__cu2cl_Kernel_naive_normalized_cross_correlation, 8, sizeof(int), &numColsTemplate);
 __cu2cl_Kernel_naive_normalized_cross_correlation_temp_arg_9 = numRowsTemplate * numColsTemplate;
clSetKernelArg(__cu2cl_Kernel_naive_normalized_cross_correlation, 9, sizeof(int), &__cu2cl_Kernel_naive_normalized_cross_correlation_temp_arg_9);
clSetKernelArg(__cu2cl_Kernel_naive_normalized_cross_correlation, 10, sizeof(float), &g_mean);
localWorkSize[0] = blockSize[0];
localWorkSize[1] = blockSize[1];
localWorkSize[2] = blockSize[2];
globalWorkSize[0] =gridSize[0];
globalWorkSize[1] = gridSize[1];
globalWorkSize[2] = gridSize[2];

clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_naive_normalized_cross_correlation, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

    clFinish(__cu2cl_CommandQueue);
     // checkCudaErrors(cudaGetLastError());

    float* h_red_data, * h_blue_data, * h_green_data;
    h_red_data = new float[numElems];
    h_green_data = new float[numElems];
    h_blue_data = new float[numElems];
    clEnqueueReadBuffer(__cu2cl_CommandQueue, red_data, CL_TRUE, 0, sizeof(float) * numElems, h_red_data, 0, NULL, NULL);
    clEnqueueReadBuffer(__cu2cl_CommandQueue, blue_data, CL_TRUE, 0, sizeof(float) * numElems, h_blue_data, 0, NULL, NULL);
    clEnqueueReadBuffer(__cu2cl_CommandQueue, green_data, CL_TRUE, 0, sizeof(float) * numElems, h_green_data, 0, NULL, NULL);
    clFinish(__cu2cl_CommandQueue); 
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
    *(void**)&d_inputVals = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(unsigned int)* numElems, NULL, NULL);
    clEnqueueWriteBuffer(__cu2cl_CommandQueue, d_inputVals, CL_TRUE, 0, sizeof(unsigned int)* numElems, inputVals, 0, NULL, NULL);

    cl_mem d_inputPos;
    *(void **)&d_inputPos = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(unsigned int)*numElems, NULL, NULL);
    clEnqueueWriteBuffer(__cu2cl_CommandQueue, d_inputPos, CL_TRUE, 0, sizeof(unsigned int) * numElems, inputPos, 0, NULL, NULL);

    outputVals = new unsigned int[numElems];
    outputPos = new unsigned int[numElems];

    cl_mem d_outputPos;
cl_mem d_outputVals;
    *(void**)&d_outputPos = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(unsigned int) * numElems, NULL, NULL);
    *(void**)&d_outputVals = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(unsigned int) * numElems, NULL, NULL);
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
    clSetKernelArg(__cu2cl_Kernel_remove_redness_from_coordinates, 0, sizeof(cl_mem), &d_outputPos);
clSetKernelArg(__cu2cl_Kernel_remove_redness_from_coordinates, 1, sizeof(cl_mem), &d_r);
clSetKernelArg(__cu2cl_Kernel_remove_redness_from_coordinates, 2, sizeof(cl_mem), &d_b);
clSetKernelArg(__cu2cl_Kernel_remove_redness_from_coordinates, 3, sizeof(cl_mem), &d_g);
clSetKernelArg(__cu2cl_Kernel_remove_redness_from_coordinates, 4, sizeof(cl_mem), &d_op_r);
int __cu2cl_Kernel_remove_redness_from_coordinates_temp_arg_5 = 40;
clSetKernelArg(__cu2cl_Kernel_remove_redness_from_coordinates, 5, sizeof(int), &__cu2cl_Kernel_remove_redness_from_coordinates_temp_arg_5);
clSetKernelArg(__cu2cl_Kernel_remove_redness_from_coordinates, 6, sizeof(int), &numRowsImg);
clSetKernelArg(__cu2cl_Kernel_remove_redness_from_coordinates, 7, sizeof(int), &numColsImg);
int __cu2cl_Kernel_remove_redness_from_coordinates_temp_arg_8 = 9;
clSetKernelArg(__cu2cl_Kernel_remove_redness_from_coordinates, 8, sizeof(int), &__cu2cl_Kernel_remove_redness_from_coordinates_temp_arg_8);
int  __cu2cl_Kernel_remove_redness_from_coordinates_temp_arg_9 = 9;
clSetKernelArg(__cu2cl_Kernel_remove_redness_from_coordinates, 9, sizeof(int), &__cu2cl_Kernel_remove_redness_from_coordinates_temp_arg_9);
localWorkSize[0] = block2Size[0];
localWorkSize[1] = block2Size[1];
localWorkSize[2] = block2Size[2];
globalWorkSize[0] =grid2Size[0];
globalWorkSize[1] = grid2Size[1];
globalWorkSize[2] = grid2Size[2]; 
clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_remove_redness_from_coordinates, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    clFinish(__cu2cl_CommandQueue); 
    // checkCudaErrors(cudaGetLastError());

    uchar* h_op_r = new uchar[numElems];
    clEnqueueReadBuffer(__cu2cl_CommandQueue, d_op_r, CL_TRUE, 0, sizeof(uchar) * numElems, h_op_r, 0, NULL, NULL);
    printf("after the kernel\n");

    // combine channels
    cl_uchar4* outputImg = new cl_uchar4[numElems];
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
