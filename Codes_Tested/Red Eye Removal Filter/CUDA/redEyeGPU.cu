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
#include <cuda.h>
#include <cuda_runtime.h>
// #include "loadSaveImage.h"
#include <stdio.h>

__global__ void naive_normalized_cross_correlation(
    float* d_response,
    unsigned char* d_original,
    unsigned char* d_template,
    int            num_pixels_y,
    int            num_pixels_x,
    int            template_half_height,
    int            template_height,
    int            template_half_width,
    int            template_width,
    int            template_size,
    float          template_mean
)
{
    int  ny = num_pixels_y;
    int  nx = num_pixels_x;
    int  knx = template_width;
    int2 image_index_2d = make_int2((blockIdx.x * blockDim.x) + threadIdx.x, (blockIdx.y * blockDim.y) + threadIdx.y);
    int  image_index_1d = (nx * image_index_2d.y) + image_index_2d.x;

    if (image_index_2d.x < nx && image_index_2d.y < ny)
    {
        //
        // compute image mean
        //
        float image_sum = 0.0f;

        for (int y = -template_half_height; y <= template_half_height; y++)
        {
            for (int x = -template_half_width; x <= template_half_width; x++)
            {
                int2 image_offset_index_2d = make_int2(image_index_2d.x + x, image_index_2d.y + y);
                int2 image_offset_index_2d_clamped = make_int2(min(nx - 1, max(0, image_offset_index_2d.x)), min(ny - 1, max(0, image_offset_index_2d.y)));
                int  image_offset_index_1d_clamped = (nx * image_offset_index_2d_clamped.y) + image_offset_index_2d_clamped.x;

                unsigned char image_offset_value = d_original[image_offset_index_1d_clamped];

                image_sum += (float)image_offset_value;
            }
        }

        float image_mean = image_sum / (float)template_size;

        //
        // compute sums
        //
        float sum_of_image_template_diff_products = 0.0f;
        float sum_of_squared_image_diffs = 0.0f;
        float sum_of_squared_template_diffs = 0.0f;

        for (int y = -template_half_height; y <= template_half_height; y++)
        {
            for (int x = -template_half_width; x <= template_half_width; x++)
            {
                int2 image_offset_index_2d = make_int2(image_index_2d.x + x, image_index_2d.y + y);
                int2 image_offset_index_2d_clamped = make_int2(min(nx - 1, max(0, image_offset_index_2d.x)), min(ny - 1, max(0, image_offset_index_2d.y)));
                int  image_offset_index_1d_clamped = (nx * image_offset_index_2d_clamped.y) + image_offset_index_2d_clamped.x;

                unsigned char image_offset_value = d_original[image_offset_index_1d_clamped];
                float         image_diff = (float)image_offset_value - image_mean;

                int2 template_index_2d = make_int2(x + template_half_width, y + template_half_height);
                int  template_index_1d = (knx * template_index_2d.y) + template_index_2d.x;

                unsigned char template_value = d_template[template_index_1d];
                float         template_diff = template_value - template_mean;

                float image_template_diff_product = image_offset_value * template_diff;
                float squared_image_diff = image_diff * image_diff;
                float squared_template_diff = template_diff * template_diff;

                sum_of_image_template_diff_products += image_template_diff_product;
                sum_of_squared_image_diffs += squared_image_diff;
                sum_of_squared_template_diffs += squared_template_diff;
            }
        }


        //
        // compute final result
        //
        float result_value = 0.0f;

        if (sum_of_squared_image_diffs != 0 && sum_of_squared_template_diffs != 0)
        {
            result_value = sum_of_image_template_diff_products / sqrt(sum_of_squared_image_diffs * sum_of_squared_template_diffs);
        }

        d_response[image_index_1d] = result_value;
    }
}


__global__ void remove_redness_from_coordinates(
    const unsigned int* d_coordinates,
    unsigned char* d_r,
    unsigned char* d_b,
    unsigned char* d_g,
    unsigned char* d_r_output,
    int    num_coordinates,
    int    num_pixels_y,
    int    num_pixels_x,
    int    template_half_height,
    int    template_half_width
)
{
    int ny = num_pixels_y;
    int nx = num_pixels_x;
    int global_index_1d = (blockIdx.x * blockDim.x) + threadIdx.x;

    int imgSize = num_pixels_x * num_pixels_y;

    if (global_index_1d < num_coordinates)
    {
        unsigned int image_index_1d = d_coordinates[imgSize - global_index_1d - 1];
        ushort2 image_index_2d = make_ushort2(image_index_1d % num_pixels_x, image_index_1d / num_pixels_x);

        for (int y = image_index_2d.y - template_half_height; y <= image_index_2d.y + template_half_height; y++)
        {
            for (int x = image_index_2d.x - template_half_width; x <= image_index_2d.x + template_half_width; x++)
            {
                int2 image_offset_index_2d = make_int2(x, y);
                int2 image_offset_index_2d_clamped = make_int2(min(nx - 1, max(0, image_offset_index_2d.x)), min(ny - 1, max(0, image_offset_index_2d.y)));
                int  image_offset_index_1d_clamped = (nx * image_offset_index_2d_clamped.y) + image_offset_index_2d_clamped.x;

                unsigned char g_value = d_g[image_offset_index_1d_clamped];
                unsigned char b_value = d_b[image_offset_index_1d_clamped];

                unsigned int gb_average = (g_value + b_value) / 2;
                //printf("heya\t");
                d_r_output[image_offset_index_1d_clamped] = (unsigned char)gb_average;
            }
        }
    }
}





static size_t numRowsImg;
static size_t numColsImg;
static size_t templateHalfWidth;
static size_t templateHalfHeight;

static uchar4* inImg;
static uchar4* eyeTemplate;



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


__global__
void histogram_kernel(unsigned int pass,
                      unsigned int * d_bins,
                      unsigned int* const d_input,
                      const int size) {
  int mid = threadIdx.x + blockDim.x * blockIdx.x;
  if (mid >= size)
    return;
  unsigned int one = 1;
  int bin = ((d_input[mid] & (one << pass)) == (one << pass)) ? 1 : 0;
  if (bin)
    atomicAdd(&d_bins[1], 1);
  else
    atomicAdd(&d_bins[0], 1);
}

// we will run 1 exclusive scan, but then when we
// do the move, for zero vals, we iwll take mid - val of scan there
__global__
void exclusive_scan_kernel(unsigned int pass,
                           unsigned int const * d_inputVals,
                           unsigned int * d_output,
                           const int size,
                           unsigned int base,
                           unsigned int threadSize) {
  int mid = threadIdx.x + threadSize * base;
  int block = threadSize * base;
  unsigned int one = 1;

  if (mid >= size)
    return;

  unsigned int val = 0;
  if (mid > 0)
    val = ((d_inputVals[mid - 1] & (one << pass))  == (one << pass)) ? 1 : 0;
  else
    val = 0;

  d_output[mid] = val;

  __syncthreads();

  for (int s = 1; s <= threadSize; s *= 2) {
    int spot = mid - s;

    if (spot >= 0 && spot >=  threadSize * base)
      val = d_output[spot];
    __syncthreads();
    if (spot >= 0 && spot >= threadSize * base)
      d_output[mid] += val;
    __syncthreads();
  }
  if (base > 0)
    d_output[mid] += d_output[base * threadSize - 1];
}

__global__
void move_kernel(
  unsigned int pass,
  unsigned int* const d_inputVals,
  unsigned int* const d_inputPos,
  unsigned int* d_outputVals,
  unsigned int* d_outputPos,
  unsigned int* d_outputMove,
  unsigned int* const d_scanned,
  unsigned int  one_pos,
  const size_t numElems) {

  int mid = threadIdx.x + blockDim.x * blockIdx.x;
  if (mid >= numElems)
    return;

  unsigned int scan = 0;
  unsigned int base = 0;
  unsigned int one = 1;
  if ( ( d_inputVals[mid] & (one << pass)) == (1 << pass)) {
    scan = d_scanned[mid];
    base = one_pos;
  } else {
    scan = (mid) - d_scanned[mid];
    base = 0;
  }

  d_outputMove[mid] = base + scan;
  d_outputPos[base + scan]  = d_inputPos[mid]; //d_inputPos[0];
  d_outputVals[base + scan] = d_inputVals[mid]; //base+scan;//d_inputVals[0];

}

// max size for n/d better one
int get_max_size(int n, int d) {
  return (int)ceil( (float)n / (float)d ) + 1;
}


// host function for radix sort
void radix_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
  unsigned int* d_bins;
  unsigned int  h_bins[2];
  unsigned int* d_scanned;
  unsigned int* d_moved;
  const size_t histo_size = 2 * sizeof(unsigned int);
  const size_t arr_size   = numElems * sizeof(unsigned int);

  cudaMalloc(&d_bins, histo_size);
  cudaMalloc(&d_scanned, arr_size);
  cudaMalloc(&d_moved, arr_size);


  // for histogram kernel defined here
  dim3 thread_dim(1024);
  dim3 hist_block_dim(get_max_size(numElems, thread_dim.x));


  for (unsigned int pass = 0; pass < 32; pass++) {
    unsigned int one = 1;
    cudaMemset(d_bins, 0, histo_size);
    cudaMemset(d_scanned, 0, arr_size);
    cudaMemset(d_outputVals, 0, arr_size);
    cudaMemset(d_outputPos, 0, arr_size);

    histogram_kernel <<< hist_block_dim, thread_dim>>>(pass, d_bins, d_inputVals, numElems);
    cudaDeviceSynchronize();
    // checkCudaErrors(cudaGetLastError());

    cudaMemcpy(&h_bins, d_bins, histo_size, cudaMemcpyDeviceToHost);

    // printf("debugging %d %d %d %d %d \n", h_bins[0], h_bins[1], h_bins[0] + h_bins[1], numElems, (one << pass));

    for (int i = 0; i < get_max_size(numElems, thread_dim.x); i++) {
      exclusive_scan_kernel <<< dim3(1), thread_dim>>>(pass,d_inputVals,d_scanned,numElems,i,thread_dim.x);
      cudaDeviceSynchronize();
      // checkCudaErrors(cudaGetLastError());
    }
    // calculate the move positions
    move_kernel <<< hist_block_dim, thread_dim>>>(
      pass,
      d_inputVals,
      d_inputPos,
      d_outputVals,
      d_outputPos,
      d_moved,
      d_scanned,
      h_bins[0],
      numElems
    );
    cudaDeviceSynchronize(); 
    // checkCudaErrors(cudaGetLastError());
    cudaMemcpy(d_inputVals, d_outputVals, arr_size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_inputPos, d_outputPos, arr_size, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize(); 
    // checkCudaErrors(cudaGetLastError());
  }
  cudaFree(d_moved);
  cudaFree(d_scanned);
  cudaFree(d_bins);
}

int main() {
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

    uchar* d_r, * d_b, * d_g;

    uchar* d_op_r;

    cudaMalloc((void**)&d_r, sizeof(uchar) * numElems);
    cudaMalloc((void**)&d_op_r, sizeof(uchar) * numElems);
    cudaMalloc((void**)&d_g, sizeof(uchar) * numElems);
    cudaMalloc((void**)&d_b, sizeof(uchar) * numElems);
    for (size_t i = 0; i < numRowsImg * numColsImg; ++i)
    {
        r[i] = (inImg[i].x);
        g[i] = (inImg[i].y);
        b[i] = (inImg[i].z);
    }
    cudaMemcpy(d_r, r, sizeof(uchar) * numElems, cudaMemcpyHostToDevice);
    cudaMemcpy(d_op_r, b, sizeof(uchar) * numElems, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(uchar) * numElems, cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, g, sizeof(uchar) * numElems, cudaMemcpyHostToDevice);
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
    uchar* d_rt, * d_bt, * d_gt;

    cudaMalloc((void**)&d_rt, sizeof(uchar) * templateSize);
    cudaMalloc((void**)&d_gt, sizeof(uchar) * templateSize);
    cudaMalloc((void**)&d_bt, sizeof(uchar) * templateSize);
    cudaMemcpy(d_rt, r, sizeof(uchar) * templateSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bt, b, sizeof(uchar) * templateSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gt, g, sizeof(uchar) * templateSize, cudaMemcpyHostToDevice);

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


    const dim3 blockSize(32, 8, 1);
    const dim3 gridSize((numColsImg + blockSize.x - 1) / blockSize.x, (numRowsImg + blockSize.y - 1) / blockSize.y, 1);

    //now compute the cross-correlations for each channel
    float* red_data;
    cudaMalloc((void**)&red_data, sizeof(float) * numElems);
    float* blue_data;
    cudaMalloc((void**)&blue_data, sizeof(float) * numElems);
    float* green_data;
    cudaMalloc((void**)&green_data, sizeof(float) * numElems);

    naive_normalized_cross_correlation <<<gridSize, blockSize >> > (red_data,
        d_r,
        d_rt,
        numRowsImg, numColsImg,
        templateHalfHeight, numRowsTemplate,
        templateHalfWidth, numColsTemplate,
        numRowsTemplate * numColsTemplate, r_mean);

    cudaDeviceSynchronize();
    // checkCudaErrors(cudaGetLastError());

    naive_normalized_cross_correlation << <gridSize, blockSize >> > (blue_data,
        d_b,
        d_bt,
        numRowsImg, numColsImg,
        templateHalfHeight, numRowsTemplate,
        templateHalfWidth, numColsTemplate,
        numRowsTemplate * numColsTemplate, b_mean);
    cudaDeviceSynchronize(); 
    // checkCudaErrors(cudaGetLastError());

    naive_normalized_cross_correlation << <gridSize, blockSize >> > (green_data,
        d_g,
        d_gt,
        numRowsImg, numColsImg,
        templateHalfHeight, numRowsTemplate,
        templateHalfWidth, numColsTemplate,
        numRowsTemplate * numColsTemplate, g_mean);

    cudaDeviceSynchronize();
     // checkCudaErrors(cudaGetLastError());

    float* h_red_data, * h_blue_data, * h_green_data;
    h_red_data = new float[numElems];
    h_green_data = new float[numElems];
    h_blue_data = new float[numElems];
    cudaMemcpy(h_red_data, red_data, sizeof(float) * numElems, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_blue_data, blue_data, sizeof(float) * numElems, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_green_data, green_data, sizeof(float) * numElems, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize(); 
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

    unsigned int* d_inputVals;
    cudaMalloc((void**)&d_inputVals, sizeof(unsigned int)* numElems);
    cudaMemcpy(d_inputVals, inputVals, sizeof(unsigned int)* numElems, cudaMemcpyHostToDevice);

    unsigned int* d_inputPos;
    cudaMalloc((void **)&d_inputPos,sizeof(unsigned int)*numElems);
    cudaMemcpy(d_inputPos,inputPos, sizeof(unsigned int) * numElems,cudaMemcpyHostToDevice);

    outputVals = new unsigned int[numElems];
    outputPos = new unsigned int[numElems];

    unsigned int* d_outputPos, * d_outputVals;
    cudaMalloc((void**)&d_outputPos, sizeof(unsigned int) * numElems);
    cudaMalloc((void**)&d_outputVals, sizeof(unsigned int) * numElems);
    // printf("before radix");

    radix_sort(d_inputVals,d_inputPos,d_outputVals,d_outputPos,numElems);

    // printf("after radix");

    const dim3 block2Size(256, 1, 1);
    const dim3 grid2Size((40 + blockSize.x - 1) / blockSize.x, 1, 1);
    /*for (int i = 0;i < 20;i++)
    {
        printf("i:%d and outputPos: %d\n",i,outputPos[i]);
    }*/

    remove_redness_from_coordinates <<<grid2Size, block2Size >>> (d_outputPos,
        d_r,
        d_b,
        d_g,
        d_op_r,
        40,
        numRowsImg, numColsImg,
        9, 9);
    cudaDeviceSynchronize(); 
    // checkCudaErrors(cudaGetLastError());

    uchar* h_op_r = new uchar[numElems];
    cudaMemcpy(h_op_r, d_op_r, sizeof(uchar) * numElems, cudaMemcpyDeviceToHost);
    printf("after the kernel\n");

    // combine channels
    uchar4* outputImg = new uchar4[numElems];
    for (int i = 0;i < numElems;i++)
    {
        outputImg[i].x = h_op_r[i];
        outputImg[i].y = g[i];
        outputImg[i].z = b[i];
        outputImg[i].w = 255;
    }


    saveImageRGBA(outputImg, numRowsImg, numColsImg, output_file);

    return 0;
}
