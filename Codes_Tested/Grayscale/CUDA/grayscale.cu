#include <iostream>
#include <string>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

cv::Mat imageRGBA;
cv::Mat imageGrey;

uchar4        *d_rgbaImage__;
unsigned char *d_greyImage__;

size_t numRows() { return imageRGBA.rows; }
size_t numCols() { return imageRGBA.cols; }

//return types are void since any internal error will be handled by quitting
//no point in returning error codes...
//returns a pointer to an RGBA version of the input image
//and a pointer to the single channel grey-scale output
//on both the host and device
void preProcess(uchar4 **inputImage, unsigned char **greyImage,
                uchar4 **d_rgbaImage, unsigned char **d_greyImage,
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
  cudaMalloc(d_rgbaImage, sizeof(uchar4) * numPixels);
  cudaMalloc(d_greyImage, sizeof(unsigned char) * numPixels);
  cudaMemset(*d_greyImage, 0, numPixels * sizeof(unsigned char)); //make sure no memory is left laying around

  //copy input array to the GPU
  cudaMemcpy(*d_rgbaImage, *inputImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice);

  d_rgbaImage__ = *d_rgbaImage;
  d_greyImage__ = *d_greyImage;
}

void postProcess(const std::string& output_file) {
  const int numPixels = numRows() * numCols();
  //copy the output back to the host
  cudaMemcpy(imageGrey.ptr<unsigned char>(0), d_greyImage__, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost);

  //output the image
  cv::imwrite(output_file.c_str(), imageGrey);

  //cleanup
  cudaFree(d_rgbaImage__);
  cudaFree(d_greyImage__);
}

//include the definitions of the above functions
__global__
void rgba_to_greyscale(const uchar4* const rgbaImage,
                       unsigned char* const greyImage,
                       int numRows, int numCols)
{
  int index_x = blockIdx.x * blockDim.x + threadIdx.x;
  int index_y = blockIdx.y * blockDim.y + threadIdx.y;

  // map the two 2D indices to a single linear, 1D index
  int grid_width = gridDim.x * blockDim.x;
  int index = index_y * grid_width + index_x;


  // int c = 0;
  // float xi[15] = {0};
  // // vector<float> xi; //make a temporary list-vector
  // for (int k = index_x - 5 / 2; k <= index_x + 5 / 2; k++) { //apply the window specified by x and y
  //   for (int m = index_y - 5 / 2; m <= index_y + 5 / 2; m++) {
  //     if ((k < 0) || (m < 0)) xi[c] = 0; //on edges of the image use 0 values
  //     else xi[c] = (rgbaImage[k * numCols + m]);
  //     c++;
  //   }
  // }
  // std::sort(std::begin(xi), std::end(xi)); //sort elements of 'xi' neighbourhood vector
  // greyImage[index] = xi[3]; //replace pixel with element specified by 'rank' (3)

  // write out the final result
  greyImage[index] =  .299f * rgbaImage[index].x + .587f * rgbaImage[index].y + .114f * rgbaImage[index].z;

}

void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
                            unsigned char* const d_greyImage, size_t numRows, size_t numCols)
{
  const int thread = 16;
  const dim3 blockSize( thread, thread, 1);
  const dim3 gridSize( ceil(numRows / (float)thread), ceil(numCols / (float)thread), 1);
  rgba_to_greyscale <<< gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);

  cudaDeviceSynchronize();
}

int main() {
  uchar4        *h_rgbaImage, *d_rgbaImage;
  unsigned char *h_greyImage, *d_greyImage;

  std::string input_file;
  std::string output_file;

  input_file  = "sample.jpg";
  output_file = "cuda.jpg";


  //load the image and give us our input and output pointers
  preProcess(&h_rgbaImage, &h_greyImage, &d_rgbaImage, &d_greyImage, input_file);

  // GpuTimer timer;
  // timer.Start();
  //call the grayscale code
  your_rgba_to_greyscale(h_rgbaImage, d_rgbaImage, d_greyImage, numRows(), numCols());
  // timer.Stop();
  cudaDeviceSynchronize();
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
}
