
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"

#define BLOCK_SIZE      16
#define FILTER_WIDTH    3       
#define FILTER_HEIGHT   3       

using namespace std;

// Run Laplacian Filter on GPU
__global__ void laplacianFilter(unsigned char *srcImage, unsigned char *dstImage, unsigned int width, unsigned int height)
{
   int x = blockIdx.x*blockDim.x + threadIdx.x;
   int y = blockIdx.y*blockDim.y + threadIdx.y;

   float kernel[3][3] = {0, -1, 0, -1, 4, -1, 0, -1, 0};
   //float kernel[3][3] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};   
   // only threads inside image will write results
   if((x>=FILTER_WIDTH/2) && (x<(width-FILTER_WIDTH/2)) && (y>=FILTER_HEIGHT/2) && (y<(height-FILTER_HEIGHT/2)))
   {
         // Sum of pixel values 
         float sum = 0;
         // Loop inside the filter to average pixel values
         for(int ky=-FILTER_HEIGHT/2; ky<=FILTER_HEIGHT/2; ky++) {
            for(int kx=-FILTER_WIDTH/2; kx<=FILTER_WIDTH/2; kx++) {
               float fl = srcImage[((y+ky)*width + (x+kx))]; 
               sum += fl*kernel[ky+FILTER_HEIGHT/2][kx+FILTER_WIDTH/2];
            }
         }
         dstImage[(y*width+x)] =  sum;
   }
}



void laplacianFilter_GPU_wrapper(const cv::Mat& input, cv::Mat& output);
  

using namespace std;


// Program main
int main() {

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
}














