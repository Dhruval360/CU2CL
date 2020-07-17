#include<iostream>
#include<stdio.h>
#include <math.h>
#include "lodepng.h"
#include <string>
#include <time.h>
#define N 20.0

// nvcc sobelEdgeFilterpng.cu lodepng.cpp -arch sm_20
// Time taken by CPU :  0.95400 ms
// Time taken by GPU :  0.09100 ms


typedef unsigned char byte;

// making a structure for the image with pixels, width and height
struct imgData {
  byte* pixels;
  unsigned int width;
  unsigned int height;
};

imgData loadImage(char* filename) {
  unsigned int width, height;
  byte* rgb;
  unsigned error = lodepng_decode_file(&rgb, &width, &height, filename, LCT_RGBA, 8);
  if (error) {
    printf("LodePNG had an error during file processing. Exiting program.\n");
    // printf("Error code: %u: %s\n", error, lodepng_error_text(error));
    exit(2);
  }
  // we convert to grayscale here
  byte* grayscale = new byte[width * height];
  // pixels stored as rgba
  byte* img = rgb;
  for (int i = 0; i < width * height; ++i) {
    int r = *img++;
    int g = *img++;
    int b = *img++;
    int a = *img++;
    grayscale[i] = 0.3 * r + 0.6 * g + 0.1 * b + 0.5;
  }
  free(rgb);
  imgData a;
  a.pixels = grayscale;a.width = width;a.height = height;
  return a;
}


void writeImage(char* filename, imgData img) {
  std::string newName = filename;
  // checking image coordinates
  // printf("starting to write %d is width and %d is height\n", img.width, img.height);
  unsigned error = lodepng_encode_file(newName.c_str(), img.pixels, img.width, img.height, LCT_GREY, 8);
  if (error) {
    printf("LodePNG had an error during file writing. Exiting program.\n");
    exit(3);
  }
  else
  {
    printf("No error while writing?\n");
  }
  delete [] img.pixels;
}



void sobelCpu(byte* orig, byte* op, const unsigned int width, const unsigned int height) {
  for (int y = 1; y < height - 1; y++) {
    for (int x = 1; x < width - 1; x++) {
      // we find dx and dy
      int dx = (-1 * orig[(y - 1) * width + (x - 1)]) + (-2 * orig[y * width + (x - 1)]) + (-1 * orig[(y + 1) * width + (x - 1)]) +
      (orig[(y - 1) * width + (x + 1)]) + (2 * orig[y * width + (x + 1)]) + (orig[(y + 1) * width + (x + 1)]);
      int dy = (orig[(y - 1) * width + (x - 1)]) + (2 * orig[(y - 1) * width + x]) + (orig[(y - 1) * width + (x + 1)]) +
      (-1 * orig[(y + 1) * width + (x - 1)]) + (-2 * orig[(y + 1) * width + x]) + (-1 * orig[(y + 1) * width + (x + 1)]);
      // sqrt(dx^2 + dy^2) gives the new pixel value
      op[y * width + x] = sqrt((float)((dx * dx) + (dy * dy)));
    }
  }
}

__global__ void sobelGpu(const byte* orig, byte* gpu, const unsigned int width, const unsigned int height) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  float dx, dy;
  if( x > 0 && y > 0 && x < (width-1) && y < (height-1)) {
    dx = (-1* orig[(y-1)*width + (x-1)]) + (-2*orig[y*width+(x-1)]) + (-1*orig[(y+1)*width+(x-1)]) +
    (    orig[(y-1)*width + (x+1)]) + ( 2*orig[y*width+(x+1)]) + (   orig[(y+1)*width+(x+1)]);
    dy = (    orig[(y-1)*width + (x-1)]) + ( 2*orig[(y-1)*width+x]) + (   orig[(y-1)*width+(x+1)]) +
    (-1* orig[(y+1)*width + (x-1)]) + (-2*orig[(y+1)*width+x]) + (-1*orig[(y+1)*width+(x+1)]);
    gpu[y*width + x] = sqrt( (dx*dx) + (dy*dy) );
  }
}



int main()
{
  byte *d_in,*d_out;

  imgData in = loadImage("person.png");
  
  int w = in.width;int h=in.height;
  
  // checking dimensions
  // printf("width is %d, height is %d\n",w,h);
  
  // imgData op(new byte[w*h],w,h);
  // imgData op2(new byte[w*h],w,h);
  imgData op;op.pixels = (byte*)malloc(sizeof(byte)*w*h);
  op.width=w;op.height=h;

  imgData op2;op2.pixels = (byte*)malloc(sizeof(byte)*w*h);
  op2.width=w;op.height=h;


  // debuggin to check about the 
  // writeImage("personcameas.png", in);

  cudaMalloc((void **)&d_in,(w*h));
  cudaMalloc((void **)&d_out,(w*h));

  cudaMemcpy(d_in,in.pixels,w*h,cudaMemcpyHostToDevice);
  cudaMemset(d_out, 0, w*h);

  dim3 threadsPerBlock(N,N,1);
  dim3 numBlocks(ceil(w/N),ceil(h/N),1);


  // struct timeval t1, t2;
  // gettimeofday(&t1, 0);
  sobelCpu(in.pixels,op.pixels,w,h);
  // gettimeofday(&t2, 0);

  // double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;

  // printf("Time taken by CPU :  %3.5f ms \n", time);
  
  // printf("before writing to cpuimg\n");
  writeImage("fromcpu.png", op);

  // printf("after writing to cpuimg\n");
  // writeImage("aftercpu.png", in);
  
  // gettimeofday(&t1, 0);
  sobelGpu<<<numBlocks,threadsPerBlock>>>(d_in,d_out,w,h);
  
  // checking if any errors occured
  cudaError_t cudaerror = cudaDeviceSynchronize(); // waits for completion, returns error code
  if ( cudaDeviceSynchronize() != cudaSuccess ) fprintf( stderr, "Cuda failed to synchronize:\n"); // if error, output error
  // gettimeofday(&t2, 0);
  
  // printf("survived kernel\n");
  // time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
  // printf("Time taken by GPU :  %3.5f ms \n", time);

  // writeImage("aftergpucpupersoncameas.png", in);
  cudaMemcpy(op2.pixels,d_out,(w*h),cudaMemcpyDeviceToHost);
  op2.width=w;
  op2.height=h;
  
  writeImage("fromgpu.png", op2);
  
}

