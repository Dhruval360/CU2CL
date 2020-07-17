


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










__kernel void sobelGpu(const __global byte* orig, __global byte* gpu, const unsigned int width, const unsigned int height) {
  int x = get_local_id(0) + get_group_id(0) * get_local_size(0);
  int y = get_local_id(1) + get_group_id(1) * get_local_size(1);
  float dx, dy;
  if( x > 0 && y > 0 && x < (width-1) && y < (height-1)) {
    dx = (-1* orig[(y-1)*width + (x-1)]) + (-2*orig[y*width+(x-1)]) + (-1*orig[(y+1)*width+(x-1)]) +
    (    orig[(y-1)*width + (x+1)]) + ( 2*orig[y*width+(x+1)]) + (   orig[(y+1)*width+(x+1)]);
    dy = (    orig[(y-1)*width + (x-1)]) + ( 2*orig[(y-1)*width+x]) + (   orig[(y-1)*width+(x+1)]) +
    (-1* orig[(y+1)*width + (x-1)]) + (-2*orig[(y+1)*width+x]) + (-1*orig[(y+1)*width+(x+1)]);
    gpu[y*width + x] = sqrt((dx*dx) + (dy*dy));
  }
}
