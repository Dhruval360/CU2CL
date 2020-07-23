//
// Median Filter using CUDA
//
#include "opencv2/imgproc/imgproc.hpp-cl.cl"









#define BLOCK_SIZE      16
#define FILTER_WIDTH    3       
#define FILTER_HEIGHT   3       

using namespace std;



// Sort function on device
/*__device__ void sort(unsigned char* filterVector)
{
	 for (int i = 0; i < FILTER_WIDTH*FILTER_HEIGHT; i++) {
	    for (int j = i + 1; j < FILTER_WIDTH*FILTER_HEIGHT; j++) {
		if (filterVector[i] > filterVector[j]) { 
	              //Swap the variables
		      unsigned char tmp = filterVector[i];
		      filterVector[i] = filterVector[j];
		      filterVector[j] = tmp;
		}
             }
         }
}*/



__kernel void medianFilter(__global unsigned char *srcImage, __global unsigned char *dstImage, unsigned int width, unsigned int height, int channel)
{
   int x = get_group_id(0)*get_local_size(0) + get_local_id(0);
   int y = get_group_id(1)*get_local_size(1) + get_local_id(1);

   // only threads inside image will write results
   if((x>=FILTER_WIDTH/2) && (x<(width-FILTER_WIDTH/2)) && (y>=FILTER_HEIGHT/2) && (y<(height-FILTER_HEIGHT/2)))
   {
      for(int c=0 ; c<channel ; c++)   
      {
         unsigned char filterVector[FILTER_WIDTH*FILTER_HEIGHT];     
         // Loop inside the filter to average pixel values
         for(int ky=-FILTER_HEIGHT/2; ky<=FILTER_HEIGHT/2; ky++) {
            for(int kx=-FILTER_WIDTH/2; kx<=FILTER_WIDTH/2; kx++) {
               filterVector[ky*FILTER_WIDTH+kx] = srcImage[((y+ky)*width + (x+kx))*channel+c];
            }
         }
         // Sorting values of filter   
         for (int i = 0; i < FILTER_WIDTH*FILTER_HEIGHT; i++) {
	    for (int j = i + 1; j < FILTER_WIDTH*FILTER_HEIGHT; j++) {
		if (filterVector[i] > filterVector[j]) { 
	              //Swap the variables
		      unsigned char tmp = filterVector[i];
		      filterVector[i] = filterVector[j];
		      filterVector[j] = tmp;
		}
             }
         }
         dstImage[(y*width+x)*channel+c] =  filterVector[(FILTER_WIDTH*FILTER_HEIGHT)/2];
      }
   }
}





// Program main














