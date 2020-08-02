
#include <opencv2/imgproc/imgproc.hpp>








#define BLOCK_SIZE      16
#define FILTER_WIDTH    3       
#define FILTER_HEIGHT   3       

/*CU2CL Warning: using namespace not allowed in the OpenCL Kernel File*/ 
 // using namespace std;


// Run Laplacian Filter on GPU
__kernel void laplacianFilter(__global unsigned char *srcImage, __global unsigned char *dstImage, unsigned int width, unsigned int height)
{
   int x = get_group_id(0)*get_local_size(0) + get_local_id(0);
   int y = get_group_id(1)*get_local_size(1) + get_local_id(1);

   float ker[3][3] = {0, -1, 0, -1, 4, -1, 0, -1, 0};
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
               sum += fl*ker[ky+FILTER_HEIGHT/2][kx+FILTER_WIDTH/2];
            }
         }
         dstImage[(y*width+x)] =  sum;
   }
}




  

/*CU2CL Warning: using namespace not allowed in the OpenCL Kernel File*/ 
 // using namespace std;


// Program main






