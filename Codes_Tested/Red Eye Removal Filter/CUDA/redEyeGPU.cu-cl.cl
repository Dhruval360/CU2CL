



#include "utils.h-cl.cl"









#include "loadSaveImage.h-cl.cl"


__kernel void naive_normalized_cross_correlation(
    __global float* d_response,
    __global unsigned char* d_original,
    __global unsigned char* d_template,
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


__kernel void remove_redness_from_coordinates(
    const __global unsigned int* d_coordinates,
    __global unsigned char* d_r,
    __global unsigned char* d_b,
    __global unsigned char* d_g,
    __global unsigned char* d_r_output,
    int    num_coordinates,
    int    num_pixels_y,
    int    num_pixels_x,
    int    template_half_height,
    int    template_half_width
)
{
    int ny = num_pixels_y;
    int nx = num_pixels_x;
    int global_index_1d = (get_group_id(0) * get_local_size(0)) + get_local_id(0);

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





















//output an exr file
//assumed to already be BGR



__kernel
void histogram_kernel(unsigned int pass,
                      __global unsigned int * d_bins,
                      __global unsigned int* const d_input,
                      const int size) {
  int mid = get_local_id(0) + get_local_size(0) * get_group_id(0);
  if (mid >= size)
    return;
  unsigned int one = 1;
  int bin = ((d_input[mid] & (one << pass)) == (one << pass)) ? 1 : 0;
  if (bin)
    atomic_add(&d_bins[1], 1);
  else
    atomic_add(&d_bins[0], 1);
}

// we will run 1 exclusive scan, but then when we
// do the move, for zero vals, we iwll take mid - val of scan there
__kernel
void exclusive_scan_kernel(unsigned int pass,
                           __global unsigned int const * d_inputVals,
                           __global unsigned int * d_output,
                           const int size,
                           unsigned int base,
                           unsigned int threadSize) {
  int mid = get_local_id(0) + threadSize * base;
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

  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  for (int s = 1; s <= threadSize; s *= 2) {
    int spot = mid - s;

    if (spot >= 0 && spot >=  threadSize * base)
      val = d_output[spot];
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    if (spot >= 0 && spot >= threadSize * base)
      d_output[mid] += val;
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  }
  if (base > 0)
    d_output[mid] += d_output[base * threadSize - 1];
}

__kernel
void move_kernel(
  unsigned int pass,
  __global unsigned int* const d_inputVals,
  __global unsigned int* const d_inputPos,
  __global unsigned int* d_outputVals,
  __global unsigned int* d_outputPos,
  __global unsigned int* d_outputMove,
  __global unsigned int* const d_scanned,
  unsigned int  one_pos,
  const size_t numElems) {

  int mid = get_local_id(0) + get_local_size(0) * get_group_id(0);
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



// host function for radix sort


