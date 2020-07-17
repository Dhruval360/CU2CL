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
    int2 image_index_2d = (int2)((blockIdx.x * blockDim.x) + threadIdx.x, (blockIdx.y * blockDim.y) + threadIdx.y);

    ; int2 image_index_2d = (int2)((blockIdx.x * blockDim.x) + threadIdx.x, (blockIdx.y * blockDim.y) + threadIdx.y);
    int  image_index_1d = (nx * image_index_2d.y) + image_index_2d.x;

    if (image_index_2d.x < nx && image_index_2d.y < ny)
    {
        float image_sum = 0.0f;

        for (int y = -template_half_height; y <= template_half_height; y++)
        {
            for (int x = -template_half_width; x <= template_half_width; x++)
            {
                int2 image_offset_index_2d = (int2)(image_index_2d.x + x, image_index_2d.y + y);
                int2 image_offset_index_2d_clamped = (int2)(min(nx - 1, max(0, image_offset_index_2d.x)), min(ny - 1, max(0, image_offset_index_2d.y)));
                int  image_offset_index_1d_clamped = (nx * image_offset_index_2d_clamped.y) + image_offset_index_2d_clamped.x;

                unsigned char image_offset_value = d_original[image_offset_index_1d_clamped];

                image_sum += (float)image_offset_value;
            }
        }

        float image_mean = image_sum / (float)template_size;

        float sum_of_image_template_diff_products = 0.0f;
        float sum_of_squared_image_diffs = 0.0f;
        float sum_of_squared_template_diffs = 0.0f;

        for (int y = -template_half_height; y <= template_half_height; y++)
        {
            for (int x = -template_half_width; x <= template_half_width; x++)
            {
                int2 image_offset_index_2d = (int2)(image_index_2d.x + x, image_index_2d.y + y);
                int2 image_offset_index_2d_clamped = (int2)(min(nx - 1, max(0, image_offset_index_2d.x)), min(ny - 1, max(0, image_offset_index_2d.y)));
                int  image_offset_index_1d_clamped = (nx * image_offset_index_2d_clamped.y) + image_offset_index_2d_clamped.x;

                unsigned char image_offset_value = d_original[image_offset_index_1d_clamped];
                float         image_diff = (float)image_offset_value - image_mean;

                int2 template_index_2d = (int2)(x + template_half_width, y + template_half_height);
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
                int2 image_offset_index_2d = (int2)(x, y);
                int2 image_offset_index_2d_clamped = (int2)(min(nx - 1, max(0, image_offset_index_2d.x)), min(ny - 1, max(0, image_offset_index_2d.y)));
                int  image_offset_index_1d_clamped = (nx * image_offset_index_2d_clamped.y) + image_offset_index_2d_clamped.x;

                unsigned char g_value = d_g[image_offset_index_1d_clamped];
                unsigned char b_value = d_b[image_offset_index_1d_clamped];

                unsigned int gb_average = (g_value + b_value) / 20;
                //printf("heya\t");
                d_r_output[image_offset_index_1d_clamped] = (unsigned char)gb_average;
            }
        }
    }
}




















//output an exr file
//assumed to already be BGR




