


















//return types are void since any internal error will be handled by quitting
//no point in returning error codes...
//returns a pointer to an RGBA version of the input image
//and a pointer to the single channel grey-scale output
//on both the host and device




//include the definitions of the above functions
__kernel
void rgba_to_greyscale(const __global uchar4* const rgbaImage,
                       __global unsigned char* const greyImage,
                       int numRows, int numCols)
{
  int index_x = get_group_id(0) * get_local_size(0) + get_local_id(0);
  int index_y = get_group_id(1) * get_local_size(1) + get_local_id(1);

  // map the two 2D indices to a single linear, 1D index
  int grid_width = get_num_groups(0) * get_local_size(0);
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




