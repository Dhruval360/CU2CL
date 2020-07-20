#define max(a,b) a>b?a:b
#define min(a,b) a<b?a:b

__kernel void box_blur(const __global unsigned char* inputChannel, __global unsigned char* outputChannel, int rows, int cols, int filterWidth, int factor) // This is for square kernels only
{
	// Calculating the coordinates of the pixel
	/*
	int x = get_group_id(0) * get_local_size(0) + get_local_id(0);
	int y = get_group_id(1) * get_local_size(1) + get_local_id(1);
	*/
	int x = get_global_id(0);
	int y = get_global_id(1);
	//printf("x: %u\ty: %u\n", x, y);
	// To prevent trying to access data outside the image
	if (x >= cols || y >= rows)
		return;

	float c = 0.f; // This is a local variable that will hold the sum of pixel values of the neighbouring pixels

	for (int dx = -filterWidth / 2; dx < filterWidth / 2; dx++)
		// dx and dy represent the offset of the neighbouring pixels along the horizontal and vertical axes respectively corresponding to the anchor pixel
	{
		for (int dy = -filterWidth / 2; dy < filterWidth / 2; dy++)
		{
			// xx and yy represent the 2D coordinates of the neighbouring pixels
			int xx = min(max(x + dx, 0), cols - 1); //This takes care of the boundary conditions by extending the image 
			int yy = min(max(y + dy, 0), rows - 1);
			//printf("xx = %u\tyy = %u\n", xx, yy);
			/*
			I have done this based on wikipedia page https://en.wikipedia.org/wiki/Kernel_(image_processing)#Edge_Handling which deals with image processing and  how edges were handled.
			I have followed the following approach here:
			The nearest border pixels are conceptually extended as far as necessary to provide values for the convolution.
			Corner pixels are extended in 90° wedges. Other edge pixels are extended in lines.
			*/
			c += inputChannel[yy * cols + xx];  // Image channels are 1D arrays, hence we need to offset the pixel coordinates to access the pixel in the channel array
			//printf("Pixel %u\n", yy * cols + xx);
		}
	}
	outputChannel[y * cols + x] = c / factor; // Same is the case here as well
}


__kernel void light_edge_detection(const __global unsigned char* inputChannel, __global unsigned char* outputChannel, int rows, int cols)
{
	// Calculating the coordinates of the pixel
	/*
	int x = get_group_id(0) * get_local_size(0) + get_local_id(0);
	int y = get_group_id(1) * get_local_size(1) + get_local_id(1);
	*/
	int x = get_global_id(0);
	int y = get_global_id(1);

	// To prevent trying to access data outside the image
	if (x >= cols || y >= rows)
		return;

	float c = 0.f;
	int filter[] = { -1, -1, -1, -1, 8, -1, -1, -1, 0 };

	for (int dx = 0; dx < 9; dx++) // Here dx - 9/2  is the offset of the neighbouring pixels from the anchor pixel along the horizontal direction
	{
		int xx = x + dx - 9 / 2; // xx is the x coordinate of the neighbouring pixel
		xx = min(max(xx, 0), cols - 1); // Edge case consideration is same as that used for the box filter kernel
		c += (filter[dx] * inputChannel[y * cols + xx]);
	}
	// Again both above and below, image channels are 1D arrays, hence we need to offset the pixel coordinates to access the pixel in the channel array
	outputChannel[y * cols + x] = c;
}



__kernel void separateChannels(const __global uchar4* inputImageRGBA, int rows, int cols, __global unsigned char* redChannel, __global unsigned char* greenChannel, __global unsigned char* blueChannel)
{
	/*
	int x = get_group_id(0) * get_local_size(0) + get_local_id(0);
	int y = get_group_id(1) * get_local_size(1) + get_local_id(1);
	*/
	int x = get_global_id(0);
	int y = get_global_id(1);

	// To prevent trying to access data outside the image
	if (x >= cols || y >= rows)
		return;

	int pixelPosition = y * cols + x; // Image channels are 1D arrays, hence we need to offset the pixel coordinates to access the pixel in the channel array

	//if (pixelPosition == 55) printf("sizeof(inputImageRGBA) = %ld\n", sizeof(inputImageRGBA));

	//printf("Pixel: %u -- %u %u %u %u\n", pixelPosition, inputImageRGBA[pixelPosition].x, inputImageRGBA[pixelPosition].y, inputImageRGBA[pixelPosition].z, inputImageRGBA[pixelPosition].w);

	redChannel[pixelPosition] = inputImageRGBA[pixelPosition].x;
	greenChannel[pixelPosition] = inputImageRGBA[pixelPosition].y;
	blueChannel[pixelPosition] = inputImageRGBA[pixelPosition].z;
}

__kernel void recombineChannels(const __global unsigned char* redChannel, const __global unsigned char* greenChannel, const __global unsigned char* blueChannel, __global uchar4* outputImageRGBA, int rows, int cols)
{
	// Calculating the coordinates of the pixel
	/*
	int x = get_group_id(0) * get_local_size(0) + get_local_id(0);
	int y = get_group_id(1) * get_local_size(1) + get_local_id(1);
	*/
	int x = get_global_id(0);
	int y = get_global_id(1);

	// To prevent trying to access data outside the image
	if (x >= cols || y >= rows)
		return;

	int pixelPosition = y * cols + x; //Image channels are 1D arrays, hence we need to offset the pixel coordinates to access the pixel in the channel array

	unsigned char red = redChannel[pixelPosition];
	unsigned char green = greenChannel[pixelPosition];
	unsigned char blue = blueChannel[pixelPosition];

	//if(x&&y == 0) printf("Pixel %u: RGB: %u %u %u\n", pixelPosition, red, green, blue);
	//printf("Pixel value at (%u,%u): %u %u %u\n", x, y, red, green, blue);

	// Alpha should be 255 for no transparency
	uchar4 outputPixel = { red, green, blue, 255 }; // This combines the red, green, blue and alpha channel values into a vector
	//printf("pixelPosition %u: %u %u %u %u\n", pixelPosition, outputPixel.x, outputPixel.y, outputPixel.z, outputPixel.w);
	outputImageRGBA[pixelPosition] = outputPixel;
}
