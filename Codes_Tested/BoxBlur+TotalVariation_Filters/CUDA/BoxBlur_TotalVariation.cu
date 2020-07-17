#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<time.h>

#include <stdio.h>

//OpenCV stuff
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;

// I learnt about vector notaion in c++ and now know that uchar4* is a 2D array of 4 element vectors of unsigned char type.  

__global__ void box_blur(const unsigned char* inputChannel, unsigned char* outputChannel, int rows, int cols, int filterWidth, int factor) // This is for square kernels only
{
	// Calculating the coordinates of the pixel
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

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
			/*
			I have done this based on wikipedia page https://en.wikipedia.org/wiki/Kernel_(image_processing)#Edge_Handling which deals with image processing and  how edges were handled.
			I have followed the following approach here:
			The nearest border pixels are conceptually extended as far as necessary to provide values for the convolution.
			Corner pixels are extended in 90° wedges. Other edge pixels are extended in lines.
			*/
			c += inputChannel[yy * cols + xx];  // Image channels are 1D arrays, hence we need to offset the pixel coordinates to access the pixel in the channel array
		}
	}
	outputChannel[y * cols + x] = c / factor; // Same is the case here as well
}

void serial_box_blur(const Mat inputImage, Mat outputImage, int rows, int cols, int filterWidth, int factor, int channels)
{
	// For every pixel in the image
	for (int x = 0; x < cols; x++)
	{
		for (int y = 0; y < rows; y++)
		{
			float b = 0.f, g = 0.f, r = 0.f;
			// For every neighbouring pixel (based on the filter's width) around the pixel at (x,y)
			for (int dx = -filterWidth / 2; dx <= filterWidth / 2; dx++)
			{
				for (int dy = -filterWidth / 2; dy <= filterWidth / 2; dy++)
					// dx and dy represent the offset of the neighbouring pixels along the horizontal and vertical axes respectively corresponding to the anchor pixel
				{
					int yy = min(max(y + dy, 0), rows - 1);
					int xx = min(max(x + dx, 0), cols - 1);
					b += inputImage.data[channels * (cols * yy + xx) + 0];
					g += inputImage.data[channels * (cols * yy + xx) + 1];
					r += inputImage.data[channels * (cols * yy + xx) + 2];
				}
			}
			outputImage.data[channels * (cols * y + x) + 0] = b / factor;
			outputImage.data[channels * (cols * y + x) + 1] = g / factor;
			outputImage.data[channels * (cols * y + x) + 2] = r / factor;
		}
	}
}

__global__ void light_edge_detection(const unsigned char* inputChannel, unsigned char* outputChannel, int rows, int cols)
{
	// Calculating the coordinates of the pixel
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

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

void serial_light_edge_detection(const Mat inputImage, Mat outputImage, int rows, int cols, int channels)
{
	int filter[] = { -1, -1, -1, -1, 8, -1, -1, -1, 0 };
	// For every pixel in the image
	for (int x = 0; x < cols; x++)
	{
		for (int y = 0; y < rows; y++)
		{
			float b = 0.f, g = 0.f, r = 0.f;
			for (int dx = 0; dx < 9; dx++) // Here dx - 9/2  is the offset of the neighbouring pixels from the anchor pixel along the horizontal direction
			{
				int xx = x + dx - 9 / 2; // xx is the x coordinate of the neighbouring pixel
				xx = min(max(xx, 0), cols - 1); // Edge case consideration is same as that used for the box filter kernel
				b += filter[dx] * inputImage.data[channels * (cols * y + xx) + 0];
				g += filter[dx] * inputImage.data[channels * (cols * y + xx) + 1];
				r += filter[dx] * inputImage.data[channels * (cols * y + xx) + 2];
			}
			outputImage.data[channels * (cols * y + x) + 0] = b;
			outputImage.data[channels * (cols * y + x) + 1] = g;
			outputImage.data[channels * (cols * y + x) + 2] = r;
		}
	}
}

__global__ void separateChannels(const uchar4* inputImageRGBA, int rows, int cols, unsigned char* redChannel, unsigned char* greenChannel, unsigned char* blueChannel)
{
	// Calculating the coordinates of the pixel
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// To prevent trying to access data outside the image
	if (x >= cols || y >= rows)
		return;

	int pixelPosition = y * cols + x; // Image channels are 1D arrays, hence we need to offset the pixel coordinates to access the pixel in the channel array

	redChannel[pixelPosition] = inputImageRGBA[pixelPosition].x;
	greenChannel[pixelPosition] = inputImageRGBA[pixelPosition].y;
	blueChannel[pixelPosition] = inputImageRGBA[pixelPosition].z;
}

__global__ void recombineChannels(const unsigned char* redChannel, const unsigned char* greenChannel, const unsigned char* blueChannel, uchar4* outputImageRGBA, int rows, int cols)
{
	// Calculating the coordinates of the pixel
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// To prevent trying to access data outside the image
	if (x >= cols || y >= rows)
		return;

	int pixelPosition = y * cols + x; //Image channels are 1D arrays, hence we need to offset the pixel coordinates to access the pixel in the channel array

	unsigned char red = redChannel[pixelPosition];
	unsigned char green = greenChannel[pixelPosition];
	unsigned char blue = blueChannel[pixelPosition];

	// Alpha should be 255 for no transparency
	uchar4 outputPixel = make_uchar4(red, green, blue, 255); // This combines the red, green, blue and alpha channel values into a vector
	outputImageRGBA[pixelPosition] = outputPixel;
}

int main()
{
	char input_file[] = "1/Original.jpg";

	cv::Mat image = cv::imread(input_file, cv::IMREAD_COLOR);
	if (image.empty())
	{
		printf("Couldn't open the file %s\n", input_file);
		exit(1);
	}

	char output_file[] = "Images set 1/Blurred_GPU.jpg";
	char output_file2[] = "Images set 1/TotalVariationFilter_GPU.jpg";
	char output_file3[] = "Images set 1/Blurred_CPU.jpg";
	char output_file4[] = "Images set 1/TotalVariationFilter_CPU.jpg";

	int filterWidth = 9; // For the box blur
	int divFactor = filterWidth * filterWidth; // For dividing the sum of neighbouring pixel values after summation for the box filter for normalization

	// For all the variable names I have used the convention I learnt from the udacity course that h_ represents host (CPU) variable and d_ represents device (GPU) variable
	uchar4* h_inputImageRGBA, * d_inputImageRGBA;
	cv::Mat inputImageRGBA;

	// For box blur
	uchar4* d_outputImageRGBA;
	unsigned char* d_redBlurred, * d_greenBlurred, * d_blueBlurred;
	unsigned char* d_red, * d_green, * d_blue;
	cv::Mat outputImageRGBA;

	// For light edge
	uchar4* d_outputImageRGBA2;
	unsigned char* d_redlight, * d_greenlight, * d_bluelight;
	cv::Mat outputImageRGBA2; // Light edge filter application

	int cols = image.cols;
	int rows = image.rows;
	int totalPixels = cols * rows;
	int channels = image.channels();
	
	// For the serial code
	Mat CPUoutput1, CPUoutput2;
	CPUoutput1 = image.clone();
	CPUoutput2 = image.clone();

	clock_t startcpu, endcpu;
	double cpu_time_used;
	startcpu = clock();

	serial_box_blur(image, CPUoutput1, rows, cols, filterWidth, divFactor, channels);
	serial_light_edge_detection(image, CPUoutput2, rows, cols, channels);

	endcpu = clock();
	cpu_time_used = (((double)(endcpu - startcpu)) / CLOCKS_PER_SEC)*1000; // For milli seconds
	printf("Total time taken for both filters for image of size %d,%d on CPU: %lf ms\n", cols, rows, cpu_time_used);

	cv::imwrite(output_file3, CPUoutput1);
	cv::imwrite(output_file4, CPUoutput2);
	
	/*
	// I have compressed this 9 by 8 matrix into a 1D array and have used that instead of this
	int h_lightEdgeFilter[lightKernelWidth * lightKernelHeight] = { 0 }; // Initializing the light edge filter on the host
	for (int c = 0; c <= lightKernelWidth; c++)
		{
		for (int r = 0; r <= lightKernelHeight; r++)
			{
			if (r == 4) h_lightEdgeFilter[c * filterWidth + r] = 1;
			else if (r == c) h_lightEdgeFilter[c * filterWidth + r] = -1;
			}
		}
	*/

	cv::cvtColor(image, inputImageRGBA, cv::COLOR_BGR2BGRA);

	// Allocating memory for the outputs
	outputImageRGBA.create(rows, cols, CV_8UC4);
	outputImageRGBA2.create(rows, cols, CV_8UC4);

	h_inputImageRGBA = (uchar4*)inputImageRGBA.ptr<unsigned char>(0);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	cudaStream_t s1, s2, s3, s4, s5, s6; // For parallelizing memory copies and kernel launches
	cudaStreamCreate(&s1); cudaStreamCreate(&s2); cudaStreamCreate(&s3);
	cudaStreamCreate(&s4); cudaStreamCreate(&s5); cudaStreamCreate(&s6);

	// Allotting memory for splitting the image into its different channels in GPU
	cudaMalloc(&d_red, sizeof(unsigned char) * totalPixels);
	cudaMalloc(&d_green, sizeof(unsigned char) * totalPixels);
	cudaMalloc(&d_blue, sizeof(unsigned char) * totalPixels);

	// Alloting memory for the output images in the GPU
	cudaMalloc(&d_inputImageRGBA, sizeof(uchar4) * totalPixels);
	cudaMalloc(&d_outputImageRGBA, sizeof(uchar4) * totalPixels);
	cudaMalloc(&d_outputImageRGBA2, sizeof(uchar4) * totalPixels);
	cudaMemcpyAsync(d_inputImageRGBA, h_inputImageRGBA, sizeof(uchar4) * totalPixels, cudaMemcpyHostToDevice, s1);

	// Alloting memory for each output channel on the GPU
	// For box blur
	cudaMalloc(&d_redBlurred, sizeof(unsigned char) * totalPixels);
	cudaMalloc(&d_greenBlurred, sizeof(unsigned char) * totalPixels);
	cudaMalloc(&d_blueBlurred, sizeof(unsigned char) * totalPixels);

	// For light edge filter
	cudaMalloc(&d_redlight, sizeof(unsigned char) * totalPixels);
	cudaMalloc(&d_greenlight, sizeof(unsigned char) * totalPixels);
	cudaMalloc(&d_bluelight, sizeof(unsigned char) * totalPixels);

	cudaDeviceSynchronize();

	const dim3 blockSize(32, 32, 1);
	const dim3 gridSize((cols / blockSize.x) + 1, (rows / blockSize.y) + 1, 1);

	separateChannels <<<gridSize, blockSize>>> (d_inputImageRGBA, rows, cols, d_red, d_green, d_blue);
	cudaDeviceSynchronize();


	//I have made processing of each channel to be run on different streams which gave me a significant speedup of 40% over running all on the same stream 
	box_blur <<<gridSize, blockSize, 0, s1>>> (d_red, d_redBlurred, rows, cols, filterWidth, divFactor);
	box_blur <<<gridSize, blockSize, 0, s2>>> (d_green, d_greenBlurred, rows, cols, filterWidth, divFactor);
	box_blur <<<gridSize, blockSize, 0, s3>>> (d_blue, d_blueBlurred, rows, cols, filterWidth, divFactor);

	light_edge_detection <<<gridSize, blockSize, 0, s4>>> (d_red, d_redlight, rows, cols);
	light_edge_detection <<<gridSize, blockSize, 0, s5>>> (d_green, d_greenlight, rows, cols);
	light_edge_detection <<<gridSize, blockSize, 0, s6>>> (d_blue, d_bluelight, rows, cols);

	cudaDeviceSynchronize();

	recombineChannels <<<gridSize, blockSize, 0, s1>>> (d_redBlurred, d_greenBlurred, d_blueBlurred, d_outputImageRGBA, rows, cols);
	recombineChannels <<<gridSize, blockSize, 0, s2>>> (d_redlight, d_greenlight, d_bluelight, d_outputImageRGBA2, rows, cols);

	cudaDeviceSynchronize();

	cudaMemcpyAsync(outputImageRGBA.ptr<unsigned char>(0), d_outputImageRGBA, sizeof(uchar4) * totalPixels, cudaMemcpyDeviceToHost, s1);
	cudaMemcpyAsync(outputImageRGBA2.ptr<unsigned char>(0), d_outputImageRGBA2, sizeof(uchar4) * totalPixels, cudaMemcpyDeviceToHost, s2);

	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);

	printf("Total time taken for both filters for image of size %d,%d on GPU: %f ms\n", cols, rows, milliseconds);
	printf("The blur kernel used was %d,%d\n", filterWidth, filterWidth);

	cv::imwrite(output_file, outputImageRGBA);
	cv::imwrite(output_file2, outputImageRGBA2);

	printf("Process complete\n");

	cudaFree(d_inputImageRGBA);	cudaFree(d_outputImageRGBA);
	cudaFree(d_redBlurred);	cudaFree(d_red); cudaFree(d_redlight);
	cudaFree(d_greenBlurred); cudaFree(d_green); cudaFree(d_greenlight);
	cudaFree(d_blueBlurred); cudaFree(d_blue); cudaFree(d_bluelight);
	cudaStreamDestroy(s1); cudaStreamDestroy(s2); cudaStreamDestroy(s3); cudaStreamDestroy(s4); cudaStreamDestroy(s5); cudaStreamDestroy(s6);

	return 0;
}

