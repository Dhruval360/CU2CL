#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include "cu2cl_util.h"

#include<time.h>

#include <stdio.h>

//OpenCV stuff
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

//I manually moved the headers to the top

cl_kernel __cu2cl_Kernel_box_blur;
cl_kernel __cu2cl_Kernel_light_edge_detection;
cl_kernel __cu2cl_Kernel_separateChannels;
cl_kernel __cu2cl_Kernel_recombineChannels;
cl_program __cu2cl_Program_BoxBlur_TotalVariation_cu;
extern const char *progSrc;
extern size_t progLen;

extern cl_platform_id __cu2cl_Platform;
extern cl_device_id __cu2cl_Device;
extern cl_context __cu2cl_Context;
extern cl_command_queue __cu2cl_CommandQueue;

extern size_t globalWorkSize[3];
extern size_t localWorkSize[3];
void __cu2cl_Cleanup_BoxBlur_TotalVariation_cu() {
    clReleaseKernel(__cu2cl_Kernel_box_blur);
    clReleaseKernel(__cu2cl_Kernel_light_edge_detection);
    clReleaseKernel(__cu2cl_Kernel_separateChannels);
    clReleaseKernel(__cu2cl_Kernel_recombineChannels);
    clReleaseProgram(__cu2cl_Program_BoxBlur_TotalVariation_cu);
}
void __cu2cl_Init_BoxBlur_TotalVariation_cu() {
    #ifdef WITH_ALTERA
    progLen = __cu2cl_LoadProgramSource("BoxBlur_TotalVariation_cu_cl.aocx", &progSrc);
    __cu2cl_Program_BoxBlur_TotalVariation_cu = clCreateProgramWithBinary(__cu2cl_Context, 1, &__cu2cl_Device, &progLen, (const unsigned char **)&progSrc, NULL, NULL);
    #else
    progLen = __cu2cl_LoadProgramSource("BoxBlur_TotalVariation.cu-cl.cl", &progSrc);
    __cu2cl_Program_BoxBlur_TotalVariation_cu = clCreateProgramWithSource(__cu2cl_Context, 1, &progSrc, &progLen, NULL);
    #endif
    free((void *) progSrc);
    clBuildProgram(__cu2cl_Program_BoxBlur_TotalVariation_cu, 1, &__cu2cl_Device, "-I . ", NULL, NULL);
    __cu2cl_Kernel_box_blur = clCreateKernel(__cu2cl_Program_BoxBlur_TotalVariation_cu, "box_blur", NULL);
    __cu2cl_Kernel_light_edge_detection = clCreateKernel(__cu2cl_Program_BoxBlur_TotalVariation_cu, "light_edge_detection", NULL);
    __cu2cl_Kernel_separateChannels = clCreateKernel(__cu2cl_Program_BoxBlur_TotalVariation_cu, "separateChannels", NULL);
    __cu2cl_Kernel_recombineChannels = clCreateKernel(__cu2cl_Program_BoxBlur_TotalVariation_cu, "recombineChannels", NULL);
}

cl_int err;
const char *getErrorString(cl_int error){
switch(error){
// run-time and JIT compiler errors
	case 0: return "CL_SUCCESS";
	case -1: return "CL_DEVICE_NOT_FOUND";
	case -2: return "CL_DEVICE_NOT_AVAILABLE";
	case -3: return "CL_COMPILER_NOT_AVAILABLE";
	case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	case -5: return "CL_OUT_OF_RESOURCES";
	case -6: return "CL_OUT_OF_HOST_MEMORY";
	case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
	case -8: return "CL_MEM_COPY_OVERLAP";
	case -9: return "CL_IMAGE_FORMAT_MISMATCH";
	case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case -11: return "CL_BUILD_PROGRAM_FAILURE";
	case -12: return "CL_MAP_FAILURE";
	case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
	case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
	case -15: return "CL_COMPILE_PROGRAM_FAILURE";
	case -16: return "CL_LINKER_NOT_AVAILABLE";
	case -17: return "CL_LINK_PROGRAM_FAILURE";
	case -18: return "CL_DEVICE_PARTITION_FAILED";
	case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
	// compile-time errors
	case -30: return "CL_INVALID_VALUE";
	case -31: return "CL_INVALID_DEVICE_TYPE";
	case -32: return "CL_INVALID_PLATFORM";
	case -33: return "CL_INVALID_DEVICE";
	case -34: return "CL_INVALID_CONTEXT";
	case -35: return "CL_INVALID_QUEUE_PROPERTIES";
	case -36: return "CL_INVALID_COMMAND_QUEUE";
	case -37: return "CL_INVALID_HOST_PTR";
	case -38: return "CL_INVALID_MEM_OBJECT";
	case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	case -40: return "CL_INVALID_IMAGE_SIZE";
	case -41: return "CL_INVALID_SAMPLER";
	case -42: return "CL_INVALID_BINARY";
	case -43: return "CL_INVALID_BUILD_OPTIONS";
	case -44: return "CL_INVALID_PROGRAM";
	case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
	case -46: return "CL_INVALID_KERNEL_NAME";
	case -47: return "CL_INVALID_KERNEL_DEFINITION";
	case -48: return "CL_INVALID_KERNEL";
	case -49: return "CL_INVALID_ARG_INDEX";
	case -50: return "CL_INVALID_ARG_VALUE";
	case -51: return "CL_INVALID_ARG_SIZE";
	case -52: return "CL_INVALID_KERNEL_ARGS";
	case -53: return "CL_INVALID_WORK_DIMENSION";
	case -54: return "CL_INVALID_WORK_GROUP_SIZE";
	case -55: return "CL_INVALID_WORK_ITEM_SIZE";
	case -56: return "CL_INVALID_GLOBAL_OFFSET";
	case -57: return "CL_INVALID_EVENT_WAIT_LIST";
	case -58: return "CL_INVALID_EVENT";
	case -59: return "CL_INVALID_OPERATION";
	case -60: return "CL_INVALID_GL_OBJECT";
	case -61: return "CL_INVALID_BUFFER_SIZE";
	case -62: return "CL_INVALID_MIP_LEVEL";
	case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
	case -64: return "CL_INVALID_PROPERTY";
	case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
	case -66: return "CL_INVALID_COMPILER_OPTIONS";
	case -67: return "CL_INVALID_LINKER_OPTIONS";
	case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
	// extension errors
	case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
	case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
	case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
	case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
	case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
	case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
	default: return "Unknown OpenCL error";
}
}

using namespace cv;

// I learnt about vector notaion in c++ and now know that cl_uchar4* is a 2D array of 4 element vectors of unsigned char type.  



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

int main()
{
__cu2cl_Init();

	char input_file[] = "1/Original.jpg";

	cv::Mat image = cv::imread(input_file, cv::IMREAD_COLOR);
	if (image.empty())
	{
		printf("Couldn't open the file %s\n", input_file);
		exit(1);
	}

	char output_file[] = "1/Blurred_GPU.jpg";
	char output_file2[] = "1/TotalVariationFilter_GPU.jpg";
	char output_file3[] = "1/Blurred_CPU.jpg";
	char output_file4[] = "1/TotalVariationFilter_CPU.jpg";

	int filterWidth = 9; // For the box blur
	int divFactor = filterWidth * filterWidth; // For dividing the sum of neighbouring pixel values after summation for the box filter for normalization

	// For all the variable names I have used the convention I learnt from the udacity course that h_ represents host (CPU) variable and d_ represents device (GPU) variable
	cl_uchar4 *h_inputImageRGBA;
	cl_mem d_inputImageRGBA;
	cv::Mat inputImageRGBA;

	// For box blur
	cl_mem d_outputImageRGBA;
	cl_mem d_redBlurred;
	cl_mem d_greenBlurred;
	cl_mem d_blueBlurred;
	cl_mem d_red;
	cl_mem d_green;
	cl_mem d_blue;
	cv::Mat outputImageRGBA;

	// For light edge
	cl_mem d_outputImageRGBA2;
	cl_mem d_redlight;
	cl_mem d_greenlight;
	cl_mem d_bluelight;
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

	h_inputImageRGBA = (cl_uchar4*)inputImageRGBA.ptr<unsigned char>(0);

	cl_event start, stop;
	start = clCreateUserEvent(__cu2cl_Context, &err);
	printf("Event start creation: %s\n", getErrorString(err));

	stop = clCreateUserEvent(__cu2cl_Context, &err);
	printf("Event stop creation: %s\n", getErrorString(err));	

	clEnqueueMarker(__cu2cl_CommandQueue, &start);

	cl_command_queue s1, s2, s3, s4, s5, s6; // For parallelizing memory copies and kernel launches
	s1 = clCreateCommandQueue(__cu2cl_Context, __cu2cl_Device, CL_QUEUE_PROFILING_ENABLE, &err);
	printf("clCreateCommandQueue s1: %s\n", getErrorString(err));

	s2 = clCreateCommandQueue(__cu2cl_Context, __cu2cl_Device, CL_QUEUE_PROFILING_ENABLE, &err);
	printf("clCreateCommandQueue s2: %s\n", getErrorString(err));

	s3 = clCreateCommandQueue(__cu2cl_Context, __cu2cl_Device, CL_QUEUE_PROFILING_ENABLE, &err);
	printf("clCreateCommandQueue s3: %s\n", getErrorString(err));

	s4 = clCreateCommandQueue(__cu2cl_Context, __cu2cl_Device, CL_QUEUE_PROFILING_ENABLE, &err); 
	printf("clCreateCommandQueue s4: %s\n", getErrorString(err));

	s5 = clCreateCommandQueue(__cu2cl_Context, __cu2cl_Device, CL_QUEUE_PROFILING_ENABLE, &err); 
	printf("clCreateCommandQueue s5: %s\n", getErrorString(err));

	s6 = clCreateCommandQueue(__cu2cl_Context, __cu2cl_Device, CL_QUEUE_PROFILING_ENABLE, &err);
	printf("clCreateCommandQueue s6: %s\n", getErrorString(err));

	// Allotting memory for splitting the image into its different channels in GPU
	d_red = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(unsigned char) * totalPixels, NULL, &err);
	printf("Mem allotment for input red channel: %s\n", getErrorString(err));

	d_green = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(unsigned char) * totalPixels, NULL, &err);
	printf("Mem allotment for input green channel: %s\n", getErrorString(err));

	d_blue = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(unsigned char) * totalPixels, NULL, &err);
	printf("Mem allotment for input blue channel: %s\n", getErrorString(err));

	// Alloting memory for the output images in the GPU
	d_inputImageRGBA = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(cl_uchar4) * totalPixels, NULL, &err);
	printf("Mem allotment for input image: %s\n", getErrorString(err));

	d_outputImageRGBA = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(cl_uchar4) * totalPixels, NULL, &err);
	printf("Mem allotment for output image 1: %s\n", getErrorString(err));

	d_outputImageRGBA2 = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(cl_uchar4) * totalPixels, NULL, &err);
	printf("Mem allotment for output image 2: %s\n", getErrorString(err));

	err = clEnqueueWriteBuffer(s1, d_inputImageRGBA, CL_FALSE, 0, sizeof(cl_uchar4) * totalPixels, h_inputImageRGBA, 0, NULL, NULL);
	printf("Mem copy of image: %s\n", getErrorString(err));


	// Alloting memory for each output channel on the GPU
	// For box blur
	d_redBlurred = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(unsigned char) * totalPixels, NULL, &err);
	printf("Mem allotment for output red blurred channel: %s\n", getErrorString(err));

	d_greenBlurred = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(unsigned char) * totalPixels, NULL, &err);
	printf("Mem allotment for output green blurred channel: %s\n", getErrorString(err));

	d_blueBlurred = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(unsigned char) * totalPixels, NULL, &err);
	printf("Mem allotment for output blue blurred channel: %s\n", getErrorString(err));

	// For light edge filter
	d_redlight = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(unsigned char) * totalPixels, NULL, &err);
	printf("Mem allotment for output red light edge channel: %s\n", getErrorString(err));

	d_greenlight = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(unsigned char) * totalPixels, NULL, &err);
	printf("Mem allotment for output green light edge channel: %s\n", getErrorString(err));

	d_bluelight = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(unsigned char) * totalPixels, NULL, &err);
	printf("Mem allotment for output blue light edge channel: %s\n", getErrorString(err));

	clFinish(__cu2cl_CommandQueue);

	const size_t blockSize[3] = {32, 32, 1};
	const size_t gridSize[3] = {(cols / blockSize[0]) + 1, (rows / blockSize[1]) + 1, 1};

/*CU2CL Note -- Fast-tracked dim3 type without cast*/
	err = clSetKernelArg(__cu2cl_Kernel_separateChannels, 0, sizeof(cl_mem), &d_inputImageRGBA);
	printf("Channel separation clSetKernelArg: %s\n", getErrorString(err));

	err = clSetKernelArg(__cu2cl_Kernel_separateChannels, 1, sizeof(int), &rows);
	printf("Channel separation clSetKernelArg: %s\n", getErrorString(err));

	err = clSetKernelArg(__cu2cl_Kernel_separateChannels, 2, sizeof(int), &cols);
	printf("Channel separation clSetKernelArg: %s\n", getErrorString(err));

	err = clSetKernelArg(__cu2cl_Kernel_separateChannels, 3, sizeof(cl_mem), &d_red);
	printf("Channel separation clSetKernelArg: %s\n", getErrorString(err)); 
	
	err = clSetKernelArg(__cu2cl_Kernel_separateChannels, 4, sizeof(cl_mem), &d_green);
	printf("Channel separation clSetKernelArg: %s\n", getErrorString(err)); 
	
	err = clSetKernelArg(__cu2cl_Kernel_separateChannels, 5, sizeof(cl_mem), &d_blue);
	printf("Channel separation clSetKernelArg: %s\n", getErrorString(err));

localWorkSize[0] = blockSize[0];
	localWorkSize[1] = blockSize[1];
	localWorkSize[2] = blockSize[2];
	globalWorkSize[0] = gridSize[0] * localWorkSize[0];
	globalWorkSize[1] = gridSize[1] * localWorkSize[1];
	globalWorkSize[2] = gridSize[2] * localWorkSize[2];

	printf("Total threads = %d*%d*%d = %d\n", globalWorkSize[0], globalWorkSize[1], globalWorkSize[2], globalWorkSize[0]* globalWorkSize[1]* globalWorkSize[2]);
	printf("Threads per group = %d*%d*%d = %d\n", localWorkSize[0], localWorkSize[1], localWorkSize[2], localWorkSize[0]* localWorkSize[1]* localWorkSize[2]);

	printf("%s\n", getErrorString(clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_separateChannels, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL)));
	clFinish(__cu2cl_CommandQueue);

	//I have made processing of each channel to be run on different streams which gave me a significant speedup of 40% over running all on the same stream 
	/*CU2CL Note -- Fast-tracked dim3 type without cast*/
	err = clSetKernelArg(__cu2cl_Kernel_box_blur, 0, sizeof(cl_mem), &d_red);
	err = clSetKernelArg(__cu2cl_Kernel_box_blur, 1, sizeof(cl_mem), &d_redBlurred);
	err = clSetKernelArg(__cu2cl_Kernel_box_blur, 2, sizeof(int), &rows);
	err = clSetKernelArg(__cu2cl_Kernel_box_blur, 3, sizeof(int), &cols);
	err = clSetKernelArg(__cu2cl_Kernel_box_blur, 4, sizeof(int), &filterWidth);
	err = clSetKernelArg(__cu2cl_Kernel_box_blur, 5, sizeof(int), &divFactor);

	/*
	localWorkSize[0] = blockSize[0];
	localWorkSize[1] = blockSize[1];
	localWorkSize[2] = blockSize[2];
	globalWorkSize[0] = (gridSize[0])*localWorkSize[0]; 
	globalWorkSize[1] = (gridSize[1])*localWorkSize[1]; 
	globalWorkSize[2] = (gridSize[2])*localWorkSize[2]; 
	*/
	printf("%s\n", getErrorString(clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_box_blur, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL)));
	/*CU2CL Note -- Fast-tracked dim3 type without cast*/
	err = clSetKernelArg(__cu2cl_Kernel_box_blur, 0, sizeof(cl_mem), &d_green);
	err = clSetKernelArg(__cu2cl_Kernel_box_blur, 1, sizeof(cl_mem), &d_greenBlurred);
	err = clSetKernelArg(__cu2cl_Kernel_box_blur, 2, sizeof(int), &rows);
	err = clSetKernelArg(__cu2cl_Kernel_box_blur, 3, sizeof(int), &cols);
	err = clSetKernelArg(__cu2cl_Kernel_box_blur, 4, sizeof(int), &filterWidth);
	err = clSetKernelArg(__cu2cl_Kernel_box_blur, 5, sizeof(int), &divFactor);
	/*
	localWorkSize[0] = blockSize[0];
	localWorkSize[1] = blockSize[1];
	localWorkSize[2] = blockSize[2];
	globalWorkSize[0] = (gridSize[0])*localWorkSize[0];
	globalWorkSize[1] = (gridSize[1])*localWorkSize[1];
	globalWorkSize[2] = (gridSize[2])*localWorkSize[2];
	*/

	printf("%s\n", getErrorString(clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_box_blur, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL)));
	/*CU2CL Note -- Fast-tracked dim3 type without cast*/
	err = clSetKernelArg(__cu2cl_Kernel_box_blur, 0, sizeof(cl_mem), &d_blue);
	err = clSetKernelArg(__cu2cl_Kernel_box_blur, 1, sizeof(cl_mem), &d_blueBlurred);
	err = clSetKernelArg(__cu2cl_Kernel_box_blur, 2, sizeof(int), &rows);
	err = clSetKernelArg(__cu2cl_Kernel_box_blur, 3, sizeof(int), &cols);
	err = clSetKernelArg(__cu2cl_Kernel_box_blur, 4, sizeof(int), &filterWidth);
	err = clSetKernelArg(__cu2cl_Kernel_box_blur, 5, sizeof(int), &divFactor);
	/*
	localWorkSize[0] = blockSize[0];
	localWorkSize[1] = blockSize[1];
	localWorkSize[2] = blockSize[2];
	globalWorkSize[0] = (gridSize[0])*localWorkSize[0];
	globalWorkSize[1] = (gridSize[1])*localWorkSize[1];
	globalWorkSize[2] = (gridSize[2])*localWorkSize[2];
	*/

	printf("%s\n", getErrorString(clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_box_blur, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL)));

	/*CU2CL Note -- Fast-tracked dim3 type without cast*/
	err = clSetKernelArg(__cu2cl_Kernel_light_edge_detection, 0, sizeof(cl_mem), &d_red);
	err = clSetKernelArg(__cu2cl_Kernel_light_edge_detection, 1, sizeof(cl_mem), &d_redlight);
	err = clSetKernelArg(__cu2cl_Kernel_light_edge_detection, 2, sizeof(int), &rows);
	err = clSetKernelArg(__cu2cl_Kernel_light_edge_detection, 3, sizeof(int), &cols);
	/*
	localWorkSize[0] = blockSize[0];
	localWorkSize[1] = blockSize[1];
	localWorkSize[2] = blockSize[2];
	globalWorkSize[0] = (gridSize[0])*localWorkSize[0];
	globalWorkSize[1] = (gridSize[1])*localWorkSize[1];
	globalWorkSize[2] = (gridSize[2])*localWorkSize[2];
	*/

	printf("%s\n", getErrorString(clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_light_edge_detection, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL)));
	/*CU2CL Note -- Fast-tracked dim3 type without cast*/
	err = clSetKernelArg(__cu2cl_Kernel_light_edge_detection, 0, sizeof(cl_mem), &d_green);
	err = clSetKernelArg(__cu2cl_Kernel_light_edge_detection, 1, sizeof(cl_mem), &d_greenlight);
	err = clSetKernelArg(__cu2cl_Kernel_light_edge_detection, 2, sizeof(int), &rows);
	err = clSetKernelArg(__cu2cl_Kernel_light_edge_detection, 3, sizeof(int), &cols);
/*
	localWorkSize[0] = blockSize[0];
	localWorkSize[1] = blockSize[1];
	localWorkSize[2] = blockSize[2];
	globalWorkSize[0] = (gridSize[0])*localWorkSize[0];
	globalWorkSize[1] = (gridSize[1])*localWorkSize[1];
	globalWorkSize[2] = (gridSize[2])*localWorkSize[2];
	*/

	printf("%s\n", getErrorString(clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_light_edge_detection, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL)));
	/*CU2CL Note -- Fast-tracked dim3 type without cast*/
	err = clSetKernelArg(__cu2cl_Kernel_light_edge_detection, 0, sizeof(cl_mem), &d_blue);
	err = clSetKernelArg(__cu2cl_Kernel_light_edge_detection, 1, sizeof(cl_mem), &d_bluelight);
	err = clSetKernelArg(__cu2cl_Kernel_light_edge_detection, 2, sizeof(int), &rows);
	err = clSetKernelArg(__cu2cl_Kernel_light_edge_detection, 3, sizeof(int), &cols);
	/*
	localWorkSize[0] = blockSize[0];
	localWorkSize[1] = blockSize[1];
	localWorkSize[2] = blockSize[2];
	globalWorkSize[0] = (gridSize[0])*localWorkSize[0];
	globalWorkSize[1] = (gridSize[1])*localWorkSize[1];
	globalWorkSize[2] = (gridSize[2])*localWorkSize[2];
	*/

	printf("%s\n", getErrorString(clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_light_edge_detection, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL)));

		clFinish(__cu2cl_CommandQueue);

	/*CU2CL Note -- Fast-tracked dim3 type without cast*/
		err = clSetKernelArg(__cu2cl_Kernel_recombineChannels, 0, sizeof(cl_mem), &d_redBlurred);
	err = clSetKernelArg(__cu2cl_Kernel_recombineChannels, 1, sizeof(cl_mem), &d_greenBlurred);
	err = clSetKernelArg(__cu2cl_Kernel_recombineChannels, 2, sizeof(cl_mem), &d_blueBlurred);
	err = clSetKernelArg(__cu2cl_Kernel_recombineChannels, 3, sizeof(cl_mem), &d_outputImageRGBA);
	err = clSetKernelArg(__cu2cl_Kernel_recombineChannels, 4, sizeof(int), &rows);
	err = clSetKernelArg(__cu2cl_Kernel_recombineChannels, 5, sizeof(int), &cols);
	/*
	localWorkSize[0] = blockSize[0];
	localWorkSize[1] = blockSize[1];
	localWorkSize[2] = blockSize[2];
	globalWorkSize[0] = (gridSize[0])*localWorkSize[0];
	globalWorkSize[1] = (gridSize[1])*localWorkSize[1];
	globalWorkSize[2] = (gridSize[2])*localWorkSize[2];
	*/

	printf("%s\n", getErrorString(clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_recombineChannels, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL)));
	/*CU2CL Note -- Fast-tracked dim3 type without cast*/
		err = clSetKernelArg(__cu2cl_Kernel_recombineChannels, 0, sizeof(cl_mem), &d_redlight);
	err = clSetKernelArg(__cu2cl_Kernel_recombineChannels, 1, sizeof(cl_mem), &d_greenlight);
	err = clSetKernelArg(__cu2cl_Kernel_recombineChannels, 2, sizeof(cl_mem), &d_bluelight);
	err = clSetKernelArg(__cu2cl_Kernel_recombineChannels, 3, sizeof(cl_mem), &d_outputImageRGBA2);
	err = clSetKernelArg(__cu2cl_Kernel_recombineChannels, 4, sizeof(int), &rows);
	err = clSetKernelArg(__cu2cl_Kernel_recombineChannels, 5, sizeof(int), &cols);
	/*
	localWorkSize[0] = blockSize[0];
	localWorkSize[1] = blockSize[1];
	localWorkSize[2] = blockSize[2];
	globalWorkSize[0] = (gridSize[0])*localWorkSize[0];
	globalWorkSize[1] = (gridSize[1])*localWorkSize[1];
	globalWorkSize[2] = (gridSize[2])*localWorkSize[2];
	*/

	printf("%s\n", getErrorString(clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_recombineChannels, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL)));

	clFinish(__cu2cl_CommandQueue);
	
	clEnqueueReadBuffer(s1, d_outputImageRGBA, CL_FALSE, 0, sizeof(cl_uchar4) * totalPixels, outputImageRGBA.ptr<unsigned char>(0), 0, NULL, NULL);
	clEnqueueReadBuffer(s2, d_outputImageRGBA2, CL_FALSE, 0, sizeof(cl_uchar4) * totalPixels, outputImageRGBA2.ptr<unsigned char>(0), 0, NULL, NULL);

	float milliseconds = 0;
	clFinish(__cu2cl_CommandQueue);	
	clEnqueueMarker(__cu2cl_CommandQueue, &stop);

	__cu2cl_EventElapsedTime(&milliseconds, start, stop);

	printf("Total time taken for both filters for image of size %d,%d on GPU: %f ms\n", cols, rows, milliseconds);
	printf("The blur kernel used was %d,%d\n", filterWidth, filterWidth);

	cv::imwrite(output_file, outputImageRGBA);
	cv::imwrite(output_file2, outputImageRGBA2);

	printf("Process complete\n");

	clReleaseMemObject(d_inputImageRGBA);	clReleaseMemObject(d_outputImageRGBA);
	clReleaseMemObject(d_redBlurred);	clReleaseMemObject(d_red); clReleaseMemObject(d_redlight);
	clReleaseMemObject(d_greenBlurred); clReleaseMemObject(d_green); clReleaseMemObject(d_greenlight);
	clReleaseMemObject(d_blueBlurred); clReleaseMemObject(d_blue); clReleaseMemObject(d_bluelight);
	clReleaseCommandQueue(s1); clReleaseCommandQueue(s2); clReleaseCommandQueue(s3); clReleaseCommandQueue(s4); clReleaseCommandQueue(s5); clReleaseCommandQueue(s6);

	return 0;
__cu2cl_Cleanup();
}
