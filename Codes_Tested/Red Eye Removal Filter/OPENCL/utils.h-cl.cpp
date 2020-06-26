#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include "cu2cl_util.h"




/*CU2CL Unhandled -- No main() found
CU2CL Boilerplate inserted here:
CU2CL Initialization:
__cu2cl_Init();


CU2CL Cleanup:
__cu2cl_Cleanup();
*/
extern cl_kernel __cu2cl_Kernel_naive_normalized_cross_correlation;
extern cl_kernel __cu2cl_Kernel_remove_redness_from_coordinates;
extern cl_kernel __cu2cl_Kernel_histogram_kernel;
extern cl_kernel __cu2cl_Kernel_exclusive_scan_kernel;
extern cl_kernel __cu2cl_Kernel_move_kernel;
extern cl_program __cu2cl_Program_redEyeGPU_cu;
extern cl_kernel __cu2cl_Kernel_naive_normalized_cross_correlation;
extern cl_kernel __cu2cl_Kernel_remove_redness_from_coordinates;
extern cl_program __cu2cl_Program_redEYECPU_cu;
extern const char *progSrc;
extern size_t progLen;

extern cl_kernel __cu2cl_Kernel___cu2cl_Memset;
extern cl_program __cu2cl_Util_Program;
extern cl_platform_id __cu2cl_Platform;
extern cl_device_id __cu2cl_Device;
extern cl_context __cu2cl_Context;
extern cl_command_queue __cu2cl_CommandQueue;

extern size_t globalWorkSize[3];
extern size_t localWorkSize[3];
#ifndef UTILS_H__
#define UTILS_H__

#include <iostream>
#include <iomanip>


#include <cuda_runtime_api.h>
#include <cassert>
#include <cmath>


#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

/*CU2CL Untranslated -- Unable to translate template function*/
template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
/*CU2CL Untranslated -- Template-dependent host call*/
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

template<typename T>
void checkResultsExact(const T* const ref, const T* const gpu, size_t numElem) {
  //check that the GPU result matches the CPU result
  for (size_t i = 0; i < numElem; ++i) {
    if (ref[i] != gpu[i]) {
      std::cerr << "Difference at pos " << i << std::endl;
      //the + is magic to convert char to int without messing
      //with other types
      std::cerr << "Reference: " << std::setprecision(17) << +ref[i] <<
                 "\nGPU      : " << +gpu[i] << std::endl;
      exit(1);
    }
  }
}

template<typename T>
void checkResultsEps(const T* const ref, const T* const gpu, size_t numElem, double eps1, double eps2) {
/*CU2CL Untranslated -- Template-dependent host call*/
  assert(eps1 >= 0 && eps2 >= 0);
  unsigned long long totalDiff = 0;
  unsigned numSmallDifferences = 0;
  for (size_t i = 0; i < numElem; ++i) {
    //subtract smaller from larger in case of unsigned types
      T smaller = ref[i] < gpu[i] ? ref[i] : gpu[i];//std::min(ref[i], gpu[i]);
    T larger = ref[i] > gpu[i] ? ref[i] : gpu[i];//std::max(ref[i], gpu[i]);
    T diff = larger - smaller;
    if (diff > 0 && diff <= eps1) {
      numSmallDifferences++;
    }
    else if (diff > eps1) {
      std::cerr << "Difference at pos " << +i << " exceeds tolerance of " << eps1 << std::endl;
      std::cerr << "Reference: " << std::setprecision(17) << +ref[i] <<
        "\nGPU      : " << +gpu[i] << std::endl;
      exit(1);
    }
    totalDiff += diff * diff;
  }
  double percentSmallDifferences = (double)numSmallDifferences / (double)numElem;
  if (percentSmallDifferences > eps2) {
    std::cerr << "Total percentage of non-zero pixel difference between the two images exceeds " << 100.0 * eps2 << "%" << std::endl;
    std::cerr << "Percentage of non-zero pixel differences: " << 100.0 * percentSmallDifferences << "%" << std::endl;
    exit(1);
  }
}

//Uses the autodesk method of image comparison
//Note the the tolerance here is in PIXELS not a percentage of input pixels
template<typename T>
void checkResultsAutodesk(const T* const ref, const T* const gpu, size_t numElem, double variance, size_t tolerance)
{

  size_t numBadPixels = 0;
  for (size_t i = 0; i < numElem; ++i) {
    T smaller = ref[i] < gpu[i] ? ref[i] : gpu[i];;
    T larger = ref[i] > gpu[i] ? ref[i] : gpu[i];;
    T diff = larger - smaller;
    if (diff > variance)
      ++numBadPixels;
  }

  if (numBadPixels > tolerance) {
    std::cerr << "Too many bad pixels in the image." << numBadPixels << "/" << tolerance << std::endl;
    exit(1);
  }
}

#endif
