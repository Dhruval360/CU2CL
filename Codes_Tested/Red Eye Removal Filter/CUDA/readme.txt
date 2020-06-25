
The files used for this task are:
1) redEyeCPU.cu - Applying the red eye filter for CPU. Radix sort is implemented serially here with the function radixSortCPU. Opencv is used to process the images.Remove redness from coordinates, removes the redness after the sort. Normalised cross correlation is also performed with a kernel function. We load the image witha ll of its features.We split it to RGB channels.We perform the sum and means of the RGB values.We perform Normalised cross correlation on each of the channels.We add the minimum value(negetive) to all the values to make it positive after combining the pixels.We need to compare them adn we sort them to remove the redness.CPU_radix is used in this file which performs radix sort and gives the output positions.We keep the g,b values and place the alteed value of the R pxels to produce our output image.
2) redEyeGPU.cu - Here we apply red eye filter for GPU. We perform radix sort in parallel using simple operations of historgram and exclusive scan. Rest is same as that of the CPU code.


Input files: red_eye_effect_5.jpg, red_eye_effect_tempalte_5.jpg
I have harcoded in this case for convienience and for other files it can just be replaced by the string input_file adn template file respectively.

OutputFiles: fromcpu.jpg and fromgpu.jpg

Timing: For the CPU process, it took 0.55s to complete the radix sort (the bottle neck of the operation), it took the GPU 0.056s for the same. 


Software Requiredments: CUDA
Hardware Requirements: GPU
