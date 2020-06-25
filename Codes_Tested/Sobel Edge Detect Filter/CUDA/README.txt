
The files used for this task are:
1) sobelEdgeFilterpng : Sobel Edge Filter is applied to the image by the host(CPU) and the device(GPU) and the timing that each takes is tracked and displayed. Two output files result namely, fromcpu.png and fromgpu.png with a hardcoded input person.png.
A structure is created for the png image with its pixels,width,height. LoadImage loads the image, writeImage writes it to a file.
SobelCpu is the operation of sobelEdgeFilter on CPU.(Serial)
SobelGpu is the operation of sobelEdgeFilter on Gpu.(Parallel)

2) lodepng.cpp: LodePNG is a PNG image decoder and encoder, all in one, no dependency or linkage to zlib or libpng required. It is opensource. I have used this since majority of images are generally in png format and other formats can be coverted to the same with ease.

Input file: person.png 
Width: 256 pixels
Height: 252 pixels
If input file is changed, the variable in, which is hardcoded with the value is to be changed with the new filename.

OutputFiles: fromcpu.png and fromgpu.png

A bar graph is plotted comparing the two times in timeComparision.jpg

To execute: 
nvcc sobelEdgeFilterpng.cu lodepng.cpp -arch sm_20
./a.out

Software Requiredments: CUDA
Hardware Requirements: GPU
