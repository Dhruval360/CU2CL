#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#define N 760 // side of matrix containing data
#define PDIM 768 // padded dimensions
#define TPB 128 //threads per block
#define DIV 6
 
//load element from da to db to verify correct memcopy
__global__ void kernel(float * da, float * db){
 int tid = blockDim.x * blockIdx.x + threadIdx.x;
 if(tid%PDIM < N) {
 int row = blockIdx.x/DIV, col = blockIdx.x%DIV;
 db[row*N + col*blockDim.x + threadIdx.x] = da[tid];
 }
}
 
void verify(float * A, float * B, int size);
void init(float * array, int size);
 
int main(int argc, char * argv[])
{
 float * A, *dA, *B, *dB;
 A = (float *)malloc(sizeof(float)*N*N);
 B = (float *)malloc(sizeof(float)*N*N);
 
 init(A,N*N);
 size_t pitch;
 cudaMallocPitch(&dA, &pitch, sizeof(float)*N, N);
 cudaMalloc(&dB, sizeof(float)*N*N);
 
//copy memory from unpadded array A of 760 by 760 dimensions
//to more efficient dimensions of 768 by 768 on the device
 cudaMemcpy2D(dA,pitch,A,sizeof(float)*N,sizeof(float)*N,N,cudaMemcpyHostToDevice);
 int threadsperblock = TPB;
 int blockspergrid = PDIM*PDIM/threadsperblock;
 kernel<<<blockspergrid,threadsperblock>>>(dA,dB);
 cudaMemcpy(B, dB, sizeof(float)*N*N, cudaMemcpyDeviceToHost);
 //cudaMemcpy2D(B,N,dB,N,N,N,cudaMemcpyDeviceToHost);
 verify(A,B,N*N);
 
 free(A);
 free(B);
 cudaFree(dA);
 cudaFree(dB);
}
 

void init(float * array, int size){
 for (int i = 0; i < size; i++){
 array[i] = i;
 }
}

void verify(float * A, float * B, int size){
 for (int i = 0; i < size; i++) {
 assert(A[i]==B[i]);
 }
 printf("Correct!\n");
}

