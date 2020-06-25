#include <stdio.h>
# define N 512


void fill_array(int *data)
{
	for(int i=0;i<N;i++)
		data[i]=i;
}

void host_add(int *a,int* b,int* c)
{
	for(int i=0;i<N;i++)
		c[i] = a[i] + b[i];
}

void print_output(int* op)
{
	for(int i=0;i<N;i++)
		printf("%d\n",op[i]);
}

__global__ void device_add(int* a,int* b,int* c)
{
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	c[index] = a[index] + b[index];
}

int main(void)
{
	int *a,*b,*c;
	int size = N * sizeof(int);
	a= (int*)malloc(size);
	b= (int*)malloc(size);
	c= (int*)malloc(size);
	fill_array(a);
	fill_array(b);

	int *d_a,*d_b,*d_c;
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);


	cudaMemcpy(d_a,a,size,cudaMemcpyHostToDevice);
	cudaMemcpy(d_b,b,size,cudaMemcpyHostToDevice);


	device_add <<<2,N/2>>> (d_a,d_b,d_c);
	
	cudaMemcpy(c,d_c,size,cudaMemcpyDeviceToHost);
	// host_add(a,b,c);

	print_output(c);

	free(a);free(b);free(c);
	cudaFree(d_a);cudaFree(d_b);cudaFree(d_c);
return 0;
}
