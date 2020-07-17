
# define N 16








__kernel void device_add(__global int* a,__global int* b,__global int* c)
{
	int index = get_local_id(0) + get_group_id(0)*get_local_size(0);
	printf("%d\n",index);
	c[index] = a[index] + b[index];
}


