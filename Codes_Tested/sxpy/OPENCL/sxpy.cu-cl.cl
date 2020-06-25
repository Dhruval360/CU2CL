

__kernel
void saxpy(int n, float a, __global float *x, __global float *y)
{
  int i = get_group_id(0)*get_local_size(0) + get_local_id(0);
  if (i < n) y[i] = a*x[i] + y[i];
}


