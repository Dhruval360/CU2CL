

__kernel void kernel(__global float *x, int n)
{
    int tid = get_local_id(0) + get_group_id(0) * get_local_size(0);
    for (int i = tid; i < n; i += get_local_size(0) * get_num_groups(0)) {
        x[i] = sqrt(pow(3.14159, i));
    }
}


