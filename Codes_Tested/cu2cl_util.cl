Did this apprear instead of license??
__kernel void __cu2cl_Memset(__global uchar *ptr, uchar value, uint num) {
    size_t id = get_global_id(0);
    if (get_global_id(0) < num) {
        ptr[id] = value;
    }
}


