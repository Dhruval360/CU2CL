


#define N 760 // side of matrix containing data
#define PDIM 768 // padded dimensions
#define TPB 128 //threads per block
#define DIV 6
 
//load element from da to db to verify correct memcopy
__kernel void kernel(__global float * da, __global float * db){
 int tid = get_local_size(0) * get_group_id(0) + get_local_id(0);
 if(tid%PDIM < N) {
 int row = get_group_id(0)/DIV, col = get_group_id(0)%DIV;
 db[row*N + col*get_local_size(0) + get_local_id(0)] = da[tid];
 }
}
 


 

 





