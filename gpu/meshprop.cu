// meshes/gpu/meshprop.cu
// Ryan Hamerly, 4/3/21
//
// nvcc meshprop.cu -I/usr/local/lib/python3.5/dist-packages/cupy/core/include -arch=sm_35 -Xptxas -v --cubin
//
// CUDA implementation of light propagation through a beamsplitter mesh.  The NumPy CPU versions are convenient
// for small meshes (N < 64) but do not scale well when the mesh size exceeds the CPU cache..  This code is 
// designed to achieve high local memory reuse and arithmetic intensity for meshes up to size N = 1024.
//
// History
//   12/12/18: First attempt at GPU code for mesh propagation.  Fast but not very extensible, soon abandoned.
//   04/03/21: Created this package.  Similar to the 2018 code, but easier to modify and extend.  CuPy compatible.
//   04/05/21: Add forward-mode differentiation routine.
//   04/06/21: Added reverse-mode differentiation (backpropagation) routine.


#include<cupy/complex.cuh>
#include "gmem.cu"

typedef complex<float> complex64;

#include "crossing.cu"
#include "shuffle.cu"


extern "C" {
	    
/*
__global__ void Tij_test(float *p, float *dp, float *s, complex64 *T, complex64 *dT)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    Tij_mzi(&p[2*id], &dp[2*id], &s[2*id], &T[4*id], &dT[4*id], (float *) 0, true);
}
    
*/

/*   
__global__ void fwdprop_test(float *p, float *s, complex64 *u_in, complex64 *u_out, complex64 *T)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    u_in += 2*id; u_out += 2*id; p += 2*id; s += 2*id; T += 4*id;

    complex64 u1   = u_in[0], 
              u2   = u_in[1], 
              temp = 0;

    Tij_mzi(p, (float *) 0, s, T, (complex64 *) 0, (float *) 0, true);
    matmult(T, u1, u2, temp, true);
    
    u_out[0] = u1;
    u_out[1] = u2;
}
__global__ void backprop_test(float *p, float *dp, float *s, 
                              complex64 *u_out, complex64 *dJdu_out,
                              complex64 *u_in, complex64 *dJdu_in, complex64 *T, complex64 *dT)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    u_out += 2*id; dJdu_out += 2*id; 
    u_in  += 2*id; dJdu_in  += 2*id; 
    p += 2*id; dp += 2*id; s += 2*id; T += 4*id; dT += 4*id;

    complex64 u1    = u_out[0],     dJdu1 = dJdu_out[0],
              u2    = u_out[1],     dJdu2 = dJdu_out[1],
              temp  = 0;
    
    Tij_mzi(p, (float *) 0, s, T, dT, (float *) 0, false);  // Set up matrix.
    matmult_bk(T, dT, u1, u2, dJdu1, dJdu2, temp, true);    // Back-propagate.
    u_in[0] = u1; u_in[1] = u2;                             // Return fields u.
    dJdu_in[0] = dJdu1; dJdu_in[1] = dJdu2;                 // Return gradients dJ/du.
    dp_mzi(p, s, dT, dp);                                   // Return updates dJ/dp.
}
*/


// Forward propagation of fields, no derivative terms.
//*

#define smem_offset 1
    
#define K 1
#define L0 36
#define nL 8
#define fname fwdprop_N64
#include "fwdprop.cu"

#define K 2
#define L0 20
#define nL 8
#define fname fwdprop_N128
#include "fwdprop.cu"

#define K 3
#define L0 14
#define nL 16
#define fname fwdprop_N192
#include "fwdprop.cu"

#define K 4
#define L0 11
#define nL 12
#define fname fwdprop_N256
#include "fwdprop.cu"

#define K 5
#define L0 8
#define nL 32
#define fname fwdprop_N320
#include "fwdprop.cu"

#define K 6
#define L0 7
#define nL 32
#define fname fwdprop_N384
#include "fwdprop.cu"

#define K 8
#define L0 5
#define nL 32
#define fname fwdprop_N512
#include "fwdprop.cu"

#define K 10
#define L0 4
#define nL 32
#define fname fwdprop_N640
#include "fwdprop.cu"
//*/



// Forward propagation of fields and gradients.
//*

#define K 1
#define L0 18
#define nL 16
#define fname fwddiff_N64
#include "fwddiff.cu"

#define K 2
#define L0 10
#define nL 32
#define fname fwddiff_N128
#include "fwddiff.cu"

#define K 3
#define L0 7
#define nL 32
#define fname fwddiff_N192
#include "fwddiff.cu"

#define K 4
#define L0 5
#define nL 32
#define fname fwddiff_N256
#include "fwddiff.cu"

#define K 5
#define L0 4
#define nL 32
#define fname fwddiff_N320
#include "fwddiff.cu"

#define K 6
#define L0 3
#define nL 32
#define fname fwddiff_N384
#include "fwddiff.cu"

#define K 8
#define L0 2
#define nL 32
#define fname fwddiff_N512
#include "fwddiff.cu"

#define K 10
#define L0 2
#define nL 32
#define fname fwddiff_N640
#include "fwddiff.cu"
//*/

    
    
// Back-propagation of fields and gradients.
//*
#define K 1
#define L0 15
#define nL 9
#define fname backdiff_N64
#include "backdiff.cu"
    
#define K 2
#define L0 8
#define nL 12
#define fname backdiff_N128
#include "backdiff.cu"
    
#define K 3
#define L0 5
#define nL 12
#define fname backdiff_N192
#include "backdiff.cu"

#define K 4
#define L0 4
#define nL 32
#define fname backdiff_N256
#include "backdiff.cu"

#define K 5
#define L0 3
#define nL 48
#define fname backdiff_N320
#include "backdiff.cu"

#define K 6
#define L0 3
#define nL 48
#define fname backdiff_N384
#include "backdiff.cu"

#define K 8
#define L0 2
#define nL 48
#define fname backdiff_N512
#include "backdiff.cu"

#define K 10
#define L0 1
#define nL 48
#define fname backdiff_N640
#include "backdiff.cu"
//*/
    

// These might be too big to be useful.
/*

#define K 16
#define L0 1
#define nL 32
#define fname fwddiff_N1024
#include "fwddiff.cu"

#define K 16
#define L0 2
#define nL 32
#define fname fwdprop_N1024
#include "fwdprop.cu"
*/

}
