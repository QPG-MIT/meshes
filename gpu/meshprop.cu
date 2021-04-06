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
//   04/05/21: Add forward-differentiation routine.


#include<cupy/complex.cuh>
#include "gmem.cu"

typedef complex<float> complex64;

extern "C" {
	

// Initializes T = [T11, T12, T21, T22] to given MZI settings (θ, φ) and imperfections (α, β).
__device__ void Tij_mzi(const float *p, const float *dp, const float *s, complex64 T[4], complex64 dT[4])
{
	// cos(θ/2), sin(θ/2), cos(θ/2+φ), sin(θ/2+φ)
	float C, S, C1, S1;
	__sincosf(p[0]/2,      &S , &C );
	__sincosf(p[0]/2+p[1], &S1, &C1);

	// cos(α ± β), sin(α ± β)
	float Cp, Sp, Cm, Sm;
	__sincosf(s[0]+s[1],   &Sp, &Cp);
	__sincosf(s[0]-s[1],   &Sm, &Cm);

	// Equivalent Python code:
    // (Cp, Cm, C, Sp, Sm, S) = [fn(x) for fn in [np.cos, np.sin] for x in [beta[0]+beta[1], beta[0]-beta[1], theta/2]]
    // T = np.exp(1j*theta/2) * np.array([[np.exp(1j*phi) * (1j*S*Cm - C*Sp),    1j*C*Cp - S*Sm],
    //                                    [np.exp(1j*phi) * (1j*C*Cp + S*Sm),   -1j*S*Cm - C*Sp]])
    T[0] = complex64(C1, S1) * complex64(-C*Sp,  S*Cm);
    T[1] = complex64(C , S ) * complex64(-S*Sm,  C*Cp);
    T[2] = complex64(C1, S1) * complex64( S*Sm,  C*Cp);
    T[3] = complex64(C , S ) * complex64(-C*Sp, -S*Cm);

    // Equivalent Python code:
    // dT = (np.exp(1j*np.array([[[phi+theta, theta]]])) *
    //       np.array([[[1j*(1j*S*Cm-C*Sp)+( 1j*C*Cm+S*Sp),   1j*( 1j*C*Cp-S*Sm)+(-1j*S*Cp-C*Sm)],
    //                  [1j*(1j*C*Cp+S*Sm)+(-1j*S*Cp+C*Sm),   1j*(-1j*S*Cm-C*Sp)+(-1j*C*Cm+S*Sp)]],
    //                 [[1j*(1j*S*Cm-C*Sp),                   0*S                               ],
    //                  [1j*(1j*C*Cp+S*Sm),                   0*S                               ]]]))
    if (dp)
    {
        dT[0] = complex64(C1, S1) * (0.5f * dp[0] * (Cm-Sp) * complex64(-S,  C) + dp[1] * complex64(-Cm*S, -C*Sp));
        dT[1] = complex64(C , S ) * (0.5f * dp[0] * (Cp+Sm) * complex64(-C, -S));
        dT[2] = complex64(C1, S1) * (0.5f * dp[0] * (Cp-Sm) * complex64(-C, -S) + dp[1] * complex64(-C*Cp,  S*Sm));
        dT[3] = complex64(C , S ) * (0.5f * dp[0] * (Cm+Sp) * complex64( S, -C));
    }
}
    
__global__ void Tij_test(float *p, float *dp, float *s, complex64 *T, complex64 *dT)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    Tij_mzi(&p[2*id], &dp[2*id], &s[2*id], &T[4*id], &dT[4*id]);
}
    
// Initializes an identity transfer matrix [[1, 0], [0, 1]].
__device__ void Tij_identity(complex64 T[4], complex64 dT[4])
{
    T[0] = 1; T[1] = 0; T[2] = 0; T[3] = 1;
    if (dT)
    {
        dT[0] = 0; dT[1] = 0; dT[2] = 0; dT[3] = 0;
    }
}

// Warp shuffle functions for complex<float>, not implemented in native CUDA.
__device__ complex64 __shfl_sync(unsigned int mask, complex64 src, int srcLane, int width)
{
    return complex64(__shfl_sync(mask, src.real(), srcLane, width),
                     __shfl_sync(mask, src.imag(), srcLane, width));
}
__device__ complex64 __shfl_up_sync(unsigned int mask, complex64 src, unsigned int delta, int width)
{
    return complex64(__shfl_up_sync(mask, src.real(), delta, width),
                     __shfl_up_sync(mask, src.imag(), delta, width));
}
__device__ complex64 __shfl_down_sync(unsigned int mask, complex64 src, unsigned int delta, int width)
{
    return complex64(__shfl_down_sync(mask, src.real(), delta, width),
                     __shfl_down_sync(mask, src.imag(), delta, width));
}
__device__ complex64 __shfl_xor_sync(unsigned int mask, complex64 src, int laneMask, int width)
{
    return complex64(__shfl_xor_sync(mask, src.real(), laneMask, width),
                     __shfl_xor_sync(mask, src.imag(), laneMask, width));
}

    
// Forward propagation of fields and errors.
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


    
    
// Forward propagation of fields, no error terms.
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
