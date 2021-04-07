// meshes/gpu/shuffle.cu
// Ryan Hamerly, 4/6/21
//
// Warp shuffle functions for types not natively supported.
//
// History:
//   04/06/21: Split off from meshprop.cu.


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
