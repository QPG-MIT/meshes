// meshes/gpu/mzi.cu
// Ryan Hamerly, 4/10/21
//
// Defines mesh propagation functions for the standard MZI crossing.
// Two phase shifters, one internal and one on the input.
//
// T = S(β) P(θ) S(α) P(φ)
//
// S(ψ) = [[cos(ψ), i sin(ψ)], [i sin(ψ), cos(ψ)]]
// P(ψ) = [[exp(iψ), 0], [0, 1]]
//
// History
//   04/10/21: Moved this to its own file (previously part of meshprop.cu).

#define CROSSING_TYPE  MZI
#define FFT            0

// Inference / forward propagation of fields, no derivative terms.
// (K=1 optimized for 24k smem, others for 48k smem)
//*
#if FWDPROP
#if ALL_SIZES

#define K 1
#define L0 22
#define nL 8
#define fname fwdprop_N64_mzi
#include "fwdprop.cu"

#define K 2
#define L0 23
#define nL 8
#define fname fwdprop_N128_mzi
#include "fwdprop.cu"

#define K 3
#define L0 15
#define nL 16
#define fname fwdprop_N192_mzi
#include "fwdprop.cu"

#endif

#define K 4
#define L0 11
#define nL 12
#define fname fwdprop_N256_mzi
#include "fwdprop.cu"

#if ALL_SIZES

#define K 5
#define L0 9
#define nL 32
#define fname fwdprop_N320_mzi
#include "fwdprop.cu"

#define K 6
#define L0 7
#define nL 32
#define fname fwdprop_N384_mzi
#include "fwdprop.cu"

#define K 8
#define L0 5
#define nL 32
#define fname fwdprop_N512_mzi
#include "fwdprop.cu"

#define K 10
#define L0 4
#define nL 32
#define fname fwdprop_N640_mzi
#include "fwdprop.cu"

#endif
#endif
//*/

// Forward propagation of fields and gradients.
//*
#if FWDDIFF
#if ALL_SIZES

#define K 1
#define L0 22
#define nL 16
#define fname fwddiff_N64_mzi
#include "fwddiff.cu"

#define K 2
#define L0 11
#define nL 32
#define fname fwddiff_N128_mzi
#include "fwddiff.cu"

#define K 3
#define L0 7
#define nL 32
#define fname fwddiff_N192_mzi
#include "fwddiff.cu"

#endif

#define K 4
#define L0 5
#define nL 32
#define fname fwddiff_N256_mzi
#include "fwddiff.cu"

#if ALL_SIZES

#define K 5
#define L0 4
#define nL 32
#define fname fwddiff_N320_mzi
#include "fwddiff.cu"

#define K 6
#define L0 3
#define nL 32
#define fname fwddiff_N384_mzi
#include "fwddiff.cu"

#define K 8
#define L0 2
#define nL 32
#define fname fwddiff_N512_mzi
#include "fwddiff.cu"

#define K 10
#define L0 2
#define nL 32
#define fname fwddiff_N640_mzi
#include "fwddiff.cu"

#endif
#endif
//*/

// Back-propagation of fields and gradients.
//*
#if BACKDIFF
#if ALL_SIZES

#define K 1
#define L0 22
#define nL 16
#define fname backdiff_N64_mzi
#include "backdiff.cu"
    
#define K 2
#define L0 11
#define nL 32
#define fname backdiff_N128_mzi
#include "backdiff.cu"
    
#define K 3
#define L0 7
#define nL 32
#define fname backdiff_N192_mzi
#include "backdiff.cu"

#endif

#define K 4
#define L0 5
#define nL 32
#define fname backdiff_N256_mzi
#include "backdiff.cu"

#if ALL_SIZES

#define K 5
#define L0 4
#define nL 32
#define fname backdiff_N320_mzi
#include "backdiff.cu"

#define K 6
#define L0 3
#define nL 32
#define fname backdiff_N384_mzi
#include "backdiff.cu"

#define K 8
#define L0 2
#define nL 48
#define fname backdiff_N512_mzi
#include "backdiff.cu"

#define K 10
#define L0 2
#define nL 32
#define fname backdiff_N640_mzi
#include "backdiff.cu"

#endif
#endif
//*/
    
#undef CROSSING_TYPE
#undef FFT
