// meshes/gpu/sym.cu
// Ryan Hamerly, 4/10/21
//
// Defines mesh propagation functions for the symmetric crossing.
//
// T = [[exp(+iφ) (sin(θ/2) + i cos(θ/2)sin(2α)),  i cos(θ/2)cos(2α)                      ],
//      [i cos(θ/2)cos(2α),                        exp(-iφ) (sin(θ/2) - i cos(θ/2)sin(2α))]]
//
// History
//   04/10/21: Moved this to its own file (previously part of meshprop.cu).

#define CROSSING_TYPE  SYM
#define FFT            0

// Inference / forward propagation of fields, no derivative terms.  
// (K=1 and K=2 optimized for 24k smem, others for 48k smem)
//*
#if FWDPROP
#if ALL_SIZES

#define K 1
#define L0 50
#define nL 12
#define fname fwdprop_N64_sym
#include "fwdprop.cu"

#define K 2
#define L0 28
#define nL 12
#define fname fwdprop_N128_sym
#include "fwdprop.cu"

#define K 3
#define L0 38
#define nL 12
#define fname fwdprop_N192_sym
#include "fwdprop.cu"

#endif

#define K 4
#define L0 30
#define nL 12
#define fname fwdprop_N256_sym
#include "fwdprop.cu"

#if ALL_SIZES

#define K 5
#define L0 24
#define nL 12
#define fname fwdprop_N320_sym
#include "fwdprop.cu"

#define K 6
#define L0 20
#define nL 16
#define fname fwdprop_N384_sym
#include "fwdprop.cu"

#define K 8
#define L0 14
#define nL 32
#define fname fwdprop_N512_sym
#include "fwdprop.cu"

#define K 10
#define L0 12
#define nL 32
#define fname fwdprop_N640_sym
#include "fwdprop.cu"

#endif
#endif
//*/
    
// Forward propagation of fields and gradients.
// (K=1 optimized for 24k smem, others for 48k smem)
//*
#if FWDDIFF
#if ALL_SIZES

#define K 1
#define L0 26
#define nL 16
#define fname fwddiff_N64_sym
#include "fwddiff.cu"

#define K 2
#define L0 27
#define nL 32
#define fname fwddiff_N128_sym
#include "fwddiff.cu"

#define K 3
#define L0 19
#define nL 32
#define fname fwddiff_N192_sym
#include "fwddiff.cu"

#endif

#define K 4
#define L0 14
#define nL 32
#define fname fwddiff_N256_sym
#include "fwddiff.cu"

#if ALL_SIZES

#define K 5
#define L0 12
#define nL 32
#define fname fwddiff_N320_sym
#include "fwddiff.cu"

#define K 6
#define L0 10
#define nL 32
#define fname fwddiff_N384_sym
#include "fwddiff.cu"

#define K 8
#define L0 7
#define nL 32
#define fname fwddiff_N512_sym
#include "fwddiff.cu"

#define K 10
#define L0 6
#define nL 32
#define fname fwddiff_N640_sym
#include "fwddiff.cu"

#endif
#endif
//*/

// Back-propagation of fields and gradients.
// (K=1 optimized for 24k smem, others for 48k smem)
//*
#if BACKDIFF
#if ALL_SIZES

#define K 1
#define L0 28
#define nL 12
#define fname backdiff_N64_sym
#include "backdiff.cu"
   
#define K 2
#define L0 30
#define nL 12
#define fname backdiff_N128_sym
#include "backdiff.cu"
    
#define K 3
#define L0 20
#define nL 16
#define fname backdiff_N192_sym
#include "backdiff.cu"

#endif

#define K 4
#define L0 14
#define nL 32
#define fname backdiff_N256_sym
#include "backdiff.cu"

#if ALL_SIZES

#define K 5
#define L0 12
#define nL 32
#define fname backdiff_N320_sym
#include "backdiff.cu"

#define K 6
#define L0 10
#define nL 32
#define fname backdiff_N384_sym
#include "backdiff.cu"

#define K 8
#define L0 7
#define nL 48
#define fname backdiff_N512_sym
#include "backdiff.cu"

#define K 10
#define L0 6
#define nL 48
#define fname backdiff_N640_sym
#include "backdiff.cu"

#endif
#endif
//*/

#undef CROSSING_TYPE
#undef FFT
