// meshes/gpu/orth.cu
// Ryan Hamerly, 4/10/21
//
// Defines mesh propagation functions for the symmetric crossing.
//
// T = [[sin(θ/2),  -cos(θ/2)],
//      [cos(θ/2),   sin(θ/2)]]
//
// History
//   04/10/21: Created this file.

#define CROSSING_TYPE  ORTH

// Inference / forward propagation of fields, no derivative terms.  
// (K=1-3 optimized for 24k smem, others for 48k smem)
//*
#define K 1
#define L0 68
#define nL 12
#define fname fwdprop_N64_orth
#include "fwdprop.cu"

#define K 2
#define L0 40
#define nL 12
#define fname fwdprop_N128_orth
#include "fwdprop.cu"

#define K 3
#define L0 28
#define nL 12
#define fname fwdprop_N192_orth
#include "fwdprop.cu"

#define K 4
#define L0 44
#define nL 10
#define fname fwdprop_N256_orth
#include "fwdprop.cu"

#define K 5
#define L0 35
#define nL 12
#define fname fwdprop_N320_orth
#include "fwdprop.cu"

#define K 6
#define L0 29
#define nL 16
#define fname fwdprop_N384_orth
#include "fwdprop.cu"

#define K 8
#define L0 21
#define nL 32
#define fname fwdprop_N512_orth
#include "fwdprop.cu"

#define K 10
#define L0 17
#define nL 32
#define fname fwdprop_N640_orth
#include "fwdprop.cu"
//*/

// Forward propagation of fields and gradients.
// (K=1-2 optimized for 24k smem, others for 48k smem)
//*
#define K 1
#define L0 48
#define nL 16
#define fname fwddiff_N64_orth
#include "fwddiff.cu"

#define K 2
#define L0 24
#define nL 32
#define fname fwddiff_N128_orth
#include "fwddiff.cu"

#define K 3
#define L0 34
#define nL 32
#define fname fwddiff_N192_orth
#include "fwddiff.cu"

#define K 4
#define L0 26
#define nL 32
#define fname fwddiff_N256_orth
#include "fwddiff.cu"

#define K 5
#define L0 22
#define nL 32
#define fname fwddiff_N320_orth
#include "fwddiff.cu"

#define K 6
#define L0 18
#define nL 32
#define fname fwddiff_N384_orth
#include "fwddiff.cu"

#define K 8
#define L0 14
#define nL 32
#define fname fwddiff_N512_orth
#include "fwddiff.cu"

#define K 10
#define L0 12
#define nL 32
#define fname fwddiff_N640_orth
#include "fwddiff.cu"
//*/

// Back-propagation of fields and gradients.
// (K=1-2 optimized for 24k smem, others for 48k smem)
//*
#define K 1
#define L0 48
#define nL 16
#define fname backdiff_N64_orth
#include "backdiff.cu"
   
#define K 2
#define L0 24
#define nL 32
#define fname backdiff_N128_orth
#include "backdiff.cu"
       
#define K 3
#define L0 34
#define nL 32
#define fname backdiff_N192_orth
#include "backdiff.cu"

#define K 4
#define L0 26
#define nL 32
#define fname backdiff_N256_orth
#include "backdiff.cu"

#define K 5
#define L0 22
#define nL 32
#define fname backdiff_N320_orth
#include "backdiff.cu"

#define K 6
#define L0 18
#define nL 32
#define fname backdiff_N384_orth
#include "backdiff.cu"

#define K 8
#define L0 14
#define nL 32
#define fname backdiff_N512_orth
#include "backdiff.cu"

#define K 10
#define L0 12
#define nL 32
#define fname backdiff_N640_orth
#include "backdiff.cu"
//*/

#undef CROSSING_TYPE