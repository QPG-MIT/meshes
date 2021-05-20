// meshes/gpu/orth_ft.cu
// Ryan Hamerly, 5/19/21
//
// FFT mesh, orthogonal crossing.
//
// History
//   05/19/21: Created this file from orth.cu.

#define CROSSING_TYPE  ORTH
#define FFT            1

// Inference / forward propagation of fields, no derivative terms.  
// (K=1-3 optimized for 24k smem, others for 48k smem)
//*
#if FWDPROP
#if ALL_SIZES

#define K 1
#define L0 68
#define nL 12
#define fname fwdprop_ft_N64_orth
#include "fwdprop.cu"

#define K 2
#define L0 40
#define nL 12
#define fname fwdprop_ft_N128_orth
#include "fwdprop.cu"

#endif

#define K 4
#define L0 44
#define nL 10
#define fname fwdprop_ft_N256_orth
#include "fwdprop.cu"

#if ALL_SIZES

#define K 8
#define L0 21
#define nL 32
#define fname fwdprop_ft_N512_orth
#include "fwdprop.cu"

#endif
#endif
//*/

// Forward propagation of fields and gradients.
// (K=1-2 optimized for 24k smem, others for 48k smem)
//*
#if FWDDIFF
#if ALL_SIZES

#define K 1
#define L0 48
#define nL 16
#define fname fwddiff_ft_N64_orth
#include "fwddiff.cu"

#define K 2
#define L0 24
#define nL 32
#define fname fwddiff_ft_N128_orth
#include "fwddiff.cu"

#endif

#define K 4
#define L0 26
#define nL 32
#define fname fwddiff_ft_N256_orth
#include "fwddiff.cu"

#if ALL_SIZES

#define K 8
#define L0 14
#define nL 32
#define fname fwddiff_ft_N512_orth
#include "fwddiff.cu"

#endif
#endif
//*/

// Back-propagation of fields and gradients.
// (K=1-2 optimized for 24k smem, others for 48k smem)
//*
#if BACKDIFF
#if ALL_SIZES

#define K 1
#define L0 48
#define nL 16
#define fname backdiff_ft_N64_orth
#include "backdiff.cu"
   
#define K 2
#define L0 24
#define nL 32
#define fname backdiff_ft_N128_orth
#include "backdiff.cu"

#endif

#define K 4
#define L0 26
#define nL 32
#define fname backdiff_ft_N256_orth
#include "backdiff.cu"

#if ALL_SIZES

#define K 8
#define L0 14
#define nL 32
#define fname backdiff_ft_N512_orth
#include "backdiff.cu"

#endif
#endif
//*/

#undef CROSSING_TYPE
#undef FFT
