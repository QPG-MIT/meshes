// meshes/gpu/sym_ft.cu
// Ryan Hamerly, 5/19/21
//
// FFT mesh, symmetric crossing.
//
// History
//   05/19/21: Created this file from sym.cu.

#define CROSSING_TYPE  SYM
#define FFT            1

// Inference / forward propagation of fields, no derivative terms.  
// (K=1 and K=2 optimized for 24k smem, others for 48k smem)
//*
#if FWDPROP
#if ALL_SIZES

#define K 1
#define L0 50
#define nL 12
#define fname fwdprop_ft_N64_sym
#include "fwdprop.cu"

#define K 2
#define L0 28
#define nL 12
#define fname fwdprop_ft_N128_sym
#include "fwdprop.cu"

#endif

#define K 4
#define L0 30
#define nL 12
#define fname fwdprop_ft_N256_sym
#include "fwdprop.cu"

#if ALL_SIZES

#define K 8
#define L0 14
#define nL 32
#define fname fwdprop_ft_N512_sym
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
#define fname fwddiff_ft_N64_sym
#include "fwddiff.cu"

#define K 2
#define L0 27
#define nL 32
#define fname fwddiff_ft_N128_sym
#include "fwddiff.cu"

#endif

#define K 4
#define L0 14
#define nL 32
#define fname fwddiff_ft_N256_sym
#include "fwddiff.cu"

#if ALL_SIZES

#define K 8
#define L0 7
#define nL 32
#define fname fwddiff_ft_N512_sym
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
#define fname backdiff_ft_N64_sym
#include "backdiff.cu"
   
#define K 2
#define L0 30
#define nL 12
#define fname backdiff_ft_N128_sym
#include "backdiff.cu"

#endif

#define K 4
#define L0 14
#define nL 32
#define fname backdiff_ft_N256_sym
#include "backdiff.cu"

#if ALL_SIZES

#define K 8
#define L0 7
#define nL 48
#define fname backdiff_ft_N512_sym
#include "backdiff.cu"

#endif
#endif
//*/

#undef CROSSING_TYPE
#undef FFT
