// meshes/gpu/mzi_ft.cu
// Ryan Hamerly, 5/19/21
//
// FFT mesh, standard MZI crossing.
//
// History
//   05/19/21: Created this file from mzi.cu.

#define CROSSING_TYPE  MZI
#define FFT            1

// Inference / forward propagation of fields, no derivative terms.
// (K=1 optimized for 24k smem, others for 48k smem)
//*
#if FWDPROP
#if ALL_SIZES

#define K 1
#define L0 22
#define nL 8
#define fname fwdprop_ft_N64_mzi
#include "fwdprop.cu"

#define K 2
#define L0 23
#define nL 8
#define fname fwdprop_ft_N128_mzi
#include "fwdprop.cu"

#endif

#define K 4
#define L0 11
#define nL 12
#define fname fwdprop_ft_N256_mzi
#include "fwdprop.cu"

#if ALL_SIZES

#define K 8
#define L0 5
#define nL 32
#define fname fwdprop_ft_N512_mzi
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
#define fname fwddiff_ft_N64_mzi
#include "fwddiff.cu"

#define K 2
#define L0 11
#define nL 32
#define fname fwddiff_ft_N128_mzi
#include "fwddiff.cu"

#endif

#define K 4
#define L0 5
#define nL 32
#define fname fwddiff_ft_N256_mzi
#include "fwddiff.cu"

#if ALL_SIZES

#define K 8
#define L0 2
#define nL 32
#define fname fwddiff_ft_N512_mzi
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
#define fname backdiff_ft_N64_mzi
#include "backdiff.cu"
    
#define K 2
#define L0 11
#define nL 32
#define fname backdiff_ft_N128_mzi
#include "backdiff.cu"

#endif

#define K 4
#define L0 5
#define nL 32
#define fname backdiff_ft_N256_mzi
#include "backdiff.cu"

#if ALL_SIZES

#define K 8
#define L0 2
#define nL 48
#define fname backdiff_ft_N512_mzi
#include "backdiff.cu"

#endif
#endif
//*/
    
#undef CROSSING_TYPE
#undef FFT
