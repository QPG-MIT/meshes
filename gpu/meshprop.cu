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
//   04/10/21: Added support for symmetric and real orthogonal crossings.


#include<cupy/complex.cuh>
#include "gmem.cu"

typedef complex<float> complex64;

#include "crossing.cu"
#include "shuffle.cu"


extern "C" {
	    
// 1: Standard crossing
// 2: Symmetric crossing 
// 3: Orthogonal crossing
#define MZI  1
#define SYM  2
#define ORTH 3


#include "mzi.cu"
#include "sym.cu"
#include "orth.cu"


// TESTING SANDBOX
// ------------------------------------------------------------------------------------------
/*
#define CROSSING_TYPE  SYM

#define K 4
#define L0 11
#define nL 12
#define fname fwdprop_N256_sym
#include "fwdprop.cu"    
  
#define K 4
#define L0 11
#define nL 12
#define fname fwddiff_N256_sym
#include "fwddiff.cu"

#define K 4
#define L0 11
#define nL 12
#define fname backdiff_N256_sym
#include "backdiff.cu"

#undef  CROSSING_TYPE
#define CROSSING_TYPE  MZI

#define K 4
#define L0 11
#define nL 12
#define fname fwdprop_N256_mzi
#include "fwdprop.cu"

#define K 4
#define L0 5
#define nL 32
#define fname fwddiff_N256_mzi
#include "fwddiff.cu"

#define K 4
#define L0 5
#define nL 32
#define fname backdiff_N256_mzi
#include "backdiff.cu"
    
#undef  CROSSING_TYPE
//*/


    
}
