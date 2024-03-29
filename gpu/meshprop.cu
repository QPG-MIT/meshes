// meshes/gpu/meshprop.cu
// Ryan Hamerly, 5/19/21
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
//   05/19/21: Added FFT mesh.  Reduced redundant code with macros.


// Compiler flags.  Use these to tailor the compilation.  Mainly for testing so that I don't have to compile all the different
// kernel types (takes about a minute) each time.
#define ALL_SIZES   1       // Include all N=64-512.  Set to false for faster compilation bug testing.

#define FWDPROP     1       // Include fwdprop.cu
#define FWDDIFF     1       // Include fwddiff.cu
#define BACKDIFF    1       // Include backdiff.cu

#define HAS_MZI     1       // Include MZI crossings.
#define HAS_SYM     1       // Include symmetric crossings.
#define HAS_ORTH    1       // Include orthogonal crossings.

#define HAS_NN      1       // Include nearest-neighbor mesh geometry.
#define HAS_FFT     1       // Include FFT mesh geometry.


// Various imports.
#include<cupy/complex.cuh>
#include "gmem.cu"
typedef complex<float> complex64;
#include "crossing.cu"
#include "shuffle.cu"


extern "C" {
	    
#define MZI         1       // Standard crossing          
#define SYM         2       // Symmetric crossing
#define ORTH        3       // Orthogonal crossing

// All of the CUDA kernels are defined in the files imported below.
#if HAS_NN
    #if HAS_MZI
        #include "mzi.cu"
    #endif
    #if HAS_SYM
        #include "sym.cu"
    #endif
    #if HAS_ORTH
        #include "orth.cu"
    #endif
#endif
#if HAS_FFT
    #if HAS_MZI
        #include "mzi_ft.cu"
    #endif
    #if HAS_SYM
        #include "sym_ft.cu"
    #endif
    #if HAS_ORTH
        #include "orth_ft.cu"
    #endif
#endif

}
