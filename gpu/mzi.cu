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

// Inference / forward propagation of fields, no derivative terms.
//*
#define K 1
#define L0 45
#define nL 8
#define fname fwdprop_N64
#include "fwdprop.cu"

#define K 2
#define L0 23
#define nL 8
#define fname fwdprop_N128
#include "fwdprop.cu"

#define K 3
#define L0 15
#define nL 16
#define fname fwdprop_N192
#include "fwdprop.cu"

#define K 4
#define L0 11
#define nL 12
#define fname fwdprop_N256
#include "fwdprop.cu"

#define K 5
#define L0 9
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

// Forward propagation of fields and gradients.
//*
#define K 1
#define L0 22
#define nL 16
#define fname fwddiff_N64
#include "fwddiff.cu"

#define K 2
#define L0 11
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
        
// Back-propagation of fields and gradients.
//*
#define K 1
#define L0 22
#define nL 16
#define fname backdiff_N64
#include "backdiff.cu"
    
#define K 2
#define L0 11
#define nL 32
#define fname backdiff_N128
#include "backdiff.cu"
    
#define K 3
#define L0 7
#define nL 32
#define fname backdiff_N192
#include "backdiff.cu"

#define K 4
#define L0 5
#define nL 32
#define fname backdiff_N256
#include "backdiff.cu"

#define K 5
#define L0 4
#define nL 32
#define fname backdiff_N320
#include "backdiff.cu"

#define K 6
#define L0 3
#define nL 32
#define fname backdiff_N384
#include "backdiff.cu"

#define K 8
#define L0 2
#define nL 48
#define fname backdiff_N512
#include "backdiff.cu"

#define K 10
#define L0 2
#define nL 32
#define fname backdiff_N640
#include "backdiff.cu"
//*/
    
#undef CROSSING_TYPE
