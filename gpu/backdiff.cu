// meshes/gpu/backdiff.cu
// Ryan Hamerly, 4/6/21
//
// Implements the back-propagation function with differentiation backdiff_N[64*K](), where [64*K] is the mesh size.  
// Requires the following preprocessor directives:
//   K  [int] = size/32.  Each thread manages 2*K waveguides.
//   L0 [int] = number of layers natively supported.  Limited by smem.  If L > L0, the propagation is broken into steps.
//   nL [int] = a total of nL*L0 shifts/lens are pre-loaded.  Must be even.  Tradeoff between smem space and gmem latency.
//   fname    = name of function (should be backdiff_N[64*K])
//
// History:
//   04/06/21: Created this file.  First working code with back-propagation.
//   05/17/21: Shortened and simplified, merging the 3 crossing types.


#define L_ker (L0)  // Actual number of layers stored in the kernel = L0*pack_T (default L0, sym: 2*L0).
#define L_preload (L0*nL)  // Number of shifts / lens pre-loaded.

#if   CROSSING_TYPE == MZI
    #define stride_T    4
    #define stride_dT   4
    #define define_T    __shared__ complex64 T[L0][4*K][32], dT[L0][4*K][32]
    #define load_u_du   load_u_du_mzi(u, dJdu, u_out, dJdu_out)
    #define load_T_dT   load_T_dT_bk_mzi
    #define save_u_du   save_u_du_mzi(u, dJdu, u_in,  dJdu_in)
    #define save_dp     save_dp_mzi
    #define matmult_bk  matmult_bk_mzi
    #define scalar      complex64
#elif CROSSING_TYPE == SYM
    #define stride_T    3
    #define stride_dT   3
    #define define_T    __shared__ float T[L0][3*K][32], dT[L0][3*K][32]
    #define load_u_du   load_u_du_sym(u, dJdu, u_out, dJdu_out)
    #define load_T_dT   load_T_dT_bk_sym
    #define save_u_du   save_u_du_sym(u, dJdu, u_in,  dJdu_in)
    #define save_dp     save_dp_sym
    #define matmult_bk  matmult_bk_sym
    #define scalar      complex64
#else
    #define stride_T    2
    #define stride_dT   1
    #define dth         dT
    #define stride_dth  stride_dT
    #define define_T    __shared__ float T[L0][2*K][32], dth[L0][K][32]
    #define load_u_du   load_u_du_orth(u, dJdu, u_out, dJdu_out)
    #define load_T_dT   load_T_dT_bk_orth
    #define save_u_du   save_u_du_orth(u, dJdu, u_in,  dJdu_in)
    #define save_dp     save_dp_orth
    #define matmult_bk  matmult_bk_orth
    #define scalar      float
#endif


__global__ void fname(int N, int L, int B, int *lens, int *shifts, 
                      float *p, float *dp, int ldp, float *s, int lds, 
                      scalar *u_out, scalar *dJdu_out,
                      scalar *u_in,  scalar *dJdu_in,  int ldu, int mode)
{
    // Definitions and Initializations.
	u_out    += ldu * (blockDim.y*blockIdx.x + threadIdx.y);    // Pointer shift, one warp per instance.
	u_in     += ldu * (blockDim.y*blockIdx.x + threadIdx.y);
    dJdu_out += ldu * (blockDim.y*blockIdx.x + threadIdx.y);
    if (dJdu_in) {dJdu_in += ldu * (blockDim.y*blockIdx.x + threadIdx.y);}
    int b = (blockDim.y*(1 + blockIdx.x) < B) ? (blockDim.y) : (B - blockDim.y*blockIdx.x);     // # active warps
    p  += ldp*(L-1);  lens   += (L-1);                          // Flip arrays since we're iterating backwards.
    dp += ldp*(L-1);  shifts += (L-1);  
    if (s) {s += lds*(L-1);}
    ldp *= -1; lds *= -1;
    define_T;                                                   // Transfer matrices T, dT (dim=[L0][s*K][32]).
    __shared__ int shifts_cache[L0*nL], lens_cache[L0*nL];      // Cache of lengths, shifts
	scalar u[2*K], dJdu[2*K];                                   // State and gradient.
    load_u_du;                                                  // Load u and dJdu, gmem -> smem [macro: gmem.cu].

    // Propagate fields through the mesh.
	for (int x = 0; x < L; x += L_ker)
    {
        int L_blk = (L_ker < L-x) ? L_ker : L-x;                // Layers in block = min(L0, L-x)
        load_pos_cache_rev;                                     // Occasionally reload cache of shifts / lengths.
        load_T_dT;                                              // Load transfer matrices [macro: gmem.cu].

        // Iterate through L_blk layers.
        if (threadIdx.y < b)
        {
            for (int l = 0; l < L_blk; l++)                     // Iterate through L_blk layers.
            {
                scalar temp, u_2k, dJdu_2k;
                if (shifts_cache[(x+l) % L_preload] % 2)        // Misaligned MZIs: need warp shuffle.
                {
                    for (int i = 0; i < K-1; i++)               // Couple (u[1], u[2]), ..., (u[2K-3], u[2K-2])
                        matmult_bk(&T[l][stride_T*i][threadIdx.x], &dT[l][stride_dT*i][threadIdx.x], 
                                   u[2*i+1], u[2*i+2], dJdu[2*i+1], dJdu[2*i+2], temp, true);
                    // Couple (u[2K-1], u[0]) with warp shuffle.  
                    u_2k = __shfl_down_sync(0xffffffffu, u[0], 1, 32); dJdu_2k = __shfl_down_sync(0xffffffffu, dJdu[0], 1, 32);
                    matmult_bk(&T[l][stride_T*K-stride_T][threadIdx.x], &dT[l][stride_dT*K-stride_dT][threadIdx.x], 
                               u[2*K-1], u_2k, dJdu[2*K-1], dJdu_2k, temp, threadIdx.x != 31);
                    u_2k = __shfl_up_sync(0xffffffffu, u_2k, 1, 32); dJdu_2k = __shfl_up_sync(0xffffffffu, dJdu_2k, 1, 32);
                    if (threadIdx.x)
                    {
                        u[0]  = u_2k;
                        dJdu[0] = dJdu_2k;
                    }
                }
                else                                            // Aligned MZIs.  Easy case!
                {
                    for (int i = 0; i < K; i++)                 // Couple (u[0], u[1]), ... (u[2K-2], u[2K-1]).
                        matmult_bk(&T[l][stride_T*i][threadIdx.x], &dT[l][stride_dT*i][threadIdx.x], 
                                       u[2*i], u[2*i+1], dJdu[2*i], dJdu[2*i+1], temp, true);
                }
            }
        }
        
        __syncthreads();
        save_dp;
        p  += L_ker * ldp;
        dp += L_ker * ldp;
        if (s) {s += L_ker * lds;}
        __syncthreads();
    }

	// Write data to output.  Same permutation as for input, but reversed.  Macro from meshprop.cu.
    save_u_du;
}


#undef L_ker
#undef L_preload
#undef K
#undef L0
#undef nL
#undef fname
#undef stride_T
#undef stride_dT
#undef define_T
#undef load_u_du
#undef load_T_dT
#undef save_u_du
#undef save_dp
#undef matmult_bk
#undef scalar
#if CROSSING_TYPE == ORTH
    #undef dth
    #undef stride_dth
#endif