// meshes/gpu/backdiff.cu
// Ryan Hamerly, 5/19/21
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
//   05/19/21: Added FFT mesh.  Spun off macros to consts.cuh.


#include "consts.cuh"

#define matmult_bk_fft_local(s) \
    for (int i = 0; i < K; i++) \
        matmult_bk(&T[l][s_T*i][threadIdx.x], &dT[l][s_dT*i][threadIdx.x], \
                   u[(i%s)+2*s*(i/s)], u[(i%s)+2*s*(i/s)+s], dJdu[(i%s)+2*s*(i/s)], dJdu[(i%s)+2*s*(i/s)+s], t1, true);

__global__ void fname(int N, int L, int B, geom_pars, 
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
    p += ldp*(L-1);dp += ldp*(L-1); if (s) {s += lds*(L-1);}   // Flip arrays since we're iterating backwards.
    #if FFT
        strides += (L-1);
    #else
        lens += (L-1); shifts += (L-1);
    #endif
    ldp *= -1; lds *= -1;
    define_T_dT;                                                // Transfer matrices T, dT (dim=[L0][s*K][32]).
    define_cache;                                               // Cache of lengths, shifts
	scalar u[2*K], dJdu[2*K];                                   // State and gradient.
    load_u_du_bk;                                               // Load u and dJdu, gmem -> smem [macro: gmem.cu].

    // Propagate fields through the mesh.
	for (int x = 0; x < L; x += L_ker)
    {
        int L_blk = (L_ker < L-x) ? L_ker : L-x;                // Layers in block = min(L0, L-x)
        load_cache_rev;                                         // Occasionally reload cache of shifts / lengths.
        load_T_dT_bk;                                           // Load transfer matrices [macro: gmem.cu].

        if (threadIdx.y < b)
        {
            for (int l = 0; l < L_blk; l++)                     // Iterate through L_blk layers.
            {
                // Below, the FFT version of the mesh.  The small-stride crossings are thread-local, while large-stride crossings
                // make extensive use of warp shuffles.  However, large-stride crossings are less frequent in practice.
                #if FFT
                scalar t1, t2;
                int p = strides_cache[(x+l) % L_preload];
                int idxT = 0;
                switch (p)
                {
                    case 0:
                        matmult_bk_fft_local(1);                // s=1:  (01)(23)(45)(67)(89)(ab)(cd)(ef)
                        break;
                    #if (K > 1)
                    case 1:
                        matmult_bk_fft_local(2);                // s=2:  (02)(13)(46)(57)(8a)(9b)(ce)(df)
                        break;
                    #endif
                    #if (K > 2)
                    case 2:
                        matmult_bk_fft_local(4);                // s=4:  (04)(15)(26)(37)(8c)(9d)(ae)(bf)
                        break;
                    #endif
                    #if (K > 4)
                    case 3:
                        matmult_bk_fft_local(8);                // s=8:  (08)(19)(2a)(3b)(4c)(5d)(6e)(7f)
                        break;
                    #endif
                    #if (K > 8)
                    case 4:
                        matmult_bk_fft_local(16);               // s=16
                        break;
                    #endif
                    default:                                    // Non-thread-local crossings.  Warp shuffle.
                        p -= P;
                        idxT = (threadIdx.x>>p)%2 + 2*(threadIdx.x&((1<<p)-1)) + (threadIdx.x>>(p+1)<<(p+1));
                        for (int i = 0; i < K; i++)
                        {
                            t1 = __shfl_xor_sync(0xffffffffu, u[i],   1<<p, 32);        // Swap
                            t2 = __shfl_xor_sync(0xffffffffu, u[i+K], 1<<p, 32);
                            if (threadIdx.x & (1<<p)) {u[i] = t2;} else {u[i+K] = t1;}
                            t1 = __shfl_xor_sync(0xffffffffu, dJdu[i],   1<<p, 32);
                            t2 = __shfl_xor_sync(0xffffffffu, dJdu[i+K], 1<<p, 32);
                            if (threadIdx.x & (1<<p)) {dJdu[i] = t2;} else {dJdu[i+K] = t1;}

                            matmult_bk(&T[l][s_T*i][idxT], &dT[l][s_dT*i][idxT],        // Crossing
                                       u[i], u[i+K], dJdu[i], dJdu[i+K], t1, true);        

                            t1 = __shfl_xor_sync(0xffffffffu, u[i],   1<<p, 32);        // Swap
                            t2 = __shfl_xor_sync(0xffffffffu, u[i+K], 1<<p, 32);
                            if (threadIdx.x & (1<<p)) {u[i] = t2;} else {u[i+K] = t1;}
                            t1 = __shfl_xor_sync(0xffffffffu, dJdu[i],   1<<p, 32);
                            t2 = __shfl_xor_sync(0xffffffffu, dJdu[i+K], 1<<p, 32);
                            if (threadIdx.x & (1<<p)) {dJdu[i] = t2;} else {dJdu[i+K] = t1;}
                        }
                        break;
                }
                // Below, the regular version of the mesh.  This supports Reck and Clements, as well as any nearest-neighbor
                // architecture.  Most crossings are thread-local, though warp shuffles is needed in the misaligned case.
                #else
                scalar temp, u_2k, dJdu_2k;
                if (shifts_cache[(x+l) % L_preload] % 2)        // Misaligned MZIs: need warp shuffle.
                {
                    for (int i = 0; i < K-1; i++)               // Couple (u[1], u[2]), ..., (u[2K-3], u[2K-2])
                        matmult_bk(&T[l][s_T*i][threadIdx.x], &dT[l][s_dT*i][threadIdx.x], 
                                   u[2*i+1], u[2*i+2], dJdu[2*i+1], dJdu[2*i+2], temp, true);
                    // Couple (u[2K-1], u[0]) with warp shuffle.  
                    u_2k = __shfl_down_sync(0xffffffffu, u[0], 1, 32); dJdu_2k = __shfl_down_sync(0xffffffffu, dJdu[0], 1, 32);
                    matmult_bk(&T[l][s_T*(K-1)][threadIdx.x], &dT[l][s_dT*(K-1)][threadIdx.x], 
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
                        matmult_bk(&T[l][s_T*i][threadIdx.x], &dT[l][s_dT*i][threadIdx.x], 
                                   u[2*i], u[2*i+1], dJdu[2*i], dJdu[2*i+1], temp, true);
                }
                #endif
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
    save_u_du_bk;
}

#include "consts.cuh"
