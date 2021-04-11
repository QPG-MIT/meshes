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


#define L_ker (L0*pack_T)  // Actual number of layers stored in the kernel = L0*pack_T (default L0, sym: 2*L0).
#define L_preload (L0*nL)  // Number of shifts / lens pre-loaded.

#if CROSSING_TYPE == MZI
__global__ void fname(int N, int L, int B, 
                      int *lens, int *shifts, 
                      float *p, float *dp, int ldp, 
                      float *s, int lds, 
                      complex64 *u_out, complex64 *dJdu_out,
                      complex64 *u_in,  complex64 *dJdu_in,  int ldu)
{
    const int pack_T = 1; // Packing factor 4 / (# T params) (default: 1, symmetric Tij: 2)
    const int stride_T = 4 / pack_T;

    // There are blockDim.y warps in each block (blockDim.x = 32).  Each references a separate instance.
	// The blocks are therefore offset by blockDim.y instances, i.e. a pointer offset of ld * blockDim.y
	// Kernel doesn't support multiplexing over p, s.  This is assumed to be easier by calling separate kernels.
	u_out     += ldu * (blockDim.y*blockIdx.x + threadIdx.y);
	u_in      += ldu * (blockDim.y*blockIdx.x + threadIdx.y);
    dJdu_out  += ldu * (blockDim.y*blockIdx.x + threadIdx.y);
    if (dJdu_in) {dJdu_in += ldu  * (blockDim.y*blockIdx.x + threadIdx.y);}
    
    // Number of active warps (this block's mini-batch).
    int b = (blockDim.y*(1 + blockIdx.x) < B) ? (blockDim.y) : (B - blockDim.y*blockIdx.x);
    
    // Since we're iterating backwards through the mesh, need to move the (s, p, dp) pointers to the final layer and
    // flip the signs of (lds, ldp).
    p      += ldp * (L-1);
    dp     += ldp * (L-1);
    s      += lds * (L-1);
    lens   += (L-1);
    shifts += (L-1);
    ldp    *= -1;
    lds    *= -1;
		
	// Transfer matrices.
	// The b^th matrix of column c goes in T[c][4(b%K):4(b%K)+4][b/K].
	__shared__ complex64 T[L0][4*K][32];
	__shared__ complex64 dT[L0][4*K][32];
    __shared__ int shifts_cache[L_preload];
    __shared__ int lens_cache[L_preload];
    
	// State.  The i^th waveguide is u[i%K] of thread i/K.
	complex64 u[2*K];
	complex64 dJdu[2*K];
	
	// Load u coalesced, gmem -> smem.  Macro defined in meshprop.cu.
    load_u_du(u, dJdu, u_out, dJdu_out);

	for (int x = 0; x < L; x += L_ker)
    {
        // Number of layers in *this* block.  Normally L0, except if last block is truncated.
        int L_blk = (L_ker < L-x) ? L_ker : L-x;

        // Preload shifts and lengths to the cache (macro from gmem.cu).
        load_pos_cache_rev;

        // Load T (coalesced in gmem, strided in smem).  
        load_T_dT_bk;

        // Iterate through L_blk layers.
        if (threadIdx.y < b)
        {
            for (int l = 0; l < L_blk; l++)
            {
                complex64 temp, u_2k, dJdu_2k;
                if (shifts_cache[(x+l) % L_preload] % 2) //((x+l) % 2)
                {
                    // Couple (u[1], u[2]), (u[3], u[4]), ... (u[2K-3], u[2K-2]).
                    for (int i = 0; i < K-1; i++)
                        matmult_bk(&T[l][4*i][threadIdx.x], &dT[l][4*i][threadIdx.x], 
                                   u[2*i+1], u[2*i+2], dJdu[2*i+1], dJdu[2*i+2], temp, true);
                    // Couple (u[2K-1], u[0]).  The latter comes from the next thread up.  Warp shuffle.
                    u_2k = __shfl_down_sync(0xffffffffu, u[0], 1, 32); dJdu_2k = __shfl_down_sync(0xffffffffu, dJdu[0], 1, 32);
                    matmult_bk(&T[l][4*K-4][threadIdx.x], &dT[l][4*K-4][threadIdx.x], 
                               u[2*K-1], u_2k, dJdu[2*K-1], dJdu_2k, temp, threadIdx.x != 31);
                    u_2k = __shfl_up_sync(0xffffffffu, u_2k, 1, 32); dJdu_2k = __shfl_up_sync(0xffffffffu, dJdu_2k, 1, 32);
                    if (threadIdx.x)
                    {
                        u[0]  = u_2k;
                        dJdu[0] = dJdu_2k;
                    }
                }
                else
                {
                    // Easy case!  Couple (u[0], u[1]), (u[2], u[3]), ... (u[2K-2], u[2K-1]).
                    for (int i = 0; i < K; i++)
                        matmult_bk(&T[l][4*i][threadIdx.x], &dT[l][4*i][threadIdx.x], 
                                   u[2*i], u[2*i+1], dJdu[2*i], dJdu[2*i+1], temp, true);
                }
            }
        }
        
        __syncthreads();
        
        save_dp;
        
        p  += L_ker * ldp;
        dp += L_ker * ldp;
        s  += L_ker * lds;
        
        __syncthreads();  // TODO -- is this necessary?
    }

	// Write data to output.  Same permutation as for input, but reversed.  Macro from meshprop.cu.
    save_u_du(u, dJdu, u_in, dJdu_in);
}
#endif



#if CROSSING_TYPE == SYM
__global__ void fname(int N, int L, int B, int *lens, int *shifts, float *p, float *dp, int ldp, float *s, int lds, 
                      complex64 *u_out, complex64 *dJdu_out, complex64 *u_in,  complex64 *dJdu_in, int ldu, bool cartesian)
{
    const int pack_T = 1, stride_T = 3;
	u_out     += ldu * (blockDim.y*blockIdx.x + threadIdx.y);
	u_in      += ldu * (blockDim.y*blockIdx.x + threadIdx.y);
    dJdu_out  += ldu * (blockDim.y*blockIdx.x + threadIdx.y);
    if (dJdu_in) {dJdu_in += ldu  * (blockDim.y*blockIdx.x + threadIdx.y);}
    int b = (blockDim.y*(1 + blockIdx.x) < B) ? (blockDim.y) : (B - blockDim.y*blockIdx.x);     // # active warps
    p += ldp * (L-1); dp += ldp * (L-1); s += lds * (L-1); lens += (L-1); shifts += (L-1); ldp *= -1; lds *= -1;

    __shared__ float T[L0][3*K][32], dT[L0][3*K][32];               // Transfer matrices
    __shared__ int shifts_cache[L_preload], lens_cache[L_preload];  // Index cache
	float u[2*K], v[2*K], dJdu[2*K], dJdv[2*K];                     // State, registers.
	
    load_u_du_sym(u, v, dJdu, dJdv, u_out, dJdu_out);               // Load state, gradient.
	for (int x = 0; x < L; x += L_ker)
    {
        int L_blk = (L_ker < L-x) ? L_ker : L-x;                    // Layers in this block.
        load_pos_cache_rev;                                         // Load cache.
        load_T_dT_bk_sym;                                           // Load matrices for this block.
        if (threadIdx.y < b)                                        // Iterate through L_blk layers.
        {
            for (int l = 0; l < L_blk; l++)
            {
                float temp1, temp2, temp3, u_2k, v_2k, dJdu_2k, dJdv_2k;
                if (shifts_cache[(x+l) % L_preload] % 2)            // MZIs not aligned with threads.  Warp shuffle.
                {
                    for (int i = 0; i < K-1; i++)
                        matmult_bk_sym(&T[l][3*i][threadIdx.x], &dT[l][3*i][threadIdx.x], 
                                       u[2*i+1], v[2*i+1], u[2*i+2], v[2*i+2], 
                                       dJdu[2*i+1], dJdv[2*i+1], dJdu[2*i+2], dJdv[2*i+2], 
                                       temp1, temp2, temp3, true);
                    u_2k = __shfl_down_sync(0xffffffffu, u[0], 1, 32); dJdu_2k = __shfl_down_sync(0xffffffffu, dJdu[0], 1, 32);
                    v_2k = __shfl_down_sync(0xffffffffu, v[0], 1, 32); dJdv_2k = __shfl_down_sync(0xffffffffu, dJdv[0], 1, 32);
                    matmult_bk_sym(&T[l][3*K-3][threadIdx.x], &dT[l][3*K-3][threadIdx.x], 
                                   u[2*K-1], v[2*K-1], u_2k, v_2k,
                                   dJdu[2*K-1], dJdv[2*K-1], dJdu_2k, dJdv_2k, 
                                   temp1, temp2, temp3, threadIdx.x != 31);
                    u_2k = __shfl_up_sync(0xffffffffu, u_2k, 1, 32); dJdu_2k = __shfl_up_sync(0xffffffffu, dJdu_2k, 1, 32);
                    v_2k = __shfl_up_sync(0xffffffffu, v_2k, 1, 32); dJdv_2k = __shfl_up_sync(0xffffffffu, dJdv_2k, 1, 32);
                    if (threadIdx.x) {u[0] = u_2k; v[0] = v_2k; dJdu[0] = dJdu_2k; dJdv[0] = dJdv_2k;}
                }
                else                                                // MZIs aligned with threads.  Easy case!
                    for (int i = 0; i < K; i++)
                        matmult_bk_sym(&T[l][3*i][threadIdx.x], &dT[l][3*i][threadIdx.x], 
                                       u[2*i], v[2*i], u[2*i+1], v[2*i+1], 
                                       dJdu[2*i], dJdv[2*i], dJdu[2*i+1], dJdv[2*i+1],
                                       temp1, temp2, temp3, true);
            }
        }
        __syncthreads();
        save_dp_sym;                                                // Save parameter gradient.
        p += L_ker * ldp; dp += L_ker * ldp; 
        if (s) {s += L_ker * lds;}
        __syncthreads();
    }
    save_u_du_sym(u, v, dJdu, dJdv, u_in, dJdu_in);                 // Write output.
}
#endif



#if CROSSING_TYPE == ORTH

#define s 0
#define lds 0

__global__ void fname(int N, int L, int B, int *lens, int *shifts, float *p, float *dp, int ldp, 
                      float *u_out, float *dJdu_out, float *u_in, float *dJdu_in, int ldu)
{
    const int pack_T = 1, stride_T = 2, stride_dth = 1;

	u_out     += ldu * (blockDim.y*blockIdx.x + threadIdx.y);
	u_in      += ldu * (blockDim.y*blockIdx.x + threadIdx.y);
    dJdu_out  += ldu * (blockDim.y*blockIdx.x + threadIdx.y);
    if (dJdu_in) {dJdu_in += ldu  * (blockDim.y*blockIdx.x + threadIdx.y);}
    int b = (blockDim.y*(1 + blockIdx.x) < B) ? (blockDim.y) : (B - blockDim.y*blockIdx.x);     // Batch size.
    p += ldp * (L-1); dp += ldp * (L-1); lens += (L-1); shifts += (L-1); ldp *= -1;             // Reverse the mesh.
	__shared__ float T[L0][2*K][32], dth[L0][K][32];                    // Transfer matrix.
    __shared__ int shifts_cache[L_preload], lens_cache[L_preload];      // Index cache.
    float u[2*K], dJdu[2*K];                                            // State, registers.
	
    load_u_du_orth(u, dJdu, u_out, dJdu_out);                           // Load data.
	for (int x = 0; x < L; x += L_ker)                                  // Loop over blocks.
    {
        int L_blk = (L_ker < L-x) ? L_ker : L-x;                        // Block size.
        load_pos_cache_rev;                                             // Refresh cache.
        load_T_dT_bk_orth;                                              // Load matrices. 
        if (threadIdx.y < b)
            for (int l = 0; l < L_blk; l++)                             // Iterate through block layers.
            {
                float temp, u_2k, dJdu_2k;
                if (shifts_cache[(x+l) % L_preload] % 2)                // Misaligned MZIs.  Warp shuffle.
                {
                    for (int i = 0; i < K-1; i++)
                        matmult_bk_orth(&T[l][2*i][threadIdx.x], &dth[l][i][threadIdx.x], 
                                        u[2*i+1], u[2*i+2], dJdu[2*i+1], dJdu[2*i+2], temp, true);
                    u_2k = __shfl_down_sync(0xffffffffu, u[0], 1, 32); dJdu_2k = __shfl_down_sync(0xffffffffu, dJdu[0], 1, 32);
                    matmult_bk_orth(&T[l][2*K-2][threadIdx.x], &dth[l][K-1][threadIdx.x], 
                                    u[2*K-1], u_2k, dJdu[2*K-1], dJdu_2k, temp, threadIdx.x != 31);
                    u_2k = __shfl_up_sync(0xffffffffu, u_2k, 1, 32); dJdu_2k = __shfl_up_sync(0xffffffffu, dJdu_2k, 1, 32);
                    if (threadIdx.x) {u[0] = u_2k; dJdu[0] = dJdu_2k;}
                }
                else                                                    // Aligned MZIs.  Easy case!
                    for (int i = 0; i < K; i++)
                        matmult_bk_orth(&T[l][2*i][threadIdx.x], &dth[l][i][threadIdx.x], 
                                        u[2*i], u[2*i+1], dJdu[2*i], dJdu[2*i+1], temp, true);
            }
        __syncthreads();
        save_dp_orth;                                                   // Save parameter gradient.
        p += L_ker * ldp; dp += L_ker * ldp;
        __syncthreads();
    }
    save_u_du_orth(u, dJdu, u_in, dJdu_in);                             // Write output.
}

#undef s
#undef lds

#endif

#undef L_ker
#undef L_preload
#undef K
#undef L0
#undef nL
#undef fname