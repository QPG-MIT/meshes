// meshes/gpu/fwddiff.cu
// Ryan Hamerly, 4/5/21
//
// Implements the foward-propagation function with differentiation fwddiff_N[64*K](), where [64*K] is the mesh size.  
// Requires the following preprocessor directives:
//   K  [int] = size/32.  Each thread manages 2*K waveguides.
//   L0 [int] = number of layers natively supported.  Limited by smem.  If L > L0, the propagation is broken into steps.
//   nL [int] = a total of nL*L0 shifts/lens are pre-loaded.  Tradeoff between smem space and gmem latency.
//   fname    = name of function (should be fwddiff_N[64*K])
//
// History:
//   04/05/21: First working CUDA code.


#define L_ker (L0*pack_T)  // Actual number of layers stored in the kernel = L0*pack_T (default L0, sym: 2*L0).
#define L_preload (L0*nL)  // Number of shifts / lens pre-loaded.

#if CROSSING_TYPE == MZI
__global__ void fname(int N, int L, int B, 
                      int *lens, int *shifts, 
                      float *p, float *dp, int ldp, 
                      float *s, int lds, 
                      complex64 *u_in,  complex64 *du_in,
                      complex64 *u_out, complex64 *du_out, int ldu)
{
    const int pack_T = 1; // Packing factor 4 / (# T params) (default: 1, symmetric Tij: 2)
    const int stride_T = 4 / pack_T;

    // There are blockDim.y warps in each block (blockDim.x = 32).  Each references a separate instance.
	// The blocks are therefore offset by blockDim.y instances, i.e. a pointer offset of ld * blockDim.y
	// Kernel doesn't support multiplexing over p, s.  This is assumed to be easier by calling separate kernels.
	u_in  += ldu * (blockDim.y*blockIdx.x + threadIdx.y);
	u_out += ldu * (blockDim.y*blockIdx.x + threadIdx.y);
    if (du_in)  {du_in  += ldu * (blockDim.y*blockIdx.x + threadIdx.y);}
    if (du_out) {du_out += ldu * (blockDim.y*blockIdx.x + threadIdx.y);}
    
    // Number of active warps (this block's mini-batch).
    int b = (blockDim.y*(1 + blockIdx.x) < B) ? (blockDim.y) : (B - blockDim.y*blockIdx.x);
		
	// Transfer matrices.
	// The b^th matrix of column c goes in T[c][4(b%K):4(b%K)+4][b/K].
	__shared__ complex64 T[L0][4*K][32];
	__shared__ complex64 dT[L0][4*K][32];
    __shared__ int shifts_cache[L_preload];
    __shared__ int lens_cache[L_preload];
    
	// State.  The i^th waveguide is u[i%K] of thread i/K.
	complex64 u[2*K];
	complex64 du[2*K];
	
	// Load u coalesced, gmem -> smem.  Macro defined in meshprop.cu.
    load_u_du(u, du, u_in, du_in);

	for (int x = 0; x < L; x += L_ker)
    {
        // Number of layers in *this* block.  Normally L0, except if last block is truncated.
        int L_blk = (L_ker < L-x) ? L_ker : L-x;

        // Every L0*nL layers, reload the cache of shifts and lengths.  This is done less frequently than the
        load_pos_cache_fwd;

        // Load T (coalesced in gmem, strided in smem).  
        load_T_dT;

        // Iterate through L_blk layers.
        for (int l = 0; l < L_blk; l++)
        {
            complex64 temp, u_2k, du_2k;
            if (shifts_cache[(x+l) % L_preload] % 2)
            {
                // Couple (u[1], u[2]), (u[3], u[4]), ... (u[2K-3], u[2K-2]).
                for (int i = 0; i < K-1; i++)
                    matmult_d(&T[l][4*i][threadIdx.x], &dT[l][4*i][threadIdx.x], 
                              u[2*i+1], u[2*i+2], du[2*i+1], du[2*i+2], temp, true);
                // Couple (u[2K-1], u[0]).  The latter comes from the next thread up.  Warp shuffle.
                u_2k = __shfl_down_sync(0xffffffffu, u[0], 1, 32); du_2k = __shfl_down_sync(0xffffffffu, du[0], 1, 32);
                matmult_d(&T[l][4*K-4][threadIdx.x], &dT[l][4*K-4][threadIdx.x], 
                          u[2*K-1], u_2k, du[2*K-1], du_2k, temp, threadIdx.x != 31);
                u_2k = __shfl_up_sync(0xffffffffu, u_2k, 1, 32); du_2k = __shfl_up_sync(0xffffffffu, du_2k, 1, 32);
                if (threadIdx.x)
                {
                    u[0]  = u_2k;
                    du[0] = du_2k;
                }
            }
            else
            {
                // Easy case!  Couple (u[0], u[1]), (u[2], u[3]), ... (u[2K-2], u[2K-1]).
                for (int i = 0; i < K; i++)
                    matmult_d(&T[l][4*i][threadIdx.x], &dT[l][4*i][threadIdx.x], 
                              u[2*i], u[2*i+1], du[2*i], du[2*i+1], temp, true);
            }
        }
        
        p  += L_ker * ldp;
        dp += L_ker * ldp;
        s  += L_ker * lds;
        
        __syncthreads();  // TODO -- is this necessary?
    }

	// Write data to output.  Same permutation as for input, but reversed.  Macro from meshprop.cu.
    save_u_du(u, du, u_out, du_out);
}
#endif



#if CROSSING_TYPE == SYM
__global__ void fname(int N, int L, int B, int *lens, int *shifts, float *p, float *dp, int ldp, float *s, int lds, 
                      complex64 *u_in, complex64 *du_in, complex64 *u_out, complex64 *du_out, int ldu, bool cartesian)
{
    const int pack_T = 1, stride_T = 3;
	u_in  += ldu * (blockDim.y*blockIdx.x + threadIdx.y);
	u_out += ldu * (blockDim.y*blockIdx.x + threadIdx.y);
    if (du_in)  {du_in  += ldu * (blockDim.y*blockIdx.x + threadIdx.y);}
    if (du_out) {du_out += ldu * (blockDim.y*blockIdx.x + threadIdx.y);}
    int b = (blockDim.y*(1 + blockIdx.x) < B) ? (blockDim.y) : (B - blockDim.y*blockIdx.x);     // # active warps
    
	__shared__ float T[L0][3*K][32], dT[L0][3*K][32];                   // Transfer matrices
    __shared__ int shifts_cache[L_preload], lens_cache[L_preload];      // Index cache
	float u[2*K], v[2*K], du[2*K], dv[2*K];                             // State and gradient, in registers.
	
    load_u_du_sym(u, v, du, dv, u_in, du_in);           // Load data to u, du.
	for (int x = 0; x < L; x += L_ker)
    {
        int L_blk = (L_ker < L-x) ? L_ker : L-x;        // Layers in this block.
        load_pos_cache_fwd;                             // Refresh cache.
        load_T_dT_sym;                                  // Load matrices for this block.
        for (int l = 0; l < L_blk; l++)                 // Iterate through L_blk layers.
        {
            float temp1, temp2, temp3, u_2k, v_2k, du_2k, dv_2k;
            if (shifts_cache[(x+l) % L_preload] % 2)        // MZIs not aligned with threads.  Warp shuffle.
            {
                for (int i = 0; i < K-1; i++)
                    matmult_d_sym(&T[l][3*i][threadIdx.x], &dT[l][3*i][threadIdx.x], 
                                  u[2*i+1], v[2*i+1], u[2*i+2], v[2*i+2], 
                                  du[2*i+1], dv[2*i+1], du[2*i+2], dv[2*i+2], 
                                  temp1, temp2, temp3, true);
                u_2k = __shfl_down_sync(0xffffffffu, u[0], 1, 32); 
                v_2k = __shfl_down_sync(0xffffffffu, v[0], 1, 32);
                du_2k = __shfl_down_sync(0xffffffffu, du[0], 1, 32); 
                dv_2k = __shfl_down_sync(0xffffffffu, dv[0], 1, 32);
                matmult_d_sym(&T[l][3*K-3][threadIdx.x], &dT[l][3*K-3][threadIdx.x], 
                              u[2*K-1], v[2*K-1], u_2k, v_2k, 
                              du[2*K-1], dv[2*K-1], du_2k, dv_2k, 
                              temp1, temp2, temp3, threadIdx.x != 31);
                u_2k = __shfl_up_sync(0xffffffffu, u_2k, 1, 32); 
                v_2k = __shfl_up_sync(0xffffffffu, v_2k, 1, 32);
                du_2k = __shfl_up_sync(0xffffffffu, du_2k, 1, 32); 
                dv_2k = __shfl_up_sync(0xffffffffu, dv_2k, 1, 32);
                if (threadIdx.x) {u[0] = u_2k; v[0] = v_2k; du[0] = du_2k; dv[0] = dv_2k;}
            }
            else
                for (int i = 0; i < K; i++)                 // MZIs aligned with threads.  Easy case!
                    matmult_d_sym(&T[l][3*i][threadIdx.x], &dT[l][3*i][threadIdx.x], 
                                  u[2*i], v[2*i], u[2*i+1], v[2*i+1], 
                                  du[2*i], dv[2*i], du[2*i+1], dv[2*i+1], 
                                  temp1, temp2, temp3, true);
        }
        p += L_ker * ldp; dp += L_ker * ldp;
        if (s) {s += L_ker * lds;}
        __syncthreads();
    }
    save_u_du_sym(u, v, du, dv, u_out, du_out);         // Save output.
}
#endif



#if CROSSING_TYPE == ORTH

#define s 0
#define lds 0

__global__ void fname(int N, int L, int B, int *lens, int *shifts, float *p, float *dp, int ldp, 
                      float *u_in, float *du_in, float *u_out, float *du_out, int ldu)
{
    const int pack_T = 1, stride_T = 2, stride_dth = 1;

	u_in  += ldu * (blockDim.y*blockIdx.x + threadIdx.y);
	u_out += ldu * (blockDim.y*blockIdx.x + threadIdx.y);
    if (du_in)  {du_in  += ldu * (blockDim.y*blockIdx.x + threadIdx.y);}
    if (du_out) {du_out += ldu * (blockDim.y*blockIdx.x + threadIdx.y);}
    int b = (blockDim.y*(1 + blockIdx.x) < B) ? (blockDim.y) : (B - blockDim.y*blockIdx.x);     // Batch size.
	__shared__ float T[L0][2*K][32], dth[L0][K][32];                    // Transfer matrix & d_theta
    __shared__ int shifts_cache[L_preload], lens_cache[L_preload];      // Index cache.
	float u[2*K], du[2*K];                                              // State, registers.
	
    load_u_du_orth(u, du, u_in, du_in);                         // Load data from memory.
	for (int x = 0; x < L; x += L_ker)                          // Loop over layer blocks.
    {
        int L_blk = (L_ker < L-x) ? L_ker : L-x;                // Layers in this block.
        load_pos_cache_fwd;                                     // Refresh cache.
        load_T_dT_orth;                                         // Load T & d_theta.
        for (int l = 0; l < L_blk; l++)                         // Iterate through layers.
        {
            float temp, u_2k, du_2k;
            if (shifts_cache[(x+l) % L_preload] % 2)            // Misaligned MZIs: warp shuffle.
            {
                for (int i = 0; i < K-1; i++)
                    matmult_d_orth(&T[l][2*i][threadIdx.x], &dth[l][i][threadIdx.x], 
                                   u[2*i+1], u[2*i+2], du[2*i+1], du[2*i+2], temp, true);
                u_2k = __shfl_down_sync(0xffffffffu, u[0], 1, 32); du_2k = __shfl_down_sync(0xffffffffu, du[0], 1, 32);
                matmult_d_orth(&T[l][2*K-2][threadIdx.x], &dth[l][K-1][threadIdx.x], 
                               u[2*K-1], u_2k, du[2*K-1], du_2k, temp, threadIdx.x != 31);
                u_2k = __shfl_up_sync(0xffffffffu, u_2k, 1, 32); du_2k = __shfl_up_sync(0xffffffffu, du_2k, 1, 32);
                if (threadIdx.x) {u[0] = u_2k; du[0] = du_2k;}
            }
            else                                                // Aligned MZIs.  Easy case!
                for (int i = 0; i < K; i++)
                    matmult_d_orth(&T[l][2*i][threadIdx.x], &dth[l][i][threadIdx.x], 
                                   u[2*i], u[2*i+1], du[2*i], du[2*i+1], temp, true);
        }
        p  += L_ker * ldp; dp += L_ker * ldp;
        __syncthreads();
    }
    save_u_du_orth(u, du, u_out, du_out);                       // Save output.
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