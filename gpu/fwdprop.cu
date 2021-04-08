// meshes/gpu/fwdprop.cu
// Ryan Hamerly, 4/3/21
//
// Implements the foward-propagation function fwdprop_N[64*K](), where [64*K] is the mesh size.  Requires the following
// preprocessor directives:
//   K  [int] = size/32.  Each thread manages 2*K waveguides.
//   L0 [int] = number of layers natively supported.  Limited by smem.  If L > L0, the propagation is broken into steps.
//   nL [int] = a total of nL*L0 shifts/lens are pre-loaded.  Must be even.  Tradeoff between smem space and gmem latency.
//   fname    = name of function (should be fwdprop_N[64*K])
//
// History:
//   04/03/21: Created this file.  First working CUDA code.
//   04/05/21: Moved the global memory I/O stuff to its own macros in gmem.cu.


#define L_ker (L0*pack_T)  // Actual number of layers stored in the kernel = L0*pack_T (default L0, sym: 2*L0).
#define L_preload (L0*nL)  // Number of shifts / lens pre-loaded.

__global__ void fname(int N, int L, int B, 
                      int *lens, int *shifts, 
                      float *p, int ldp, 
                      float *s, int lds, 
                      complex64 *u_in, int ld_in,
                      complex64 *u_out, int ld_out)/*,
                      int pack_u_T=2,
                      bool sym=false)*/
{
    const int pack_u = 2; // Packing factor = T.shape[2]/2 (default 2)
    const int pack_T = 1; // Packing factor 4 / (# T params) (default: 1, symmetric Tij: 2)
    const int stride_T = 4 / pack_T;

    // There are blockDim.y warps in each block (blockDim.x = 32).  Each references a separate instance.
	// The blocks are therefore offset by blockDim.y instances, i.e. a pointer offset of ld * blockDim.y
	// Kernel doesn't support multiplexing over p, s.  This is assumed to be easier by calling separate kernels.
	u_in  += ld_in * (blockDim.y*blockIdx.x + threadIdx.y);
	u_out += ld_out * (blockDim.y*blockIdx.x + threadIdx.y);
    // Number of active warps (this block's mini-batch).
    int b = (blockDim.y*(1 + blockIdx.x) < B) ? (blockDim.y) : (B - blockDim.y*blockIdx.x);
    
		
	// Transfer matrices.
	// The b^th matrix of column c goes in T[c][4(b%K):4(b%K)+4][b/K].
	__shared__ complex64 T[L0][4*K][32];
    __shared__ int shifts_cache[nL*L0];
    __shared__ int lens_cache[nL*L0];
    
	// State.  The i^th waveguide is u[i%K] of thread i/K.
	complex64 u[2*K];
	
	// Load u coalesced, gmem -> smem.  Macro defined in meshprop.cu.
    load_u(u, u_in);

	for (int x = 0; x < L; x += L_ker)
    {
        // Number of layers in *this* block.  Normally L0, except if last block is truncated.
        int L_blk = (L_ker < L-x) ? L_ker : L-x;

        // Every L0*nL layers, reload the cache of shifts and lengths.  This is done less frequently than the
        load_pos_cache_fwd;

        // Load T (coalesced in gmem, strided in smem). 
        load_T;

        // Iterate through L_blk layers.
        for (int l = 0; l < L_blk; l++)
        {
            complex64 temp, u_2k;
            if (shifts_cache[(x+l) % L_preload] % 2) //((x+l) % 2)
            {
                // Couple (u[1], u[2]), (u[3], u[4]), ... (u[2K-3], u[2K-2]).
                for (int i = 0; i < K-1; i++)
                    matmult(&T[l][4*i][threadIdx.x], u[2*i+1], u[2*i+2], temp, true);
                // Couple (u[2K-1], u[0]).  The latter comes from the next thread up.  Warp shuffle.
                u_2k = __shfl_down_sync(0xffffffffu, u[0], 1, 32);
                matmult(&T[l][4*K-4][threadIdx.x], u[2*K-1], u_2k, temp, threadIdx.x != 31);
                u_2k = __shfl_up_sync(0xffffffffu, u_2k, 1, 32);
                if (threadIdx.x)
                    u[0] = u_2k;
            }
            else
            {
                // Easy case!  Couple (u[0], u[1]), (u[2], u[3]), ... (u[2K-2], u[2K-1]).
                for (int i = 0; i < K; i++)
                    matmult(&T[l][4*i][threadIdx.x], u[2*i], u[2*i+1], temp, true);
            }
        }
        
        p += L_ker * ldp;
        s += L_ker * lds;
        
        __syncthreads();  // TODO -- is this necessary?
    }

	// Write data to output.  Same permutation as for input, but reversed.  Macro from meshprop.cu.
    save_u(u, u_out);
}

#undef L_ker
#undef L_preload
#undef K
#undef L0
#undef nL
#undef fname