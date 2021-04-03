// meshes/gpu/fwdprop.cu
// Ryan Hamerly, 4/3/21
//
// Implements the foward-propagation function fwdprop_N[64*K](), where [64*K] is the mesh size.  Requires the following
// preprocessor directives:
//   K  [int] = size/32.  Each thread manages 2*K waveguides.
//   L0 [int] = number of layers natively supported.  Limited by smem.  If L > L0, the propagation is broken into steps.
//   nL [int] = a total of nL*L0 shifts/lens are pre-loaded.  Tradeoff between smem space and gmem latency.
//   fname    = name of function (should be fwdprop_N[64*K])
//
// History:
//   04/03/21: Created this file.  First working CUDA code.


__global__ void fname(int N, int L, int B, 
                      int *lens, int *shifts, 
                      float *p, int ldp, 
                      float *s, int lds, 
                      complex64 *u_in, int ld_in,
                      complex64 *u_out, int ld_out)
{
	// There are blockDim.y warps in each block (blockDim.x = 32).  Each references a separate instance.
	// The blocks are therefore offset by blockDim.y instances, i.e. a pointer offset of ld * blockDim.y
	// Kernel doesn't support multiplexing over p, s.  This is assumed to be easier by calling separate kernels.
	u_in  += ld_in * (blockDim.y*blockIdx.x + threadIdx.y);
	u_out += ld_out * (blockDim.y*blockIdx.x + threadIdx.y);
    // Number of active warps (this block's mini-batch).
    int b = (blockDim.y*(1 + blockIdx.x) < B) ? (blockDim.y) : (B - blockDim.y*blockIdx.x);
		
	// Transfer matrices.
	// The b^th matrix of column c goes in T[c][b/K][4(b%K):4(b%K)+4].  TODO: Offset to avoid bank conflicts?
	__shared__ complex64 T[L0][32][4*K+smem_offset];   // 0 -> 1 for offset.
    __shared__ int shifts_cache[nL*L0];
    __shared__ int lens_cache[nL*L0];
	// State.  The i^th waveguide is u[i%K] of thread i/K.
	complex64 u[2*K];
	
	// Load u coalesced, gmem -> smem.  Then reorder strided, smem -> regs.  (Direct gmem -> regs would be strided, poor bandwidth).
	// Loads to smem can be parallelized over thread warps, but if threadIdx.y > 2*L, then there's not enough smem to do them
	// all at once, hence the thread-sync'd for loop.
	// Python code that checks this (for deubugging):
	/*
	B = 4    # Batch size (variable)
	L = 2    # Length of mesh
	K = 2    # Width of mesh: 64*K (2*K variables / thread)
	ldu = 2*K*32
	u_in = np.arange(B*ldu)
	T = np.zeros([L, 32, 4*K]).astype(int)
	u = np.zeros([B, 32, 2*K]).astype(int)

	for i in range(0, B, 2*L):
	    print (f"i = {i}")
	    for tix_y in range(B):
	        if (i <= tix_y < i + 2*L and tix_y < B):
	            print (f"tix_y = {tix_y}")
	            for k in range(2*K):
	                ind = 32*k + ldu*tix_y
	                print (f"idx=[{(tix_y//2)%L}][:32][{k+2*K*(tix_y%2)}] <-- [{ind}+:32]")
	                for tix_x in range(32):
	                    T[(tix_y//2)%L, tix_x, k+2*K*(tix_y%2)] = u_in[ind+tix_x]
	            for k in range(2*K):
	                for tix_x in range(32):
	                    u[tix_y, tix_x, k] = T[(tix_y//2)%L, (2*K*tix_x + k)%32,
	                                           (2*K*tix_x + k)//32 + 2*K*(tix_y%2)]       
	(u.flatten() == u_in).all()
	*/
    //int iMax = (blockDim.y*(1 + blockIdx.x) < B) ? (blockDim.y) : (B - blockDim.y*blockIdx.x);
	for (int i = 0; i < b; i += 2*L0)
	{
        int l = threadIdx.y - i;
		if (0 <= l && l < 2*L0 && threadIdx.y < b)
			for (int k = 0; k < 2*K; k++)
            {
                T[(l/2)%L0][threadIdx.x][k+2*K*(l%2)] = (32*k + threadIdx.x < N) ? u_in[32*k + threadIdx.x] : 0;
            }
        __syncthreads();
		if (0 <= l && l < 2*L0)
			for (int k = 0; k < 2*K; k++)
				u[k] = T[(l/2)%L0][(2*K*threadIdx.x + k)%32][(2*K*threadIdx.x + k)/32+2*K*(l%2)];
		__syncthreads();
	}

    // TODO -- fix.  Testing the index manipulation first.
    //for (int k = 0; k < 2*K; k++)
    //    out[2*K*threadIdx.x + k] = u[k];  
    //return;
    
	for (int x = 0; x < L; x += L0)
    {
        // Number of layers in *this* block.  Normally L0, except if last block is truncated.
        int L_blk = (L0 < L-x) ? L0 : L-x;
        
        // Every L0*nL layers, reload the cache of shifts and lengths.  This is done less frequently than the
        // (p, s) updates because there's less data to load, so the cache can store more layers.  More importantly,
        // the (p, s) updates rely on the shifts and lengths (i.e. to only load certain regions); doing these
        // updates less frequently reduces memory latency.
        /*
        for (L0, nL, bd_y) in zip([36,20,14,11,7,5,4,2], [8,8,16,16,32,32,32,32], [8,10,15,18,26,20,16,12]):
            L = 1024; shifts = np.arange(L); shifts_cache = np.repeat(-1, [nL*L0]); shifts_cache_list = []
            for x in range(0, L, L0):
                if (x % (L0*nL) == 0):
                    for i in range(0, L0*nL, 32*bd_y):
                        for tix_y in range(bd_y):
                            for tix_x in range(32):
                                idx = i + 32*tix_y + tix_x
                                if (idx < L0*nL and x + idx < L):
                                    shifts_cache[idx] = shifts[x + idx]
                    shifts_cache_list.append(np.array(shifts_cache))
                    shifts_cache[:] = -1
            print ((np.concatenate(shifts_cache_list)[:len(shifts)] == shifts).all())        
        */
        if (x % (L0*nL) == 0)
        {
            for (int i = 0; i < L0*nL; i += 32*blockDim.y)
            {
                int id = i + 32*threadIdx.y + threadIdx.x;
                if (id < L0*nL && x + id < L)
                {
                    lens_cache[id]   = lens[x + id];
                    shifts_cache[id] = shifts[x + id];
                }
            }
            __syncthreads();
        }
        
        // Load T (coalesced in gmem, strided in smem).
        // Python code that checks this (for debugging):
        /*
        B = 3    # Batch size (variable)
        L = 10   # Length of mesh
        K = 2    # Width of mesh: 64*K (2*K variables / thread)
        ldp = lds = 2*K*32
        p = np.array([1*np.arange(32*K*L), 2*np.arange(32*K*L)]).T.flatten()
        s = np.array([3*np.arange(32*K*L), 4*np.arange(32*K*L)]).T.flatten()
        T = np.zeros([L, 32, 4*K], dtype=int)

        def T_test(p1, p2, s1, s2):
            return np.array([p1, p2, s1, s2])

        for i in range(0, K*L, B):
            print (f"i={i}")
            for tix_y in range(B):
                print (f"   -> tix_y={tix_y}")
                l = (i + tix_y)//K; m = (i + tix_y)%K
                if (l < L):
                    print (f"               -> m={m}, l={l}, idx={ldp*l + 2*32*m}")
                    for tix_x in range(32):
                        dm = (m*32 + tix_x)
                        idx_p = ldp*l + 2*dm; idx_s = lds*l + 2*dm
                        T[l, dm//K, 4*(dm%K):4*(dm%K+1)] = T_test(p[idx_p], p[idx_p+1],
                                                                  s[idx_s], s[idx_s+1])
        (T.reshape(L*32*K, 4) == np.array([p[::2], p[1::2], s[::2], s[1::2]]).T).all()
        */
        for (int i = 0; i < K*L0; i += blockDim.y)
        {
            int l = (i + threadIdx.y)/K, m = (i + threadIdx.y)%K;
            if (l < L_blk)
            {
                int dm = (m*32 + threadIdx.x);
                int idx_p = ldp*l + 2*dm, idx_s = lds*l + 2*dm;
                Tij_identity(&T[l][dm/K][4*(dm%K)]);
                if (dm >= shifts_cache[(x+l) % (L0*nL)]/2 &&
                    dm <  shifts_cache[(x+l) % (L0*nL)]/2 + lens_cache[(x+l) % (L0*nL)])
                {
                    Tij_mzi(&p[idx_p], &s[idx_s], &T[l][dm/K][4*(dm%K)]);
                }
            }
        }
        __syncthreads();

        // Iterate through L_blk layers.
        for (int l = 0; l < L_blk; l++)
        {
            complex64 temp, u_2k;
            if (shifts_cache[(x+l) % (L0*nL)] % 2) //((x+l) % 2)
            {
                // Couple (u[1], u[2]), (u[3], u[4]), ... (u[2K-3], u[2K-2]).
                for (int i = 0; i < K-1; i++)
                {
                    temp     = T[l][threadIdx.x][4*i  ] * u[2*i+1] + T[l][threadIdx.x][4*i+1] * u[2*i+2];
                    u[2*i+2] = T[l][threadIdx.x][4*i+2] * u[2*i+1] + T[l][threadIdx.x][4*i+3] * u[2*i+2];
                    u[2*i+1] = temp;
                }
                // Couple (u[2K-1], u[0]).  The latter comes from the next thread up.  Warp shuffle.
                u_2k     = __shfl_down_sync(0xffffffffu, u[0], 1, 32);
                temp     = T[l][threadIdx.x][4*K-4] * u[2*K-1] + T[l][threadIdx.x][4*K-3] * u_2k;
                u_2k     = T[l][threadIdx.x][4*K-2] * u[2*K-1] + T[l][threadIdx.x][4*K-1] * u_2k;
                if (threadIdx.x != 31)
                    u[2*K-1] = temp;
                u_2k = __shfl_up_sync(0xffffffffu, u_2k, 1, 32);
                if (threadIdx.x)
                    u[0] = u_2k;
            }
            else
            {
                // Easy case!  Couple (u[0], u[1]), (u[2], u[3]), ... (u[2K-2], u[2K-1]).
                for (int i = 0; i < K; i++)
                {
                    temp     = T[l][threadIdx.x][4*i  ] * u[2*i] + T[l][threadIdx.x][4*i+1] * u[2*i+1];
                    u[2*i+1] = T[l][threadIdx.x][4*i+2] * u[2*i] + T[l][threadIdx.x][4*i+3] * u[2*i+1];
                    u[2*i] = temp;
                }
            }
        }
        
        p += L0 * ldp;
        s += L0 * lds;
        
        __syncthreads();  // TODO -- is this necessary?
    }

	// Write data to output.  Same permutation as for input, but reversed.
	for (int i = 0; i < b; i += 2*L0)
	{
        int l = threadIdx.y - i;
		if (0 <= l && l < 2*L0)
            for (int k = 0; k < 2*K; k++)
				T[(l/2)%L0][(2*K*threadIdx.x + k)%32][(2*K*threadIdx.x + k)/32+2*K*(l%2)] = u[k];
        __syncthreads();
        if (0 <= l && l < 2*L0 && threadIdx.y < b)
			for (int k = 0; k < 2*K; k++)
                if (32*k + threadIdx.x < N)
                    u_out[32*k + threadIdx.x] = T[(l/2)%L0][threadIdx.x][k+2*K*(l%2)];
		__syncthreads();
	}
}

#undef K
#undef L0
#undef nL
#undef fname