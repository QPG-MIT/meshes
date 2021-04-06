// meshes/gpu/gmem.cu
// Ryan Hamerly, 4/5/21
//
// Macros for interfacing with global memory:
//   load_u, load_u_du:   read from u_in [gmem] -> T [smem] -> u [regs]
//   save_u, save_u_du:   write from u [regs] -> T [smem] -> u_out [gmem]
//   load_T:              read from (s, p) [gmem] -> T [smem]
//   load_pos_cache:      read (lens, shifts) [gmem] -> (lens_cache, shifts_cache) [smem]
//
// History:
//   04/03/21: Implemented the algorithms in fwdprop.cu.
//   04/05/21: Generalized code with macros, split off to gmem.cu.


// Load u coalesced, gmem -> smem.  Then reorder strided, smem -> regs.  (Direct gmem -> regs would be strided, poor BW).
// Loads to smem can be parallelized over thread warps, but if threadIdx.y > 2*L, then there's not enough smem to do them
// all at once, hence the thread-sync'd for loop.
//
// It's a macro because I use it in multiple functions.  The code gets inlined anyway because the array u[] is in registers.
// Tried to write an inline function but multidimensional array arguments (of undefined size) seem unsupported.
//
// Python code that checks this (for deubugging):
/*
(B, L, K) = (4, 2, 2)    # Batch size, mesh length, mesh width/64.
ldu = 2*K*32; u_in = np.arange(B*ldu); T = np.zeros([L, 32, 4*K]).astype(int); u = np.zeros([B, 32, 2*K]).astype(int)

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

// One-line code segments that get reused in the loaders below.
//
// load_readcode: reads u_in -> T.  
// load_shflcode: shuffles indices to load T into registers u[k].
// save_shflcode: shuffles indices to put registers u[k] into T.
// save_writcode: writes T -> u_out.
#define load_readcode(T, u_in, cond)    T[(l/pack_u)%L0][threadIdx.x][k+2*K*(l%pack_u)] = cond ? u_in[32*k + threadIdx.x] : 0
#define load_shflcode(T, u)             u[k] = T[(l/pack_u)%L0][(2*K*threadIdx.x + k)%32][(2*K*threadIdx.x + k)/32+2*K*(l%pack_u)]
#define save_shflcode(T, u)             T[(l/pack_u)%L0][(2*K*threadIdx.x + k)%32][(2*K*threadIdx.x + k)/32+2*K*(l%pack_u)] = u[k]
#define save_writcode(T, u_out, cond)   if (cond) {u_out[32*k + threadIdx.x] = T[(l/pack_u)%L0][threadIdx.x][k+2*K*(l%pack_u)];}
// Generic loader.  Permutation keeps gmem reads coalesced.
#define load_generic(read_code, shuffle_code, zero_code) { \
for (int i = 0; i < b; i += pack_u*L0) \
{ \
    int l = threadIdx.y - i; \
    if (0 <= l && l < pack_u*L0 && threadIdx.y < b) \
        for (int k = 0; k < 2*K; k++) \
            if (32*k + threadIdx.x < N) {read_code;} \
    __syncthreads(); \
    if (0 <= l && l < pack_u*L0) \
        for (int k = 0; k < 2*K; k++) \
        { \
            if (2*K*threadIdx.x + k < N) {shuffle_code;} \
            else {zero_code;} \
        } \
    __syncthreads(); \
} \
}
// Generic saver.  Same permutation as for input, but reversed.
#define save_generic(shuffle_code, save_code) { \
for (int i = 0; i < b; i += pack_u*L0) \
{ \
    int l = threadIdx.y - i; \
    if (0 <= l && l < pack_u*L0) \
        for (int k = 0; k < 2*K; k++) {shuffle_code;} \
    __syncthreads(); \
    if (0 <= l && l < pack_u*L0 && threadIdx.y < b) \
        for (int k = 0; k < 2*K; k++) \
            if (32*k + threadIdx.x < N) {save_code;} \
    __syncthreads(); \
} \
}
// Loads u_in -> u.
#define load_u \
    load_generic(load_readcode(T, u_in, true), load_shflcode(T, u), u[k] = 0)
// Loads u_in -> u, du_in -> du
#define load_u_du \
    load_generic( \
        load_readcode(T, u_in, true); load_readcode(dT, du_in, du_in), \
        load_shflcode(T, u); load_shflcode(dT, du), \
        u[k] = 0; du[k] = 0)
// Saves u -> u_out.
#define save_u \
    save_generic(save_shflcode(T, u), save_writcode(T, u_out, true))
// Saves u -> u_out, du -> du_out.
#define save_u_du \
    save_generic( \
        save_shflcode(T, u); save_shflcode(dT, du), \
        save_writcode(T, u_out, true); save_writcode(dT, du_out, du_out))


// Every L_preload = L0*nL layers, reload the cache of shifts and lengths.  This is done less frequently than
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
#define load_pos_cache { \
if (x % L_preload == 0) \
{ \
    for (int i = 0; i < L_preload; i += 32*blockDim.y) \
    { \
        int id = i + 32*threadIdx.y + threadIdx.x; \
        if (id < L_preload && x + id < L) \
        { \
            lens_cache[id]   = lens[x + id]; \
            shifts_cache[id] = shifts[x + id]; \
        } \
    } \
    __syncthreads(); \
} \
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
#define load_T(code_in, code_out) { \
for (int i = 0; i < K*L_ker; i += blockDim.y) \
{ \
    int l = (i + threadIdx.y)/K, m = (i + threadIdx.y)%K; \
    if (l < L_blk) \
    { \
        int dm = (m*32 + threadIdx.x); \
        int idx_p = ldp*l + 2*dm, idx_s = lds*l + 2*dm; \
        if (dm >= shifts_cache[(x+l) % L_preload]/2 && \
            dm <  shifts_cache[(x+l) % L_preload]/2 + lens_cache[(x+l) % L_preload]) \
        { \
            code_in; \
        } \
        else \
        { \
            code_out; \
        } \
    } \
} \
__syncthreads(); \
}

