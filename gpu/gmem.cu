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
//   04/06/21: Added macros for back-propagation.


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
ldu = 2*K*32; u_in = np.arange(B*ldu); T = np.zeros([L, 4*K, 32]).astype(int); u = np.zeros([B, 32, 2*K]).astype(int)

for i in range(0, B, 2*L):
    print (f"i = {i}")
    for tix_y in range(B):
        if (i <= tix_y < i + 2*L and tix_y < B):
            print (f"tix_y = {tix_y}")
            for k in range(2*K):
                ind = 32*k + ldu*tix_y
                print (f"idx=[{(tix_y//2)%L}][{k+2*K*(tix_y%2)}][:32] <-- [{ind}+:32]")
                for tix_x in range(32):
                    T[(tix_y//2)%L, k+2*K*(tix_y%2), tix_x] = u_in[ind+tix_x]
            for k in range(2*K):
                for tix_x in range(32):
                    u[tix_y, tix_x, k] = T[(tix_y//2)%L, (2*K*tix_x + k)//32 + 2*K*(tix_y%2), (2*K*tix_x + k)%32]       
(u.flatten() == u_in).all()
*/

// One-line code segments that get reused in the loaders below.
//
// load_readcode: reads u_in -> T.  
// load_shflcode: shuffles indices to load T into registers u[k].
// save_shflcode: shuffles indices to put registers u[k] into T.
// save_writcode: writes T -> u_out.
#define load_readcode(T, u_in, cond)    T[(l/pack_u)%L0][k+2*K*(l%pack_u)][threadIdx.x] = cond ? u_in[32*k + threadIdx.x] : 0
#define load_shflcode(T, u)             u[k] = T[(l/pack_u)%L0][(2*K*threadIdx.x + k)/32+2*K*(l%pack_u)][(2*K*threadIdx.x + k)%32]
#define save_shflcode(T, u)             T[(l/pack_u)%L0][(2*K*threadIdx.x + k)/32+2*K*(l%pack_u)][(2*K*threadIdx.x + k)%32] = u[k]
#define save_writcode(T, u_out, cond)   if (cond) {u_out[32*k + threadIdx.x] = T[(l/pack_u)%L0][k+2*K*(l%pack_u)][threadIdx.x];}

#define load_readcode_sym(T, u_in, cond) \
    {complex64 u_temp = cond ? u_in[32*k + threadIdx.x] : 0; \
    T[2*(l%(L0/2))+(k/K)][2*(k%K)  ][threadIdx.x] = u_temp.real(); \
    T[2*(l%(L0/2))+(k/K)][2*(k%K)+1][threadIdx.x] = u_temp.imag();}
#define save_writcode_sym(T, u_out, cond) \
    if (cond) {u_out[32*k + threadIdx.x] = \
        complex64(T[2*(l%(L0/2))+(k/K)][2*(k%K)  ][threadIdx.x], \
                  T[2*(l%(L0/2))+(k/K)][2*(k%K)+1][threadIdx.x]); }

// TODO -- check which is faster (using idx adds registers, weirdly)
/*#define load_shflcode_sym(T, u, v) \
    {int idx = 2*K*threadIdx.x + k; \
    u[k] = T[2*(l%(L0/2))+(idx/32/K)][2*((idx/32)%K)  ][idx%32]; \
    v[k] = T[2*(l%(L0/2))+(idx/32/K)][2*((idx/32)%K)+1][idx%32];}
#define save_shflcode_sym(T, u, v) \
    {int idx = 2*K*threadIdx.x + k; \
    T[2*(l%(L0/2))+(idx/32/K)][2*((idx/32)%K)  ][idx%32] = u[k]; \
    T[2*(l%(L0/2))+(idx/32/K)][2*((idx/32)%K)+1][idx%32] = v[k];}*/

#define load_shflcode_sym(T, u, v) \
    u[k] = T[2*(l%(L0/2))+((2*K*threadIdx.x + k)/32/K)][2*(((2*K*threadIdx.x + k)/32)%K)  ][(2*K*threadIdx.x + k)%32]; \
    v[k] = T[2*(l%(L0/2))+((2*K*threadIdx.x + k)/32/K)][2*(((2*K*threadIdx.x + k)/32)%K)+1][(2*K*threadIdx.x + k)%32];
#define save_shflcode_sym(T, u, v) \
    T[2*(l%(L0/2))+((2*K*threadIdx.x + k)/32/K)][2*(((2*K*threadIdx.x + k)/32)%K)  ][(2*K*threadIdx.x + k)%32] = u[k]; \
    T[2*(l%(L0/2))+((2*K*threadIdx.x + k)/32/K)][2*(((2*K*threadIdx.x + k)/32)%K)+1][(2*K*threadIdx.x + k)%32] = v[k];


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
#define load_generic_sym(read_code, shuffle_code, zero_code) { \
for (int i = 0; i < b; i += L0/2) \
{ \
    int l = threadIdx.y - i; \
    if (0 <= l && l < L0/2 && threadIdx.y < b) \
        for (int k = 0; k < 2*K; k++) \
            if (32*k + threadIdx.x < N) {read_code;} \
    __syncthreads(); \
    if (0 <= l && l < L0/2) \
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
#define save_generic_sym(shuffle_code, save_code) { \
for (int i = 0; i < b; i += L0/2) \
{ \
    int l = threadIdx.y - i; \
    if (0 <= l && l < L0/2) \
        for (int k = 0; k < 2*K; k++) {shuffle_code;} \
    __syncthreads(); \
    if (0 <= l && l < L0/2 && threadIdx.y < b) \
        for (int k = 0; k < 2*K; k++) \
            if (32*k + threadIdx.x < N) {save_code;} \
    __syncthreads(); \
} \
}
// Loads u_in -> u.
#define load_u(u, u_in) \
    load_generic(load_readcode(T, u_in, true), load_shflcode(T, u), u[k] = 0)
#define load_u_sym(u, v, u_in) \
    load_generic_sym(load_readcode_sym(T, u_in, true), load_shflcode_sym(T, u, v), u[k] = 0; v[k] = 0)
// Loads u_in -> u, du_in -> du
#define load_u_du(u, du, u_in, du_in) \
    load_generic( \
        load_readcode(T, u_in, true); load_readcode(dT, du_in, du_in), \
        load_shflcode(T, u); load_shflcode(dT, du), \
        u[k] = 0; du[k] = 0)
#define load_u_du_sym(u, v, du, dv, u_in, du_in) \
    load_generic_sym( \
        load_readcode_sym(T, u_in, true); load_readcode_sym(dT, du_in, du_in), \
        load_shflcode_sym(T, u, v); load_shflcode_sym(dT, du, dv), \
        u[k] = 0; v[k] = 0; du[k] = 0; dv[k] = 0)
// Saves u -> u_out.
#define save_u(u, u_out) \
    save_generic(save_shflcode(T, u), save_writcode(T, u_out, true))
#define save_u_sym(u, v, u_out) \
    save_generic_sym(save_shflcode_sym(T, u, v), save_writcode_sym(T, u_out, true))
// Saves u -> u_out, du -> du_out.
#define save_u_du(u, du, u_out, du_out) \
    save_generic( \
        save_shflcode(T, u); save_shflcode(dT, du), \
        save_writcode(T, u_out, true); save_writcode(dT, du_out, du_out))
#define save_u_du_sym(u, v, du, dv, u_out, du_out) \
    save_generic_sym( \
        save_shflcode_sym(T, u, v); save_shflcode_sym(dT, du, dv), \
        save_writcode_sym(T, u_out, true); save_writcode_sym(dT, du_out, du_out))

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
#define load_pos_cache(sign) \
    if (x % L_preload == 0) \
    { \
        for (int i = 0; i < L_preload; i += 32*blockDim.y) \
        { \
            int id = i + 32*threadIdx.y + threadIdx.x; \
            if (id < L_preload && x + id < L) \
            { \
                lens_cache[id]   = lens[sign*(x + id)]; \
                shifts_cache[id] = shifts[sign*(x + id)]; \
            } \
        } \
        __syncthreads(); \
    }
#define load_pos_cache_fwd load_pos_cache(+1)
#define load_pos_cache_rev load_pos_cache(-1)

// Load T (coalesced in gmem, strided in smem).
// Python code that checks this (for debugging):
/*
B = 3    # Batch size (variable)
L = 10   # Length of mesh
K = 2    # Width of mesh: 64*K (2*K variables / thread)
ldp = lds = 2*K*32
p = np.array([1*np.arange(32*K*L), 2*np.arange(32*K*L)]).T.flatten()
s = np.array([3*np.arange(32*K*L), 4*np.arange(32*K*L)]).T.flatten()
T = np.zeros([L, 4*K, 32], dtype=int)

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
                T[l, 4*(dm%K):4*(dm%K+1), dm//K] = T_test(p[idx_p], p[idx_p+1],
                                                          s[idx_s], s[idx_s+1])
(T.reshape(L*32*K, 4) == np.array([p[::2], p[1::2], s[::2], s[1::2]]).T).all()
*/

#define i1_T (l/pack_T)
#define i2_T (stride_T*(dm%K + K*(l%pack_T)))
#define i3_T (dm/K)

#define matrix_io(code_in, code_out) { \
for (int i = 0; i < K*L_ker; i += blockDim.y) \
{ \
    int l = (i + threadIdx.y)/K, m = (i + threadIdx.y)%K; \
    if (l < L_blk) \
    { \
        int dm = (m*32 + threadIdx.x); \
        int idx_p = ldp*l + 2*dm, idx_s = s ? (lds*l + stride_s*dm) : 0; \
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

// Loads matrix T
#define load_T \
    matrix_io(Tij_mzi(&p[idx_p], (float*) 0, &s[idx_s], &T[i1_T][i2_T][i3_T], (complex64 *) 0, false), \
              Tij_identity(&T[i1_T][i2_T][i3_T], (complex64 *) 0))
#define load_T_sym \
    matrix_io(Tij_mzi_sym(&p[idx_p], (float *) 0, &s[idx_s], &T[i1_T][i2_T][i3_T], (float *) 0, cartesian, false), \
              Tij_identity_sym(&T[i1_T][i2_T][i3_T], (float *) 0))
// Loads matrix T and its differential dT.
#define load_T_dT \
    matrix_io(Tij_mzi(&p[idx_p], &dp[idx_p], &s[idx_s], &T[i1_T][i2_T][i3_T], &dT[i1_T][i2_T][i3_T], true), \
              Tij_identity(&T[i1_T][i2_T][i3_T], &dT[i1_T][i2_T][i3_T]))
#define load_T_dT_sym \
    matrix_io(Tij_mzi_sym(&p[idx_p], &dp[idx_p], &s[idx_s], &T[i1_T][i2_T][i3_T],  &dT[i1_T][i2_T][i3_T], \
                          cartesian, true), \
              Tij_identity_sym(&T[i1_T][i2_T][i3_T], &dT[i1_T][i2_T][i3_T]))

// NOTE: for packed symmetric matrices (pack_T=2), only works if ps_cache can be packed as well (no s data stored),
// so that stride_T = 2, stride_P = 2.  I think.  Maybe more general if you define these at compile time.
#define load_T_dT_bk \
    matrix_io(Tij_mzi(&p[idx_p], &dp[idx_p], &s[idx_s], &T[i1_T][i2_T][i3_T], &dT[i1_T][i2_T][i3_T], false), \
              Tij_identity(&T[i1_T][i2_T][i3_T], &dT[i1_T][i2_T][i3_T]))
#define load_T_dT_bk_sym \
    matrix_io(Tij_mzi_sym(&p[idx_p], &dp[idx_p], &s[idx_s], &T[i1_T][i2_T][i3_T], &dT[i1_T][i2_T][i3_T], \
                          cartesian, false), \
              Tij_identity_sym(&T[i1_T][i2_T][i3_T], &dT[i1_T][i2_T][i3_T]))

#define save_dp \
    matrix_io(dp_mzi(&p[idx_p], &s[idx_s], &dT[i1_T][i2_T][i3_T], &dp[idx_p]), )
#define save_dp_sym \
    matrix_io(dp_mzi_sym(&p[idx_p], &s[idx_s], &dT[i1_T][i2_T][i3_T], &dp[idx_p]), )
