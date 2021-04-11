# meshes/gpu/test.py
# Ryan Hamerly, 4/5/21
#
# Testing utility for this package.  Tests both speed and accuracy.
#
# History:
#   04/03/21: Created this file.
#   04/05/21: Added support for forward differentiation.
#   04/06/21: Reverse differentiation and CPU timing comparison.
#   04/10/21: Added symmetric and orthogonal representations.


import numpy as np
import cupy as cp
from time import time
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import sys
from cupy_backends.cuda.api.driver import CUDADriverError


# Command line options.
acc   = ('-a'  in sys.argv)
speed = ('-s'  in sys.argv)
inf   = ('inf' in sys.argv)
fd    = ('fd'  in sys.argv)
bd    = ('bd'  in sys.argv)
cpu   = ('cpu' in sys.argv)
mzi   = ('-mzi' in sys.argv)
sym   = ('-sym' in sys.argv)
orth  = ('-orth' in sys.argv)

if (not (acc or speed) or not (inf or fd or bd or cpu)):
    print ("Usage: python test.py [-a] [-s] [-mzi] [-sym] [-orth] [inf] [fwd] [rev] [cpu]")
    print ("[-a] Test accuracy.")
    print ("[-s] Test speed.")
    print ("[inf] Test inference function.")
    print ("[fd]  Test forward error propagation.")
    print ("[bd] Test error back-propagation.")
    print ("[-mzi] Standard MZI")
    print ("[-sym] Symmetric crossing")
    print ("[-orth] Orthogonal (real) crossing (not yet supported).")
    exit()
    
assert (mzi + sym + orth <= 1)
mode = 'mzi'
if sym: mode = 'sym'
if orth: mode = 'orth'

if (inf or fd or bd):
    print ("Loading module.\n")
    mod = cp.RawModule(path="meshprop.cubin")
    
# Step 1: Accuracy Test.
# Runs a bunch of parameters, checks GPU result against block-diagonal matrix multiplication.
# Randomly varies N, L, B, nWarp, shifts, lens.
def test_fwd_acc(diff=False, mode='mzi'):
    (n_p, n_s, dtype) = dict(mzi=(2, 2, cp.complex64), sym=(2, 1, cp.complex64), orth=(1, 0, cp.float32))[mode]
    post = {'mzi': '_mzi', 'sym': '_sym', 'orth': '_orth'}[mode]
    print ("Accuracy Test: " + ("fwddiff_N256" if diff else "fwdprop_N256")+post)
    print ("--------------------------------------")
    for moo in range(20):
        (K, L, B) = (4, np.random.randint(4, 21), np.random.randint(4, 41)); 
        fname = (f"fwddiff_N{64*K}" if diff else f"fwdprop_N{64*K}")+post
        N = np.random.randint(128, 256+1); Nwarp = np.random.randint(2, 21 if diff else 31); Nblk = int(np.ceil(B/Nwarp))
        print (f"N={N}, L={L:2d}, B={B:2d}, Nwarp={Nwarp:2d}...", end="")
        # Inputs.
        (p, dp) = np.random.randn(2, L, N//2, n_p); s = np.random.randn(L, N//2, n_s)
        (u_in, du_in) = np.random.randn(2, B, N, 2).dot([1, 1j]); 
        if (mode == 'orth'): u_in = np.real(u_in); du_in = np.real(du_in)
        ldp = p[0].size; lds = s[0].size; ldu = N
        shifts = np.random.randint([N-1]*L); lens = np.random.randint((N-shifts)//2)   # Random splitter placement.
        # GPU code.
        func = mod.get_function(fname)
        shifts_d = cp.asarray(shifts, dtype=cp.int32); lens_d = cp.asarray(lens, dtype=cp.int32)
        (p_d, dp_d, s_d) = [cp.asarray(x, dtype=cp.float32) for x in (p, dp, s)]
        (N_d, L_d, B_d, ldp_d, lds_d, ldu_d) = map(cp.int32, (N, L, B, ldp, lds, ldu))
        in_d = cp.asarray(u_in, dtype=dtype); out_d = cp.asarray(in_d*0); 
        din_d = cp.asarray(du_in, dtype=dtype); dout_d = cp.asarray(din_d*0)
        args = ([N_d, L_d, B_d, lens_d, shifts_d, p_d] + [dp_d][:diff] + [ldp_d] + [s_d, lds_d][:2*(mode!='orth')] + 
                [in_d] + [din_d][:diff] + [out_d] + [dout_d][:diff] + [ldu_d] + [cp.bool(False)][:mode=='sym'])
        #args = ([N_d, L_d, B_d, lens_d, shifts_d] + 
        #        ([p_d, dp_d, ldp_d, s_d, lds_d, in_d, din_d, out_d, dout_d, ldu_d] if diff else
        #         [p_d, ldp_d, s_d, lds_d, in_d, out_d, ldu_d]) + {'mzi': [], 'sym': [cp.bool(False)]}[mode])
        func((Nblk,), (32,Nwarp), tuple(args))
        u_out = out_d.get(); du_out = dout_d.get()
        # CPU code for comparison.
        def Tij_cpu(p, s):
            if mode == 'mzi':
                (theta, phi) = p.T; (a, b) = s.T
                (Cp, Cm, C, Sp, Sm, S) = [fn(x) for fn in [np.cos, np.sin] for x in [a+b, a-b, theta/2]]
                return np.exp(1j*theta/2) * np.array([[np.exp(1j*phi) * (1j*S*Cm - C*Sp),    1j*C*Cp - S*Sm],
                                                      [np.exp(1j*phi) * (1j*C*Cp + S*Sm),   -1j*S*Cm - C*Sp]])
            elif mode == 'sym':
                (theta, phi) = p.T; (beta,) = s.T
                (C, C_2a, S, S_2a) = [fn(x) for fn in [np.cos, np.sin] for x in [theta/2, 2*beta]]
                return np.array([[np.exp(1j*phi)*(S + 1j*C*S_2a),  1j*C*C_2a],
                                 [1j*C*C_2a, np.exp(-1j*phi)*(S - 1j*C*S_2a)]])
            elif mode == 'orth':
                (theta,) = p.T; (C, S) = (np.cos(theta/2), np.sin(theta/2))
                return np.array([[S, -C], [C, S]])
        u   = u_in;          du  = du_in
        u_p = u + 1e-5*du;   u_m = u - 1e-5*du
        for i in range(L):
            mats_i = Tij_cpu(p[i], s[i]).transpose(2, 0, 1)
            M = block_diag(np.eye(shifts[i]), *mats_i[shifts[i]//2:shifts[i]//2+lens[i]], np.eye(N-shifts[i]-2*lens[i]))
            u = u.dot(M.T)
            if diff:
                mats_p = Tij_cpu(p[i] + 1e-5*dp[i], s[i]).transpose(2, 0, 1)
                mats_m = Tij_cpu(p[i] - 1e-5*dp[i], s[i]).transpose(2, 0, 1)
                Mp = block_diag(np.eye(shifts[i]), *mats_p[shifts[i]//2:shifts[i]//2+lens[i]], np.eye(N-shifts[i]-2*lens[i]))
                Mm = block_diag(np.eye(shifts[i]), *mats_m[shifts[i]//2:shifts[i]//2+lens[i]], np.eye(N-shifts[i]-2*lens[i]))
                u_p = u_p.dot(Mp.T); u_m = u_m.dot(Mm.T)
        du = (u_p - u_m) / 2e-5
        # Error evaluation.
        err = np.linalg.norm(u_out-u, axis=1) / np.linalg.norm(u, axis=1)
        d_err = np.linalg.norm(du_out-du, axis=1) / (np.linalg.norm(du, axis=1)+1e-15) * diff
        errT = np.linalg.norm(u_out-u, axis=0) / np.linalg.norm(u, axis=0)
        d_errT = np.linalg.norm(du_out-du, axis=0) / (np.linalg.norm(du, axis=0)+1e-15) * diff
        if ((err < 1e-4).all() and (d_err < 1e-4).all()): print("Success.")
        else: 
            print(f"FAIL!  p: {(err > 1e-4).sum()}/{len(err)} had relative error > 1e-4."); 
            print("[p] err/batch = ", '[' + "".join(np.array(['*', '.'])[(err < 1e-4).astype(int)]) + ']')
            print("[p] err/ind   = ", '[' + "".join(np.array(['*', '.'])[(errT < 1e-4).astype(int)]) + ']')
            print("[dp] err/batch = ", '[' + "".join(np.array(['*', '.'])[(d_err < 1e-4).astype(int)]) + ']')
            print("[dp] err/ind   = ", '[' + "".join(np.array(['*', '.'])[(d_errT < 1e-4).astype(int)]) + ']')
    print()    
def test_rev_acc(mode):
    (n_p, n_s, dtype) = dict(mzi=(2, 2, cp.complex64), sym=(2, 1, cp.complex64), orth=(1, 0, cp.float32))[mode]
    post = {'mzi': '_mzi', 'sym': '_sym', 'orth': '_orth'}[mode]
    print ("Accuracy Test: backdiff_N256"+post)
    print ("--------------------------------------")
    fwd = mod.get_function("fwdprop_N256"+post)
    rev = mod.get_function("backdiff_N256"+post)
    for moo in range(20):
        (K, L, B) = (4, np.random.randint(4, 21), np.random.randint(4, 41)); 
        N = np.random.randint(128, 256+1); Nwarp = np.random.randint(2, 29); Nblk = int(np.ceil(B/Nwarp))
        print (f"N={N}, L={L:2d}, B={B:2d}, Nwarp={Nwarp:2d}...", end="")
        shifts = np.random.randint([N-1]*L); lens = np.random.randint((N-shifts)//2)   # Random splitter placement.
        p = np.random.randn(L, N//2, n_p).astype(np.float32); 
        s = np.random.randn(L, N//2, n_s).astype(np.float32)
        (u_in, u0) = np.random.randn(2, B, N, 2).dot([1, 1j]).astype(np.complex64)
        if (mode == 'orth'): u_in = np.real(u_in); u0 = np.real(u0)
        ldp = p[0].size; lds = s[0].size; ldu = N
        # Send data to the GPU.
        (p_d, s_d, u_in_d, u0_d, lens_d, shifts_d) = map(cp.asarray, (p, s, u_in, u0, lens, shifts))
        u_out_d = cp.zeros(u_in_d.shape, dtype=dtype); u_in2_d = cp.zeros(u_in_d.shape, dtype=dtype)
        dJdu_in_d = cp.zeros(u_in_d.shape, dtype=dtype); dp_d = cp.zeros(p_d.shape, dtype=cp.float32)
        args_fwd = ([cp.int32(N), cp.int32(L), cp.int32(B), lens_d, shifts_d, p_d, cp.int32(ldp)] + 
                    [s_d, cp.int32(lds)][:2*(mode!='orth')] + [u_in_d, u_out_d, cp.int32(ldu)] + 
                    [cp.bool(False)][:mode=='sym'])
        # Fwd-propagate, get gradient, then back-propagate.
        fwd((Nblk,), (32, Nwarp), tuple(args_fwd))
        dJdu_out_d = 2*(u_out_d - u0_d)
        args_rev = ([cp.int32(N), cp.int32(L), cp.int32(B), lens_d, shifts_d, p_d, dp_d, cp.int32(ldp)] + 
                    [s_d, cp.int32(lds)][:2*(mode!='orth')] + [u_out_d, dJdu_out_d, u_in2_d, dJdu_in_d, cp.int32(ldu)] + 
                    [cp.bool(False)][:mode=='sym'])
        rev((Nblk,), (32, Nwarp), tuple(args_rev))
        (u_out, dJdu_out, u_in2, dJdu_in2, dp) = map(cp.asnumpy, (u_out_d, dJdu_out_d, u_in2_d, dJdu_in_d, dp_d))

        # Error in back-propagated field.
        err_uin = np.linalg.norm(u_in2 - u_in) / np.linalg.norm(u_in)
        # Error in the gradient.
        def numdiff(dp):
            pp = (p + 1e-3*dp).astype(np.float32); pm = (p - 1e-3*dp).astype(np.float32)
            (pp_d, pm_d, s_d, u_in_d, u0_d, lens_d, shifts_d) = map(cp.asarray, (pp, pm, s, u_in, u0, lens, shifts))
            args_fwd[5] = pp_d;
            fwd((Nblk,), (32, Nwarp), tuple(args_fwd))
            Jp = (cp.abs(u_out_d - u0_d)**2).sum().get(); args_fwd[5] = pm_d;
            fwd((Nblk,), (32, Nwarp), tuple(args_fwd))
            Jm = (cp.abs(u_out_d - u0_d)**2).sum().get()
            return (Jp - Jm) / 2e-3
        errList = []
        for i in range(50):
            dp1 = np.random.randn(*dp.shape); errList.append(numdiff(dp1) / (dp * dp1).sum())
        err_dp = np.abs(np.median(errList) - 1)

        if (err_uin < 1e-4) and (err_dp < 1e-2): print ("Success.")
        else: 
            print (f"FAIL!  err[dJ/du_in]={err_uin:.2e}, err[dJ/dp]={err_dp:.2e}")
    print()

# Step 2: Speed Test.
# Performance is a function of mesh size N, depth L, batch size B, and warps/block.  The latter
# is a tuning parameter that must be swept.
def test_fwd_speed(diff, mode='mzi'):
    (n_p, n_s, dtype) = dict(mzi=(2, 2, cp.complex64), sym=(2, 1, cp.complex64), orth=(1, 0, cp.float32))[mode]
    post = {'mzi': '_mzi', 'sym': '_sym', 'orth': '_orth'}[mode]
    print ("Speed Test: " + ("fwddiff_N***" if diff else "fwdprop_N***")+post)
    print ("--------------------------------------")
    def timetest(N, L, B, Nwarp):
        K = N//32
        Nblk = int(np.ceil(B/Nwarp))
        func = mod.get_function((f"fwddiff_N{N}" if diff else f"fwdprop_N{N}")+post)
        p_d = cp.random.randn(L, 32*K, n_p, dtype=np.float32);
        s_d = cp.random.randn(L, 32*K, n_s, dtype=np.float32)
        if (mode == 'orth'): in_d = cp.random.randn(B, 32*2*K, dtype=np.float32)
        else: in_d = cp.random.randn(B, 32*2*K, 2, dtype=np.float32).dot(cp.asarray([1.0, 1.0j], dtype=np.complex64))
        shifts_d = cp.arange(L, dtype=cp.int32) % 2; lens_d = (32*K) - shifts_d;
        out_d = cp.zeros([B, 32*2*K], dtype=np.complex64)
        if diff:
            dp_d = cp.random.randn(L, 32*K, 2, dtype=np.float32); 
            if (mode == 'orth'):
                dout_d = cp.zeros([B, 32*2*K], dtype=np.float32); din_d = cp.random.randn(B, 32*2*K, dtype=np.float32)
            else:
                dout_d = cp.zeros([B, 32*2*K], dtype=np.complex64)
                din_d = cp.random.randn(B, 32*2*K, 2, dtype=np.float32).dot(cp.asarray([1.0, 1.0j], dtype=np.complex64))
        else:
            (dp_d, dout_d, din_d) = (None, None, None)
        ldp = (32*K)*n_p; lds = (32*K)*n_s; ldu = 2*(32*K)
        (N_d, L_d, B_d, ldp_d, lds_d, ldu_d) = map(cp.int32, (N, L, B, ldp, lds, ldu))
        t = 0; ct = 1
        args = ([N_d, L_d, B_d, lens_d, shifts_d, p_d] + [dp_d][:diff] + [ldp_d] + [s_d, lds_d][:2*(mode!='orth')] + 
                ([in_d, din_d, out_d, dout_d, ldu_d] if diff else [in_d, out_d, ldu_d]) + [cp.bool(False)][:mode=='sym'])
        while (t < 1e-2):
            cp.cuda.runtime.deviceSynchronize(); t = time()
            for i in range(ct):
                func((Nblk,), (32, Nwarp), tuple(args))
            cp.cuda.runtime.deviceSynchronize(); t = time() - t; ct *= 2
        return t / (ct/2)

    flops1 = np.zeros([8, 32])*np.nan; flops2 = np.zeros([8, 32])*np.nan
    wsList = np.zeros([2, 8], dtype=np.int)

    print ("Square Matrices: N x N x N")
    Nlist = [64, 128, 192, 256, 320, 384, 512, 640]
    for (i, N) in enumerate(Nlist):
        print(f"N = {N:4d}: ", end="")
        for (j, ws) in enumerate(range(1, 33)):
            try:
                t = timetest(N, N, N, ws); 
            except CUDADriverError as e:
                print ("x" + " "*(31-j), end=""); break
            flops1[i, j] = (3 if diff else 1) * (32 * (N*N*N/2) / t) / 1e9; print (".", end="", flush=True)
        j = np.argmax(np.nan_to_num(flops1[i])); wsList[0, i] = j+1
        print(f" {flops1[i,j]:6.1f} GFLOP/s [{j+1:2d}*32={32*(j+1):4d} threads]")
    print ("Fat Matrices: N x N x 4096")
    for (i, N) in enumerate(Nlist):
        print(f"N = {N:4d}: ", end="")
        for (j, ws) in enumerate(range(1, 33)):
            try:
                t = timetest(N, N, 4096, ws); 
            except CUDADriverError as e:
                print ("x" + " "*(31-j), end=""); break
            flops2[i, j] = (3 if diff else 1) * (32 * (N*N*4096/2) / t) / 1e9; print (".", end="", flush=True)
        j = np.argmax(np.nan_to_num(flops2[i])); wsList[1, i] = j+1
        print(f" {flops2[i,j]:6.1f} GFLOP/s [{j+1:2d}*32={32*(j+1):4d} threads]")
    print()

    np.savetxt("tuned_warpsize.txt", np.concatenate([np.array([Nlist]), wsList], axis=0).T, delimiter='\t', fmt='%d')

    (f, (ax1, ax2)) = plt.subplots(1, 2, figsize=(9, 4.5), sharex=True, sharey=True)
    (flops_mesh1, flops_mesh2) = ([], [])
    for (ax, flops_i, flops_max_i) in zip([ax1, ax2], [flops1, flops2], [flops_mesh1, flops_mesh2]):
        for (j, flops_ij) in enumerate(flops_i):
            ax.plot(range(1, 33), flops_ij, '.-')
        ax.plot([np.nan], 'o', mec='k', mfc='w')
        for (j, flops_ij) in enumerate(flops_i):
            k = np.argmax(np.nan_to_num(flops_ij))
            ax.plot([k+1], [flops_ij[k]], 'o', mec='C'+str(j), mfc='w')
            flops_max_i.append(flops_ij[k])
        ax.set_xlim(-1, 33); ax.set_ylim(-20, 900); ax.grid()
    ax2.legend(["N = " + str(Nlist[0])] + Nlist[1:], loc=4, ncol=2, framealpha=1)

    ax1.set_xlabel(r"# Warps"); ax2.set_xlabel(r"# Warps"); ax1.set_ylabel("K40 Perf (GFLOP/s)")
    ax1.set_title("N x N x N"); ax2.set_title("N x N x 4096")
    plt.tight_layout()
    plt.savefig("test-fig1.pdf", format="pdf")

def test_rev_speed(mode):
    (n_p, n_s, dtype) = dict(mzi=(2, 2, cp.complex64), sym=(2, 1, cp.complex64), orth=(1, 0, cp.float32))[mode]
    post = {'mzi': '_mzi', 'sym': '_sym', 'orth': '_orth'}[mode]
    print ("Speed Test: backdiff_N***"+post)
    print ("--------------------------------------")
    def timetest(N, L, B, Nwarp):
        K = N//32
        Nblk = int(np.ceil(B/Nwarp))
        func = mod.get_function(f"backdiff_N{N}"+post)
        p_d = cp.random.randn(L, 32*K, n_p, dtype=np.float32); s_d = cp.random.randn(L, 32*K, n_s, dtype=np.float32)
        if (mode == 'orth'):
            u_out_d = cp.random.randn(B, 32*2*K, dtype=np.float32)
            dJdu_out_d = cp.random.randn(B, 32*2*K, dtype=np.float32)
        else:
            u_out_d = cp.random.randn(B, 32*2*K, 2, dtype=np.float32).dot(cp.asarray([1.0, 1.0j], dtype=np.complex64))
            dJdu_out_d = cp.random.randn(B, 32*2*K, 2, dtype=np.float32).dot(cp.asarray([1.0, 1.0j], dtype=np.complex64))
        shifts_d = cp.arange(L, dtype=cp.int32) % 2; lens_d = (32*K) - shifts_d;
        u_in_d = cp.zeros([B, 32*2*K], dtype=dtype)
        dJdu_in_d = cp.zeros([B, 32*2*K], dtype=dtype)
        dp_d = cp.random.randn(L, 32*K, n_p, dtype=np.float32)
        args_post = {'mzi': (), 'sym': (cp.bool(False),), 'orth': ()}[mode]

        ldp = (32*K)*n_p; lds = (32*K)*n_s; ldu = 2*(32*K)
        t = 0; ct = 1
        args = ([cp.int32(N), cp.int32(L), cp.int32(B), lens_d, shifts_d, p_d, dp_d, cp.int32(ldp)] +
                [s_d, cp.int32(lds)][:2*(mode!='orth')] + [u_out_d, dJdu_out_d, u_in_d, dJdu_in_d, cp.int32(ldu)] + 
                [cp.bool(False)][:mode=='sym'])
        while (t < 1e-2):
            cp.cuda.runtime.deviceSynchronize(); t = time()
            for i in range(ct):
                func((Nblk,), (32, Nwarp), tuple(args))
                #func((Nblk,), (32, Nwarp), (cp.int32(N), cp.int32(L), cp.int32(B), lens_d, shifts_d, 
                #                            p_d, dp_d, cp.int32(ldp), s_d, cp.int32(lds), 
                #                            u_out_d, dJdu_out_d, u_in_d, dJdu_in_d, cp.int32(ldu)) + args_post)
            cp.cuda.runtime.deviceSynchronize(); t = time() - t; ct *= 2
        return t / (ct/2)

    flops1 = np.zeros([8, 32])*np.nan; flops2 = np.zeros([8, 32])*np.nan
    wsList = np.zeros([2, 8], dtype=np.int)

    print ("Square Matrices: N x N x N")
    Nlist = [64, 128, 192, 256, 320, 384, 512, 640]
    for (i, N) in enumerate(Nlist):
        print(f"N = {N:4d}: ", end="")
        for (j, ws) in enumerate(range(1, 33)):
            try:
                t = timetest(N, N, N, ws); 
            except CUDADriverError as e:
                print ("x" + " "*(31-j), end=""); break
            flops1[i, j] = 3 * (32 * (N*N*N/2) / t) / 1e9; print (".", end="", flush=True)
        j = np.argmax(np.nan_to_num(flops1[i])); wsList[0, i] = j+1
        print(f" {flops1[i,j]:6.1f} GFLOP/s [{j+1:2d}*32={32*(j+1):4d} threads]")
    print ("Fat Matrices: N x N x 4096")
    for (i, N) in enumerate(Nlist):
        print(f"N = {N:4d}: ", end="")
        for (j, ws) in enumerate(range(1, 33)):
            try:
                t = timetest(N, N, 4096, ws); 
            except CUDADriverError as e:
                print ("x" + " "*(31-j), end=""); break
            flops2[i, j] = 3 * (32 * (N*N*4096/2) / t) / 1e9; print (".", end="", flush=True)
        j = np.argmax(np.nan_to_num(flops2[i])); wsList[1, i] = j+1
        print(f" {flops2[i,j]:6.1f} GFLOP/s [{j+1:2d}*32={32*(j+1):4d} threads]")
    print()
        
    np.savetxt("tuned_warpsize.txt", np.concatenate([np.array([Nlist]), wsList], axis=0).T, delimiter='\t', fmt='%d')

    return
    
    (f, (ax1, ax2)) = plt.subplots(1, 2, figsize=(9, 4.5), sharex=True, sharey=True)
    (flops_mesh1, flops_mesh2) = ([], [])
    for (ax, flops_i, flops_max_i) in zip([ax1, ax2], [flops1, flops2], [flops_mesh1, flops_mesh2]):
        for (j, flops_ij) in enumerate(flops_i):
            ax.plot(range(1, 33), flops_ij, '.-')
        ax.plot([np.nan], 'o', mec='k', mfc='w')
        for (j, flops_ij) in enumerate(flops_i):
            k = np.argmax(np.nan_to_num(flops_ij))
            ax.plot([k+1], [flops_ij[k]], 'o', mec='C'+str(j), mfc='w')
            flops_max_i.append(flops_ij[k])
        ax.set_xlim(-1, 33); ax.set_ylim(-20, 900); ax.grid()
    ax2.legend(["N = " + str(Nlist[0])] + Nlist[1:], loc=4, ncol=2, framealpha=1)

    ax1.set_xlabel(r"# Warps"); ax2.set_xlabel(r"# Warps"); ax1.set_ylabel("K40 Perf (GFLOP/s)")
    ax1.set_title("N x N x N"); ax2.set_title("N x N x 4096")
    plt.tight_layout()
    #plt.savefig("test-fig1.pdf", format="pdf")


def test_cpu_speed():
    import meshes as ms
    print ("Speed Test: meshes/mesh.py (NumPy)")
    print ("--------------------------------------")
    def time_clem(N, B):
        U = np.random.randn(N, B).astype(np.complex)
        V = np.random.randn(N, B).astype(np.complex)
        clem = ms.ClementsNetwork(N=N)
        clem.p_splitter = np.random.randn(N*(N-1)//2, 2)
        clem.p_phase[:] = np.random.randn(N**2)

        t = time(); n = 0
        while (time() - t < 0.2):
            clem.dot(U); n += 1
        t_inf = (time() - t)/n
        t = time(); n = 0
        while (time() - t < 0.2):
            clem.grad_phi(U, V); n += 1
        t_bd = (time() - t)/n - t_inf

        flops_inf = 32 * (N*N*N/2); flops_bd = 3*flops_inf

        return (t_inf, t_bd, flops_inf/t_inf/1e9, flops_bd/t_bd/1e9)

    print ("Square Matrices: N x N x N")
    for N in [64, 128, 192, 256, 320, 384, 512, 640]:
        (t_inf, t_bd, flops_inf, flops_bd) = time_clem(N, N)
        print (f"N = {N:3d}:   {flops_inf:.3f} GFLOP/s (fwdprop, t={t_inf:.1e}s) | " + 
               f"{flops_bd:.3f} GFLOP/s (backprop, t={t_inf:.1e}s)")    

        
# Options
if inf:
    if acc:   test_fwd_acc(False, mode)
    if speed: test_fwd_speed(False, mode)  
if fd:
    if acc:   test_fwd_acc(True, mode)
    if speed: test_fwd_speed(True, mode)  
if bd:
    if acc:   test_rev_acc(mode)
    if speed: test_rev_speed(mode)
if cpu:
    assert mode == 'mzi'
    if acc:   raise NotImplementedError()
    if speed: test_cpu_speed()