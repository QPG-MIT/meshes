# meshes/gpu/test.py
# Ryan Hamerly, 4/3/21
#
# Testing utility for this package.  Tests both speed and accuracy.
#
# History:
#   04/03/21: Created this file.


import numpy as np
import cupy as cp
from time import time
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import sys
from cupy_backends.cuda.api.driver import CUDADriverError


print ("Loading module.")
mod = cp.RawModule(path="/home/rhamerly/PNP/meshprop.cubin")


# Step 1: Accuracy Test.
# TODO: make this more comprehensive, looking at a range of K, L, B, Nwarp.

(K, L, B) = (4, 20, 64); Nwarp = 30; Nblk = int(np.ceil(B/Nwarp))
print (f"Accuracy Test: N={32*K}, L={L}, B={B}, Nwarp={Nwarp}...", end="")
# Inputs.
p = np.random.randn(L, 32*K, 2).astype(np.float32)
s = np.random.randn(L, 32*K, 2).astype(np.float32)
ldp = lds = 2*(32*K); ldu = 2*(32*K)
u_in = np.random.randn(B, 32*2*K, 2).dot([1, 1j]).astype(np.complex64); u = u_in
# GPU code.
func = mod.get_function(f"fwdprop_N{64*K}")
p_d = cp.asarray(p); s_d = cp.asarray(s); in_d = cp.asarray(u_in); out_d = cp.asarray(u_in*0)
func((Nblk,), (32,Nwarp), (p_d, s_d, in_d, out_d, cp.int32(ldu), cp.int32(ldp), cp.int32(lds), cp.int32(L), cp.int32(B)))
u_out = out_d.get()
# CPU code for comparison.
def Tij_cpu(p, s):
    (theta, phi) = p.T; beta = s.T
    (Cp, Cm, C, Sp, Sm, S) = [fn(x) for fn in [np.cos, np.sin] for x in [beta[0]+beta[1], beta[0]-beta[1], theta/2]]
    return np.exp(1j*theta/2) * np.array([[np.exp(1j*phi) * (1j*S*Cm - C*Sp),    1j*C*Cp - S*Sm],
                                          [np.exp(1j*phi) * (1j*C*Cp + S*Sm),   -1j*S*Cm - C*Sp]])
for i in range(L):
    if (i % 2): M = block_diag(np.eye(1), *Tij_cpu(p[i], s[i]).transpose(2, 0, 1)[:-1], np.eye(1))
    else:       M = block_diag(*Tij_cpu(p[i], s[i]).transpose(2, 0, 1))
    u = u.dot(M.T)
# Error evaluation.
err = np.linalg.norm(u_out-u, axis=1) / np.linalg.norm(u, axis=1)
if ((err < 1e-4).all()): print("Success.")
else: print(f"FAIL!  {(err < 1e-4).sum()}/{len(err)} had relative error > 1e-4."); print("err = ", err)


    
# Step 2: Speed Test.
# Performance is a function of mesh size N, depth L, batch size B, and warps/block.  The latter
# is a tuning parameter that must be swept.
print ("Speed Test: N = 64, ..., 1024.  Configurations: (N x N x N), (N x N x 4096).")
def timetest(N, L, B, Nwarp):
    K = N//32
    Nblk = int(np.ceil(B/Nwarp))
    func = mod.get_function(f"fwdprop_N{N}")
    p_d = cp.random.randn(L, 32*K, 2, dtype=np.float32); s_d = cp.random.randn(L, 32*K, 2, dtype=np.float32)
    in_d = cp.random.randn(B, 32*2*K, 2, dtype=np.float32).dot(cp.asarray([1.0, 1.0j], dtype=np.complex64))
    out_d = cp.zeros([B, 32*2*K], dtype=np.complex64)
    ldp = 2*(32*K); ldu = 2*(32*K)
    t = 0; ct = 1
    while (t < 1e-2):
        cp.cuda.runtime.deviceSynchronize(); t = time()
        for i in range(ct):
            func((Nblk,), (32,Nwarp), (p_d, s_d, in_d, out_d, cp.int32(ldu), cp.int32(ldp), 
                                       cp.int32(ldp), cp.int32(L), cp.int32(B)))
        cp.cuda.runtime.deviceSynchronize(); t = time() - t; ct *= 2
    return t / (ct/2)

flops1 = np.zeros([8, 32])*np.nan; flops2 = np.zeros([8, 32])*np.nan
wsList = np.zeros([2, 8], dtype=np.int)

print ("FwdProp Test: N x N x N")
Nlist = [64, 128, 192, 256, 384, 512, 640, 1024]
for (i, N) in enumerate(Nlist):
    print(f"N = {N:4d}: ", end="")
    for (j, ws) in enumerate(range(1, 33)):
        try:
            t = timetest(N, N, N, ws); 
        except CUDADriverError as e:
            print ("x" + " "*(31-j), end=""); break
        flops1[i, j] = (32 * (N*N*N/2) / t) / 1e9; print (".", end="", flush=True)
    j = np.argmax(np.nan_to_num(flops1[i])); wsList[0, i] = j+1
    print(f" {flops1[i,j]:6.1f} GFLOP/s [{j+1:2d}*32={32*(j+1):4d} threads]")
print ("FwdProp Test: N x N x 4096")
for (i, N) in enumerate(Nlist):
    print(f"N = {N:4d}: ", end="")
    for (j, ws) in enumerate(range(1, 33)):
        try:
            t = timetest(N, N, 4096, ws); 
        except CUDADriverError as e:
            print ("x" + " "*(31-j), end=""); break
        flops2[i, j] = (32 * (N*N*4096/2) / t) / 1e9; print (".", end="", flush=True)
    j = np.argmax(np.nan_to_num(flops2[i])); wsList[1, i] = j+1
    print(f" {flops2[i,j]:6.1f} GFLOP/s [{j+1:2d}*32={32*(j+1):4d} threads]")
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