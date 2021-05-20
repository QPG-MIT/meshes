# meshes/gpu/opt.py
# Ryan Hamerly, 4/14/21
#
# Implements MeshOptimizer class for interfacing with the GPU code.
#
# History
#   04/14/21: Created this file.

import numpy as np
import cupy as cp
from .mesh import MeshNetworkGPU
from typing import List
from scipy.optimize import minimize
from time import time

class MeshOptimizer:
    mesh: MeshNetworkGPU
    _Jcache: List
    p: cp.ndarray
    U: cp.ndarray
    yt: List
    t: List
    _inds: np.ndarray
    _t: List
    
    def __init__(self, mesh, U):
        is_phase_temp = mesh.is_phase
        mesh.is_phase = False; gpu = mesh.gpu(); mesh.is_phase = is_phase_temp
        self.mesh = gpu
        self.p = np.copy(mesh.p_crossing.flatten())
        self.U = cp.asarray(U, order="C", dtype=cp.complex64)
        self._Jcache = [1, 0]
        self._inds = np.concatenate([np.arange(i+j, i+j+k) for (i, j, k) in 
                                     zip(np.arange(mesh.L)*mesh.N//2, 
                                         np.array(mesh.shifts)//2, 
                                         np.array(mesh.lens))])

    def _to_gpu(self, p):
        out = cp.zeros([self.mesh.n_cr, self.mesh.X_np], dtype=cp.float32)
        out[self._inds] = p.reshape([p.size//self.mesh.X_np, self.mesh.X_np]); return out.reshape([out.size])
    def _from_gpu(self, p):
        out = p.reshape([self.mesh.n_cr, self.mesh.X_np])[self._inds].get()
        return out.reshape([out.size])
    def f(self, p):
        if (p.view(cp.int32)).sum() == self._Jcache[0]:
            return self._Jcache[1]
        else:
            out = self.mesh.dot(self.U.T.conj(), self._to_gpu(p))
            out.T.reshape([out.size])[::out.shape[0]+1] = 0
            return (cp.linalg.norm(out)**2).get()[()]
    def jac(self, p):
        (J, dp) = self.mesh.grad_phi_target(self.U, self._to_gpu(p))
        self._Jcache[0] = p.view(cp.int32).sum(); self._Jcache[1] = J.get()[()]
        return self._from_gpu(dp)
    def run(self, method, p=None, tol=1e-5, maxiter=10000, tick=-1):
        if (p is None): p = self.p
        self.yt = []; self.t = []; self._t = [time()]
        def f(p):
            out = self.f(p); self.yt.append(out); self.t.append(time() - self._t[0]); 
            if (len(self.yt) % tick == 0 and tick > 0): print (".", end="")
            return out
        def _f(x): return np.float64(f(x))
        def _jac(x): return self.jac(x).astype(np.float64)
        out = minimize(fun=_f, x0=p.astype(np.float64), jac=_jac, 
                       method=method, tol=tol, options={'maxiter': maxiter})
        if (tick > 0): print()
        self.yt = out.yt = np.sqrt(self.yt) / np.sqrt(len(self.U))
        self.t = out.t = np.array(self.t)
        return out