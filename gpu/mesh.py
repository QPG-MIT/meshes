# meshes/gpu/mesh.py
# Ryan Hamerly, 4/13/21
#
# Implements MeshNetworkGPU class for interfacing with the GPU code.
#
# History
#   04/13/21: Created this file.

import numpy as np
import cupy as cp
from typing import List, Tuple, Callable
from ..mesh import MeshNetwork

def _mat(val, shape, check_order=None, **args):
    if val is None:
        return cp.empty(shape, **args)
    else:
        if check_order: assert (cp.isfortran(val) == (check_order=='F'))
        return cp.asarray(val, **args)

class MeshNetworkGPU(MeshNetwork):
    _N: int
    _L: int
    lens: cp.ndarray
    shifts: cp.ndarray
    X: str
    X_ns: int
    X_np: int
    _fwdprop: cp.RawKernel
    _fwddiff: cp.RawKernel
    _backdiff: cp.RawKernel
    _nwarp_fwdprop: List[int]
    _nwarp_fwddiff: List[int]
    _nwarp_backdiff: List[int]
    phi_pos: str
    is_phase: bool
    dtype: cp.dtype

    def __init__(self,
                 N: int,
                 L: int,
                 lens: np.ndarray,
                 shifts: np.ndarray,
                 p_phase: np.ndarray,
                 p_splitter: np.ndarray,
                 X: str,
                 phi_pos='out',
                 is_phase=True):
        r"""
        Manages the beamsplitters for a Clements beamsplitter mesh.
        :param N: Number of inputs / outputs.
        :param lens: List of number of MZM's in each column.
        :param shifts: List of the shifts for each column.
        :param p_phase: Parameters [phi_i] for phase shifters.
        :param p_splitter: Parameters [beta_i - pi/4] for beam splitters.
        :param p_crossing: Crossing parameterrs (theta, phi).  Takes place of p_phase.
        :param phi_out: External phase screen.  Takes place of p_phase.
        :param perm: List of permutations, if any.
        :param X: Crossing type.
        :param phi_pos: Position of external phase screen.
        :param is_phase: Whether to include the phase screen.  If not, mesh will not be universal.
        """
        from . import mod, nwarps_opt
        
        assert not is_phase     # TODO -- generalize.
        assert X in ['mzi', 'sym', 'orth']
        assert N%2 == 0
        self._N = N
        self._L = L
        self.X = X
        self.phi_pos = phi_pos
        self.is_phase = is_phase
        self.dtype = {'mzi': cp.complex64, 'sym': cp.complex64, 'orth': cp.float32}[self.X]
        self.X_ns = {'mzi': 2, 'sym': 1, 'orth': 0}[self.X]
        self.X_np = {'mzi': 2, 'sym': 2, 'orth': 1}[self.X]
        
        assert p_phase.size == self.n_phase
        assert p_splitter.size in [self.n_splitter, 0]
        self.p_phase = cp.asarray(p_phase.flatten(), dtype=cp.float32, order="C")
        self.p_splitter = (cp.asarray(p_splitter.reshape([L, N//2, self.X_ns]), dtype=cp.float32, order="C")
                           if p_splitter.size else cp.empty((L, N//2, 0), dtype=cp.float32, order="C"))
        
        assert (N <= 640)    # Can't make larger meshes than this right now.
        N0 = (N-1)//64 + 64
        print (f"fwdprop_N{N0}_{X}")
        self._fwdprop  = mod.get_function(f"fwdprop_N{N0}_{X}")
        self._fwddiff  = mod.get_function(f"fwddiff_N{N0}_{X}")
        self._backdiff = mod.get_function(f"backdiff_N{N0}_{X}")
        self._nwarp_fwdprop  = [nwarps_opt['fwdprop',  X, sh, N0] for sh in ['sq', 'fat']]
        self._nwarp_fwddiff  = [nwarps_opt['fwddiff',  X, sh, N0] for sh in ['sq', 'fat']]
        self._nwarp_backdiff = [nwarps_opt['backdiff', X, sh, N0] for sh in ['sq', 'fat']]
        
        self.lens = cp.array(lens, dtype=cp.int32)
        self.shifts = cp.array(shifts, dtype=cp.int32)

    @property
    def L(self):
        return self._L
    @property
    def M(self):
        return self._N
    @property
    def N(self):
        return self._N
    @property
    def n_splitter(self):
        return self.n_cr * self.X_ns
    @property
    def n_phase(self):
        return self.n_cr * self.X_np + self.is_phase * self.N

    @property
    def n_cr(self):
        r"""
        Number of 2x2 crossings in the beamsplitter mesh (including blank crossings).
        :return: int
        """
        return self.L*(self.N//2)
    @property
    def p_crossing(self):
        r"""
        Returns the segment of self.p_phase that encodes the 2x2 crossing parameters.
        :return: array of size (L, N/2, n_phase)
        """
        return self.p_phase[:self.n_cr*X_np].reshape([L, N//2, self.X_np])
    @p_crossing.setter
    def p_crossing(self, p):
        self.p_crossing[:] = p
    @property
    def phi_out(self):
        r"""
        Returns the segment of self.p_phase that encodes the output phases.
        :return: array of size (self.N,)
        """
        self.p_phase[self.n_cr*self.X_np:]
    @phi_out.setter
    def phi_out(self, p):
        self.phi_out[:] = p
        
    def _defaults(self, p_phase, p_splitter):
        (p, s) = (self.p_phase if (p_phase==None) else cp.asarray(p_phase, dtype=cp.float32, order="C"),
                  self.p_splitter if (p_splitter==None) else cp.asarray(p_splitter, dtype=cp.float32, order="C"))
        assert (p.size == self.n_phase) and (s.size == self.n_splitter); return (p, s)

    def matrix(self, p_phase=None, p_splitter=None) -> np.ndarray:
        return self.dot(cp.eye(self.N, dtype=self.dtype), p_phase, p_splitter)
        
    def dot(self, v, p_phase=None, p_splitter=None, dp=None, dv=None, out=None, dout=None) -> cp.ndarray:
        # Prepare data.
        (argsf, argsf_f32, argsc, argsc_f32) = [dict(dtype=t, order=o) for o in "FC" for t in [self.dtype, cp.float32]]
        v = cp.asarray(v, **argsf); 
        out = _mat(out, v.shape, "F", **argsf); 
        (p, s) = self._defaults(p_phase, p_splitter)
        assert (v.ndim == 2) and (v.shape[0] == self.N) and (out.shape == v.shape)
        (N, L, B, ldp, lds, ldu) = map(cp.int32, [self.N, self.L, v.shape[1], 
                                                  self.N//2*self.X_np, self.N//2*self.X_ns, self.N])
         
        # Calls fwdprop or fwddiff, depending on whether derivatives are specified.
        if (dp is None) and (dv is None):
            # Call fwdprop().
            args = [N, L, B, self.lens, self.shifts, p, ldp, s, lds, v, out, ldu, cp.int32(0)]
            Nwarp = self._nwarp_fwdprop[int(B > 2*N)]
            Nblk = int(np.ceil(B/Nwarp))
            self._fwdprop((Nblk,), (32, Nwarp), tuple(args))
            return out
        else:
            # Prepare derivatives.
            dout = _mat(dout, v.shape, "F", **argsf); dp = _mat(dp, (0,), **argsc_f32); dv = _mat(dv, (0,), **argsc_f32);
            assert (dout.shape == v.shape) and (dp.shape in [(0,), p.shape]) and (dv.shape in [(0,), v.shape])
            # Call fwddiff().
            args = [N, L, B, self.lens, self.shifts, p, dp, ldp, s, lds, v, dv, out, dout, ldu, cp.bool(0)]
            Nwarp = self._nwarp_fwddiff[int(B > 2*N)]
            Nblk = int(np.ceil(B/Nwarp))
            self._fwddiff((Nblk,), (32, Nwarp), tuple(args))
            return (out, dout)

    def _L2norm_fn(self, J):
        def f(U):
            V = U - cp.eye(self.N, dtype=self.dtype)
            if not self.is_phase: V.reshape([N*N])[::N+1] = 0
            J[0] = cp.linalg.norm(V)**2; return 2*V
        return f

    def L2norm(self, U_target, p_phase=None, p_splitter=None) -> float:
        r"""
        Returns the L2 norm |U_target^dag * U - 1|^2.  If is_phase=False, only takes norm over off-diagonal elements.
        :param U_target: Target matrix, size (M, N)
        :param p_phase: Phase parameters (phi_k).  Scalar or vector of size N^2.
        :param p_splitter: Beamsplitter parameters (beta_k - pi/4).  Scalar of vector of size N(N-1).
        :return:
        """
        J = [0]; f = self._L2norm_fn(J); 
        f(self.dot(cp.asarray(U_target, dtype=self.dtype).T.conj(), p_phase, p_splitter))
        return J[0]

    def grad_phi_target(self, U_target, p_phase=None, p_splitter=None) -> Tuple[float, np.ndarray]:
        r"""
        Gets the gradient of the L2 norm |U_target^dag * U - 1|^2 with respect to the phase parameters.  If is_phase=False
        (no phase screen), only takes the norm over off-diagonal elements (phase screen can correct for the diagonal).
        :param U_target: Target matrix, size (M, N)
        :param p_phase: Phase parameters (phi_k).  Scalar or vector of size N^2.
        :param p_splitter: Beamsplitter parameters (beta_k - pi/4).  Scalar of vector of size N(N-1).
        :return:
        """
        J = [0]; f = self._L2norm_fn(J)
        grad = np.real(self.grad_phi(cp.asarray(U_target, dtype=self.dtype).T.conj(), f, p_phase, p_splitter)[0])
        return (J[0], grad)

    def grad_phi(self, v, dJdv, p_phase=None, p_splitter=None, out_dp=None, out_dJdv=None, vpos='in'):
        assert not self.is_phase   # TODO -- handle output phases.
        assert vpos in ['in', 'out']
        (argsf, argsf_f32, argsc, argsc_f32) = [dict(dtype=t, order=o) for o in "FC" for t in [self.dtype, cp.float32]]
        (p, s) = self._defaults(p_phase, p_splitter);
        v = _mat(v, v.shape, **argsf); 
        out_dJdv = _mat(out_dJdv, v.shape, 'F', **argsf); out_dp = _mat(out_dp, p.shape, 'C', **argsc_f32)
        (N, L, B, ldp, lds, ldu) = map(cp.int32, [self.N, self.L, v.shape[1], 
                                                  self.N//2*self.X_np, self.N//2*self.X_ns, self.N])

        # Forward propagate, if needed.
        if (vpos == 'in'):
            self.dot(v, p_phase, p_splitter, out=v)

        # Get gradient.
        if callable(dJdv):
            dJdv = dJdv(v)
        dJdv = _mat(dJdv, v.shape, **argsf)
            
        # Backward propagate, call backdiff().
        args = ([N, L, B, self.lens, self.shifts, p, out_dp, ldp, s, lds, v, dJdv, v, out_dJdv, ldu, cp.int32(0)])
        Nwarp = self._nwarp_backdiff[int(B > 2*N)]
        Nblk = int(np.ceil(B/Nwarp))
        self._backdiff((Nblk,), (32, Nwarp), tuple(args))

        # Return both weight updates and back-propagated gradients.
        return (out_dp, out_dJdv)

    def hvp(self, vec, v, w: Callable, p_phase=None, p_splitter=None, eps=1e-3) -> np.ndarray:
        r"""
        Hessian-vector product vec -> H*vec.  Computed by finite difference.
        :param vec: Vector in parameter space.
        :param v: Input matrix
        :param w: Function that computes gradient of objective J (in terms of output matrix).
        :param p_phase: Phase parameters (phi_k).
        :param p_splitter: Splitter parameters.
        :param eps: Finite-difference step amplitude.
        :return:
        """
        p_phase = self.p_phase if (p_phase is None) else p_phase

        return (self.grad_phi(v, w, p_phase + vec*eps, p_splitter)[0]
                - self.grad_phi(v, w, p_phase - vec*eps, p_splitter)[0])/(2*eps)

    def hvp_target(self, vec: np.ndarray, U_target, p_phase=None, p_splitter=None, eps=1e-3) -> np.ndarray:
        r"""
        Hessian-vector product vec -> H*vec for L2 norm |U - U_target|^2.
        :param vec: Vector in parameter space.
        :param U_target: Target unitary.
        :param p_phase: Phase parameters (phi_k).
        :param p_splitter: Splitter parameters.
        :param eps: Finite-difference step amplitude.
        :return:
        """
        J = [0]; w = self._L2norm_fn(J); v = U_target.T.conj()
        return self.hvp(vec, v, w, p_phase, p_splitter, eps)

    def hess(self, v, w: Callable, p_phase=None, p_splitter=None, eps=1e-3) -> np.ndarray:
        r"""
        Hessian.  Computed by finite difference.
        :param v: Input matrix
        :param w: Function that computes gradient of objective J (in terms of output matrix).
        :param p_phase: Phase parameters (phi_k).
        :param p_splitter: Splitter parameters.
        :param eps: Finite-difference step amplitude.
        :return:
        """
        vec = np.eye(self.n_phase)
        return np.array([self.hvp_target(vec_i, v, w, p_phase, p_splitter, eps) for vec_i in vec])

    def hess_target(self, U_target, p_phase=None, p_splitter=None, eps=1e-3) -> np.ndarray:
        r"""
        Hessian for the L2 norm |U - U_target|^2.
        :param U_target: Target unitary.
        :param p_phase: Phase parameters (phi_k).
        :param p_splitter: Splitter parameters.
        :param eps: Finite-difference step amplitude.
        :return:
        """
        vec = np.eye(self.n_phase)
        return np.array([self.hvp_target(vec_i, U_target, p_phase, p_splitter, eps) for vec_i in vec])

    def copy(self) -> 'StructuredMeshNetwork':
        r"""
        Returns a copy of the mesh.
        :return:
        """
        return StructuredMeshNetwork(self.N, self.lens.copy(), self.shifts.copy(), np.array(self.p_phase),
                                     np.array(self.p_splitter), perm=self.perm, is_phase=self.is_phase,
                                     X=self.X, phi_pos=self.phi_pos)
