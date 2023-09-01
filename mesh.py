# meshes/mesh.py
# Ryan Hamerly, 4/27/22
#
# Implements the MeshNetwork class, which can be used for describing Reck and Clements meshes (or any other
# beamsplitter mesh where couplings occur between neighboring waveguides).  Clements decomposition is also
# implemented.
#
# History
#   06/19/20: Made Reck / Clements code object-oriented using the MeshNetwork class (meshes.py).
#   07/09/20: Turned module into package, renamed file mesh.py.
#   07/10/20: Added compatibility for custom crossings in crossing.py.
#   12/15/20: Added Ratio Method tuning triangular meshes.
#   12/19/20: Added Direct Method for tuning triangular meshes.
#   03/29/21: Added utility to convert between crossing types.  Tweaks to gradient function.  Hessian support.
#   04/12/21: Forward differentiation support.
#   04/27/22: JIT'ed mesh.dot() to speed up matrix-vector multiplication.
#   07/25/22: JIT'ed mesh.dot_vjp() (formerly grad_phi()) to speed up VJP, streamline JAX differentiation.
#   05/30/23: Added ClippedNetwork class and support for indexing a MeshNetwork class.
#   09/01/23: Converted meshdot_helper to C code and added OpenMP support, runs up to 20x faster.


import numpy as np
import warnings
from typing import List, Any, Tuple, Callable
from numba import njit
from ctypes import c_int, c_float, c_double, c_void_p, CDLL
from .crossing import Crossing, MZICrossing, MZICrossingOutPhase, SymCrossing

@njit
def meshdot_helper_nb(T, dT, ph, dph, v, dv, inds, lens, shifts, perms, pos, mode):
    is_fd = (mode == 1); is_bd = (mode == 2); tr = int(is_bd); #is_diff = (dT.size != 0);
    if is_bd: pos = 1-pos; T[:] = T.conj(); ph[:] = -ph;
    is_perm = (perms.size != 0); L = len(lens); (i, j) = (tr, 1-tr)
    (n1, n2, dn) = (L-1, -1, -1) if (mode == 2) else (0, L, 1)

    if (pos == 0):
        T_ph = np.exp(1j*ph).reshape((len(ph),1)); v[:] *= T_ph
        if is_fd: dv[:] = dv*T_ph + 1j*v*dph.reshape((len(ph),1))
        if is_bd: dv[:] = dv*T_ph; dph[:] = np.real(-1j*dv*v.conj()).sum(-1)
    for n in range(n1, n2, dn):
        i1 = inds[n]; l = lens[n]; s = shifts[n]; i2 = i1+l
        v1 = v[s:s+2*l:2]; v2 = v[s+1:s+2*l:2]
        T00 = T[0,0,i1:i2].reshape((l,1)); T01 = T[i,j,i1:i2].reshape((l,1))
        T10 = T[j,i,i1:i2].reshape((l,1)); T11 = T[1,1,i1:i2].reshape((l,1))
        if is_perm:
            p = perms[n + int(is_bd)]; v[:] = v[p]
            if is_fd or is_bd: dv[:] = dv[p]
        if is_fd:
            dv1 = dv[s:s+2*l:2]; dv2 = dv[s+1:s+2*l:2]
            dT00 = dT[0,0,i1:i2].reshape((l,1)); dT01 = dT[i,j,i1:i2].reshape((l,1))
            dT10 = dT[j,i,i1:i2].reshape((l,1)); dT11 = dT[1,1,i1:i2].reshape((l,1))
            # dv -> dT*v + T*dv
            temp   = dT00*v1 + dT01*v2 + T00*dv1 + T01*dv2
            dv2[:] = dT10*v1 + dT11*v2 + T10*dv1 + T11*dv2
            dv1[:] = temp
        # v -> T*v
        temp  = T00*v1 + T01*v2
        v2[:] = T10*v1 + T11*v2
        v1[:] = temp
        if is_bd:
            dv1 = dv[s:s+2*l:2]; dv2 = dv[s+1:s+2*l:2]
            # dJ/dT_ij = (dJ/dy_i)* x_j
            dT[0,0,i1:i2] = (dv1.conj()*v1).sum(-1); dT[0,1,i1:i2] = (dv1.conj()*v2).sum(-1)
            dT[1,0,i1:i2] = (dv2.conj()*v1).sum(-1); dT[1,1,i1:i2] = (dv2.conj()*v2).sum(-1)
            # dv -> T*dv
            temp   = T00*dv1 + T01*dv2
            dv2[:] = T10*dv1 + T11*dv2
            dv1[:] = temp
    if is_perm:
        p = perms[0 if is_bd else L]; v[:] = v[p]
        if is_fd or is_bd: dv[:] = dv[p]
    if (pos == 1):
        T_ph = np.exp(1j*ph).reshape((len(ph),1)); v[:] *= T_ph
        if is_fd: dv[:] = dv*T_ph + 1j*v*dph.reshape((len(ph),1))
        if is_bd: dv[:] = dv*T_ph; dph[:] = np.real(-1j*dv*v.conj()).sum(-1)
    if is_bd: ph[:] = -ph

OMP_THREADS = 'auto'

try:
    dot_c = CDLL("/".join(__file__.split("/")[:-1] + ["c", "dot.so"]))
    meshdot_helper32 = dot_c.meshdot_helper32; meshdot_helper64 = dot_c.meshdot_helper64
    for h in [meshdot_helper32, meshdot_helper64]:
        h.argtypes = (c_int, c_int, c_int, c_int,
                      c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
                      c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int)
    HAS_C_CODE = True
    BACKEND = "c"
except OSError:
    print ("C code in meshes/c not compiled (run makefile to compile), falling back to Numba.")
    HAS_C_CODE = False
    BACKEND = "numba"

def meshdot_helper_c(T, dT, ph, dph, v, dv, inds, lens, shifts, perms, pos, mode):
    is64 = np.any([X.dtype == np.complex128 for X in [T, dT, ph, dph, v, dv]])
    N = v.shape[0]; B = v.size//N
    (ftype, ctype, fn) = [(np.float32, np.complex64, meshdot_helper32),
                          (np.float64, np.complex128, meshdot_helper64)][int(is64)]
    (T_, dT_, v_, dv_) = (np.asarray(X, dtype=ctype, order="C") for X in [T, dT, v, dv])
    (ph_, dph_) = (np.asarray(X, dtype=ftype, order="C") for X in [ph, dph])
    (inds_, lens_, shifts_, perms_) = (np.asarray(X, dtype=np.int32) for X in [inds, lens, shifts, perms])

    # Hack that works on a Mac :)
    threads = max(1, int(np.round(np.sqrt(N*B)/64, 0))) if OMP_THREADS == 'auto' else OMP_THREADS

    fn(c_int(N), c_int(B), c_int(T.shape[2]), c_int(len(lens)),
       T_.ctypes, dT_.ctypes, ph_.ctypes, dph_.ctypes, v_.ctypes, dv_.ctypes, inds_.ctypes, lens_.ctypes,
       shifts_.ctypes, perms_.ctypes, c_int(perms.size), c_int(pos), c_int(mode), c_int(threads))
    for (X_, X) in zip([T_, dT_, ph_, dph_, v_, dv_], [T, dT, ph, dph, v, dv]):
        if (X_.ctypes.data != X.ctypes.data): X[:] = X_

meshdot_helper = {"c": meshdot_helper_c, "numba": meshdot_helper_nb}


# The base class MeshNetwork.

class MeshNetwork:
    p_phase: np.array
    p_splitter: np.array

    @property
    def L(self) -> int:
        r"""
        Number of layers in the mesh network.
        """
        raise NotImplementedError()
    @property
    def M(self) -> int:
        r"""
        Output dimension of the mesh network.  Thus, self.matrix().shape == (self.M, self.N)
        """
        raise NotImplementedError()
    @property
    def N(self) -> int:
        r"""
        Output dimension of the mesh network.  Thus, self.matrix().shape == (self.M, self.N)
        """
        raise NotImplementedError()
    @property
    def n_phase(self) -> int:
        r"""
        Number of phase degrees of freedom (due to crossings plus phase screen)
        """
        return len(p_phase)
    @property
    def shape(self) -> Tuple[int, int]:
        r"""
        Dimension of the mesh network, same as self.matrix().shape
        :return:
        """
        return (self.M, self.N)
    def dot(self, v, p_phase=None, p_splitter=None) -> np.ndarray:
        r"""
        Computes the dot product between the splitter and a vector v.
        :param v: Input vector / matrix.
        :param p_phase: Phase parameters.  Defaults to stored values.
        :param p_splitter: Splitter angle parameters (deviation from pi/4).  Defaults to stored values.
        :return: Output vector / matrix.
        """
        raise NotImplementedError()
    def dot_vjp(self, v, dJdv, p_phase=None, p_splitter=None, p_crossing=None, phi_out=None) \
            -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Performs the vector-Jacobi product of the dot() function.
        :param v: Output field
        :param dJdv: Gradient dJ/dv*
        :param p_phase: Phase parameters (phi_k), dim=(N^2)
        :param p_splitter: Beamsplitter parameters.
        :param p_crossing: Crossing parameters, dim=(N(N-1)/2, 2)
        :param phi_out: Phase screen, dim=(N)
        :return: A tuple (dJ/dp, dJ/dv*)
        """
        raise NotImplementedError()
    def matrix(self, p_phase=None, p_splitter=None) -> np.ndarray:
        r"""
        Computes the input-output matrix.  Equivalent to self.dot(np.eye(self.N))
        :param p_phase: Phase parameters.  Defaults to stored values.
        :param p_splitter: Splitter angle parameters (deviation from pi/4).  Defaults to stored values.
        :return: NxN matrix.
        """
        return self.dot(np.eye(self.N), p_phase, p_splitter)

    def L2norm(self, U_target, p_phase=None, p_splitter=None) -> float:
        r"""
        Returns the L2 norm |U_target^dag * U - 1|^2.  If is_phase=False, only takes norm over off-diagonal elements.
        :param U_target: Target matrix, size (M, N)
        :param p_phase: Phase parameters (phi_k).  Scalar or vector of size N^2.
        :param p_splitter: Beamsplitter parameters (beta_k - pi/4).  Scalar of vector of size N(N-1).
        :return:
        """
        J = [0]; f = self._L2norm_fn(J); f(self.dot(U_target.T.conj(), p_phase=p_phase, p_splitter=p_splitter))
        return J[0]
    def grad_L2(self, U_target, p_phase=None, p_splitter=None) -> Tuple[float, np.ndarray]:
        r"""
        Gets the gradient of the L2 norm |U_target^dag * U - 1|^2 with respect to the phase parameters.  If is_phase=False
        (no phase screen), only takes the norm over off-diagonal elements (phase screen can correct for the diagonal).
        :param U_target: Target matrix, size (M, N)
        :param p_phase: Phase parameters (phi_k).  Scalar or vector of size N^2.
        :param p_splitter: Beamsplitter parameters (beta_k - pi/4).  Scalar of vector of size N(N-1).
        :return:
        """
        J = [0]; f = self._L2norm_fn(J)
        v = self.dot(U_target.T.conj()); dJdv = f(v)
        dJdp = self.dot_vjp(v, dJdv, p_phase, p_splitter)[0]
        return (J[0], dJdp)

    # These functions are deprecated and will soon be removed (use dot_vjp and meshes.jax routines instead).
    def grad_phi(self, v, w, p_phase=None, p_splitter=None) -> np.ndarray:
        warnings.warn("Function grad_phi() is deprecated, use dot_vjp() or import meshes.jax instead.")
        Mv = self.dot(v)
        return self.dot_vjp(Mv, w, p_phase=p_phase, p_splitter=p_splitter)
    def grad_phi_target(self, U_target, p_phase=None, p_splitter=None) -> Tuple[float, np.ndarray]:
        warnings.warn("Function grad_phi_target() is deprecated, use grad_L2() instead.")
        return self.grad_L2(U_target, p_phase, p_splitter)

    def __getitem__(self, item) -> 'ClippedNetwork':
        if (type(item) == tuple):
            if len(item) != 2:
                raise IndexError(f"Too many indices: {len(item)}, should be 1 or 2.")
            (idx_out, idx_in) = item
        else:
            (idx_out, idx_in) = (item, slice(None, None, None))
        if type(idx_in ) != slice: idx_in  = np.asarray(idx_in )
        if type(idx_out) != slice: idx_out = np.asarray(idx_out)
        return ClippedNetwork(self, idx_in, idx_out)


class StructuredMeshNetwork(MeshNetwork):
    _N: int
    inds: List[int]
    lens: List[int]
    shifts: List[int]
    X: Crossing
    phi_pos: str
    perm: List
    is_phase: bool

    def __init__(self,
                 N: int,
                 lens: List[int],
                 shifts: List[int],
                 p_phase: Any=0.,
                 p_splitter: Any=0.,
                 p_crossing=None,
                 phi_out=None,
                 perm=None,
                 X: Crossing=MZICrossing(),
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
        self.X = X
        self._N = N
        assert phi_pos in ['in', 'out']
        self.lens = lens; n_cr = sum(lens)
        self.shifts = shifts
        self.inds = np.cumsum([0] + list(lens)).tolist()
        self.p_phase = p_phase * np.ones(n_cr*X.n_phase + N*is_phase, dtype=np.float)
        self.p_splitter = np.array(p_splitter)
        self.phi_pos = phi_pos
        self.is_phase = is_phase
        self.perm = ([None]*(self.L+1) if (perm is None) else perm); assert len(self.perm) == self.L+1
        if not (p_crossing is None): self.p_crossing[:] = p_crossing
        if not (phi_out is None): self.phi_out[:] = phi_out
        assert len(shifts) == len(lens)
        assert self.p_phase.shape in [(), (n_cr*X.n_phase + self.N*is_phase,)]
        assert self.p_splitter.shape in [(), (n_cr, X.n_splitter)]

    @property
    def L(self):
        return len(self.lens)
    @property
    def M(self):
        return self._N
    @property
    def N(self):
        return self._N
    @property
    def n_phase(self) -> int:
        return self.n_cr*self.X.n_phase + self.is_phase*self.N
    @property
    def perm_r(self):
        return [None if (p is None) else np.argsort(p) for p in self.perm]

    @property
    def n_cr(self):
        r"""
        Number of 2x2 crossings in the beamsplitter mesh.
        :return: int
        """
        return sum(self.lens)
    @property
    def p_crossing(self):
        r"""
        Returns the segment of self.p_phase that encodes the 2x2 crossing parameters.
        :return: array of size (self.n_cr, self.X.n_phase)
        """
        return self._get_p_crossing(None)
    @p_crossing.setter
    def p_crossing(self, p):
        self.p_crossing[:] = p
    def _get_p_crossing(self, p_phase=None):
        return (self.p_phase if (p_phase is None)
                else p_phase)[:self.n_cr*self.X.n_phase].reshape([self.n_cr, self.X.n_phase])
    @property
    def phi_out(self):
        r"""
        Returns the segment of self.p_phase that encodes the output phases.
        :return: array of size (self.N,)
        """
        return self._get_phi_out(None)
    @phi_out.setter
    def phi_out(self, p):
        self.phi_out[:] = p
    def _get_phi_out(self, p_phase=None):
        return (self.p_phase if (p_phase is None) else p_phase)[self.n_cr*self.X.n_phase:]

    def _defaults(self, p_phase, p_splitter, p_crossing, phi_out):
        # Gets default values assuming certain inputs and current state of the mesh.
        assert (p_crossing is None) == (phi_out is None)
        p_splitter = (self.p_splitter if (p_splitter is None) else p_splitter) * np.ones([self.n_cr, self.X.n_splitter])
        if (p_crossing is not None):
            assert (p_crossing.shape == (self.n_cr, self.X.n_phase)) and (phi_out.shape == (self.N*self.is_phase,))
            return (p_crossing, phi_out, p_splitter)
        else:
            return (self._get_p_crossing(p_phase), self._get_phi_out(p_phase), p_splitter)

    def _dot_pars(self, p_phase, p_splitter, p_crossing, phi_out):
        (p_crossing, phi_out, p_splitter) = self._defaults(p_phase, p_splitter, p_crossing, phi_out)
        inds = np.array(self.inds); lens = np.array(self.lens); shifts = np.array(self.shifts)
        if np.all([x is None for x in self.perm]): perms = np.empty((0, 0), dtype=int)
        else: perms = np.array([(np.arange(self.N) if (perm_i is None) else perm_i) for perm_i in self.perm])
        pos = {'in': 0, 'out': 1}[self.phi_pos] if self.is_phase else -1
        return (p_crossing, phi_out, p_splitter, inds, lens, shifts, perms, pos)

    def dot(self, v, p_phase=None, p_splitter=None, p_crossing=None, phi_out=None, dp=None, dv=None):
        assert (v.shape[0] == self.N)
        v = np.array(v, complex, order="C"); sh = v.shape; sh_f = (sh[0], v.size//sh[0]); v = v.reshape(sh_f);
        (p_cr, ph, p_sp, inds, lens, shifts, perms, pos) = self._dot_pars(p_phase, p_splitter, p_crossing, phi_out)

        T = np.ascontiguousarray(self.X.T(p_cr, p_sp))
        if (dp is None) and (dv is None):
            # Propagate fields only
            dT = np.empty((0,0,0), complex); dv = np.empty((0,0), complex); dph = np.empty((0,), float)
            if self.L: meshdot_helper[BACKEND](T, dT, ph, dph, v, dv, inds, lens, shifts, perms, pos, 0) # <-- Fast!
            return v.reshape(sh)
        else:
            # Propagate fields and gradients
            dv = (v*0 if (dv is None) else np.array(dv, dtype=complex, order="C")); dv = dv.reshape(sh_f)
            dp = np.zeros([self.n_phase]) if (dp is None) else dp; dph = dp[self.n_cr*self.X.n_phase:]
            dp_cr = dp[:self.n_cr*self.X.n_phase].reshape([self.n_cr, self.X.n_phase])
            #dT = np.einsum('ijkl,li->jkl', self.X.dT(p_crossing, p_splitter), dp_cr)
            dT = np.einsum('ijkl,li->jkl', self.X.dT(p_cr, p_sp), dp_cr)
            if self.L: meshdot_helper[BACKEND](T, dT, ph, dph, v, dv, inds, lens, shifts, perms, pos, 1) # <-- Fast!
            return (v.reshape(sh), dv.reshape(sh))

    def _L2norm_fn(self, J):
        def f(U):
            V = U - np.eye(self.N, dtype=np.complex)
            if not self.is_phase: V -= np.diag(np.diag(V))
            J[0] = np.linalg.norm(V)**2; return 2*V
        return f

    def dot_vjp(self, v, dJdv, p_phase=None, p_splitter=None, p_crossing=None, phi_out=None) \
            -> Tuple[np.ndarray, np.ndarray]:
        (p_cr, ph, p_sp, inds, lens, shifts, perms, pos) = self._dot_pars(p_phase, p_splitter, p_crossing, phi_out)
        perms_bk = np.array([np.argsort(p) for p in perms]) if perms.size else perms

        sh = v.shape; sh_f = (sh[0], v.size//sh[0])
        v = np.array(v, complex, order="C").reshape(sh_f); dJdv = np.array(dJdv, complex, order="C").reshape(sh_f)
        dJdp = np.zeros([self.n_phase]); dJdph = dJdp[self.n_cr*self.X.n_phase:]
        dJdp_cr = dJdp[:self.n_cr*self.X.n_phase].reshape([self.n_cr, self.X.n_phase])

        T = np.ascontiguousarray(self.X.T(p_cr, p_sp)); dJdT = T*0
        if self.L: meshdot_helper[BACKEND](T, dJdT, ph, dJdph, v, dJdv, inds, lens, shifts, perms_bk, pos, 2) # <- Fast!
        dJdp_cr[:] = self.X.grad(p_cr, p_sp, gradT=dJdT.transpose(2, 0, 1))
        return (dJdp, dJdv)

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

    def flip_crossings(self, inplace=False, override=False) -> 'StructuredMeshNetwork':
        r"""
        Flips the mesh, i.e. from one with output phase shifters to input phase shifters and vice versa.  Only works
        for MZI meshes.
        :param inplace: Performs the operation inplace or returns a copy.
        :param override: Instead of throwing an error for non-MZI meshes, simply doesn't do the phase propagation
        (not needed in code e.g. when initializing a mesh before self-configuration).
        :return: The flipped StructuredMeshNetwork object.
        """
        if not inplace:
            self = self.copy()
        X = self.X; self.X = X.flip()

        if (self.phi_pos == 'out'):
            if isinstance(X, MZICrossing):
                for (m, len, shift, ind) in list(zip(range(self.L), self.lens, self.shifts, self.inds))[::-1]:
                    phi = self.p_crossing[ind:ind+len, 1]
                    (psi1, psi2) = self.phi_out[shift:shift+2*len].reshape([len, 2]).T
                    (phi[:], psi1[:], psi2[:]) = np.array([psi2-psi1, phi+psi1, psi1])
            else:
                if not override: raise NotImplementedError()
            self.phi_pos = 'in'
        else:
            if isinstance(X, MZICrossingOutPhase):
                for (m, len, shift, ind) in list(zip(range(self.L), self.lens, self.shifts, self.inds)):
                    phi = self.p_crossing[ind:ind+len, 1]
                    (psi1, psi2) = self.phi_out[shift:shift+2*len].reshape([len, 2]).T
                    (phi[:], psi1[:], psi2[:]) = np.array([psi1-psi2, psi2, phi+psi2])
            else:
                if not override: raise NotImplementedError()
            self.phi_pos = 'out'
        return self

    def flip(self, inplace=False) -> 'StructuredMeshNetwork':
        r"""
        Flips the entire network geometry about both position and time axes.  The resulting transfer matrix
        transforms as U -> U[::-1,::-1].T
        :param self:
        :param inplace: Perform the operation inplace.
        :return:
        """
        if not inplace:
            self = StructuredMeshNetwork(self.N, self.lens.copy(), self.shifts.copy(), np.array(self.p_phase),
                                         np.array(self.p_splitter), X=self.X, phi_pos=self.phi_pos)
        assert isinstance(self.X, MZICrossing) or isinstance(self.X, MZICrossingOutPhase)
        self.X = self.X.flip()
        self.phi_pos = {'in': 'out', 'out': 'in'}[self.phi_pos]
        self.p_splitter = (self.p_splitter + np.zeros([self.n_cr, 2]))[:, ::-1] + np.pi/2*np.array([[1, -1]])
        self.p_splitter = self.p_splitter[::-1]
        self.p_crossing = self.p_crossing[::-1]
        self.phi_out = self.phi_out[::-1]
        self.lens = self.lens[::-1]; self.shifts = (self.N-2*np.array(self.lens)-self.shifts[::-1]).tolist()
        self.inds = np.cumsum([0] + self.lens).tolist()

        return self

    @property
    def grid_phase(self) -> np.ndarray:
        r"""
        Arranges the data of self.p_crossing onto a 2D array.
        :return:
        """
        out = np.zeros([self.L, (self.N+1)//2, self.X.n_phase]) * np.nan
        for (i, j0, nj, ind) in zip(range(self.L), self.shifts, self.lens, self.inds):
            out[i, j0//2:j0//2+nj, :] = self.p_crossing[ind:ind+nj, :]
        return out

    @property
    def grid_splitter(self) -> np.ndarray:
        r"""
        Arranges the data of self.p_splitter onto a 2D array.
        :return:
        """
        if np.iterable(self.p_splitter):
            out = np.zeros([self.L, (self.N+1)//2, self.X.n_splitter]) * np.nan
            for (i, j0, nj, ind) in zip(range(self.L), self.shifts, self.lens, self.inds):
                out[i, j0//2:j0//2+nj, :] = self.p_splitter[ind:ind+nj, :]
            return out
        else:
            return np.zeros([self.L, 0])

    def convert(self,
                X: Crossing,
                p_splitter: np.ndarray=0.):
        r"""
        Converts the network to a different crossing type.
        :param X: New crossing type.
        :param p_splitter: Splitter imperfections (if any) in the new type.
        :return: A StructuredMeshNetwork object.
        """
        assert (np.all([p is None for p in self.perm]))  # TODO -- generalize
        phi_out = np.zeros([self.N])
        s_in = self.p_splitter + np.zeros([self.n_cr, self.X.n_splitter])
        p_in = self.p_crossing
        s_out = p_splitter + np.zeros([self.n_cr, X.n_splitter])
        p_out = np.zeros([self.n_cr, X.n_phase])

        # Convert each layer and propagate the phases to the right [left].  Merge with output [input] phase screen.
        if (self.phi_pos == 'out'):
            for (ind, L, s) in zip(self.inds, self.lens, self.shifts):
                (p_out[ind:ind+L], phi_out[s:s+2*L]) = self.X.convert(X, p_in[ind:ind+L], s_in[ind:ind+L], s_out[ind:ind+L],
                                                                      phi_out[s:s+2*L], self.phi_pos)
            phi_out += self.phi_out
        else:
            raise NotImplementedError()  # TODO -- generalize to 'in'

        return StructuredMeshNetwork(N=self.N, lens=self.lens, shifts=self.shifts, p_splitter=s_out, p_crossing=p_out,
            phi_out=phi_out if self.is_phase else None, perm=self.perm, X=X, phi_pos=self.phi_pos, is_phase=self.is_phase)
    
    def gpu(self):
        r"""
        Converts this network to a MeshNetworkGPU instance.
        """
        from .gpu import MeshNetworkGPU
        assert np.all([x == None for x in self.perm])
        
        p = np.nan_to_num(self.grid_phase.astype(np.float32))
        s = np.nan_to_num(self.grid_splitter.astype(np.float32))
        X = {MZICrossing: 'mzi', SymCrossing: 'sym'}[type(self.X)]
        return MeshNetworkGPU(self.N, self.L, np.array(self.lens), np.array(self.shifts),
                              p, s, X, phi_pos='out', is_phase=self.is_phase)


def calibrateTriangle(mesh: StructuredMeshNetwork, U, diag, method, warn=False):
    r"""
    A general Ratio Method to calibrate an arbitrarily shaped triangular mesh.
    :param mesh: Mesh to configure.
    :param U: Target matrix.
    :param diag: Direction of diagonals ['up' or 'down']
    :param method: Programming method ['direct' or 'ratio']
    :param warn: Warns the presence of errors.
    :return:
    """
    if (diag == 'up') and (mesh.phi_pos == 'out'):
        mesh.flip(True); calibrateTriangle(mesh, U, 'down', method, warn); mesh.flip(True); return
    assert (diag == 'down') and (mesh.phi_pos == 'in')
    X = {'direct': mesh.X.flip(), 'mod': mesh.X.flip(), 'ratio': mesh.X}[method]
    N = len(U); Upost = np.eye(N, dtype=np.complex); assert U.shape == (N, N)

    # Divide mesh into diagonals.  Make sure each diagonal can be uniquely targeted with a certain input waveguide.
    pos = np.concatenate([[[i, j, 2*j+j0, i-(2*j+j0)]
                           for j in range(mesh.lens[i])] for (i, j0) in enumerate(mesh.shifts)])
    pos = pos[np.lexsort(pos[:, ::3].T)][::-1]
    pos = np.split(pos, np.where(np.roll(pos[:,3], 1) != pos[:,3])[0])[1:]
    assert (np.diff(np.concatenate([pos_i[-1:] for pos_i in pos])[:, 2]) > 0).all()  # Diagonals have distinct inputs.
    if not isinstance(mesh.X, MZICrossingOutPhase):
        raise NotImplementedError("Only MZICrossingOutPhase supported for now.")
    p_splitter = mesh.p_splitter * np.ones([mesh.n_cr, 2])
    phi_out = 0*mesh.phi_out

    err = 0
    if method in ['direct', 'mod']:
        p = np.array([[2*j+j0, i+(2*j+j0)] for (i, j0) in enumerate(mesh.shifts) for j in range(mesh.lens[i])])
        p = p[np.lexsort(p.T)]; outfield = dict(p[np.where(np.roll(p[:, 1], 1) != p[:, 1])[0]][:, ::-1])  # out-fields
        X = mesh.X.flip(); assert ('T11' in X.tunable_indices);
        env = np.maximum.accumulate((np.array(mesh.shifts) + np.array(mesh.lens)*2 - 2)[::-1])[::-1]
        Psum = np.cumsum(np.abs(U[::-1]**2), axis=0)[::-1]  # Psum[i, j] = norm(U[i:, j])^2
        for pos_i in pos:
            E_in = 1.0; Tlist = []; ptr_last = None; l = pos_i[-1, 2]; #v = np.zeros([N], dtype=np.complex)
            w = np.zeros([N], dtype=np.complex)
            for (m, (i, j, ind, _)) in enumerate(pos_i[::-1]):  # Adjust MZIs *down* the diagonal one by one.
                ptr = mesh.inds[i] + j
                k = outfield[i+ind] if (i+ind in outfield) else ind  # Output index (or two) k:k+dk
                dk = (min(2, outfield[i+ind+2]-k) if (i+ind+2 in outfield) else 2) if (i < mesh.L-1) else 1
                # T11 = (U[k:k+dk, l] - Upost[k:k+dk, :].dot(v)).sum()/(E_in*Upost[k:k+dk, ind]).sum()
                U_kl = np.array(U[k:k+dk, l])
                if (method == 'mod'):
                    U_kl *= np.sqrt((np.linalg.norm(w[k:])**2 + np.abs(E_in)**2) / Psum[k, l])
                T11 = (U_kl - w[k:k+dk]).sum()/(E_in*Upost[k:k+dk, ind]).sum()
                ((theta, phi), d_err) = X.Tsolve(T11, 'T11', p_splitter[ptr]); err += d_err
                mesh.p_crossing[ptr, 0] = theta
                if m: mesh.p_crossing[ptr_last, 1] = phi  # Set theta for crossing & phi to *upper-left* of crossing.
                else: phi_out[ind] = phi
                T = X.T((theta, phi), p_splitter[ptr]); Tlist.append(T)
                # v[ind] += E_in * T[0, 0];
                w += E_in * T[0, 0] * Upost[:, ind]
                E_in *= T[1, 0]
                phi_out[ind+1] = np.angle(U[k, ind+1]/(Upost[k, ind]*T[0, 1]))
                ptr_last = ptr
            k = (ind+1) if (i == mesh.L-1 or ind > env[i+1]) else outfield[i+ind+2]
            dk = (min(2, outfield[i+ind+4]-k) if (i+ind+4 in outfield) else 2) if (i < mesh.L-1) else 1
            # Set final phase shift.
            # T11 = (U[k:k+dk,l] - Upost[k:k+dk,:].dot(v))/(E_in * Upost[k,ind+1])
            # mesh.p_crossing[ptr, 1] = phi = np.angle((U[k:k+dk,l] - Upost[k:k+dk,:].dot(v)).sum()/
            #                                          (E_in * Upost[k:k+dk,ind+1]).sum())
            T11 = (U[k:k+dk,l] - w[k:k+dk])/(E_in * Upost[k,ind+1])
            mesh.p_crossing[ptr,1] = phi = np.angle((U[k:k+dk,l] - w[k:k+dk]).sum()/ (E_in * Upost[k:k+dk,ind+1]).sum())
            Upost[:, ind+1] *= np.exp(1j*phi)
            for (ind, T) in zip(pos_i[:, 2], Tlist[::-1]):
                Upost[:, ind:ind+2] = Upost[:, ind:ind+2].dot(T)   # Multiply Upost by diagonal's T.

    elif method == 'ratio':
        X = mesh.X; assert ('T:1' in X.tunable_indices)
        for pos_i in pos:
            for (i, j, ind, _) in pos_i:  # Adjust MZIs *up* the diagonal one by one.
                ptr = mesh.inds[i] + j
                bc = Upost[:, ind:ind+2]  # Vectors [b, c]
                (T11, T21) =  bc.conj().T.dot(U[:, pos_i[-1][2]])   # Targets: (T11, T21) ~ (b* u_i, c* u_i)
                ((theta, phi), d_err) = X.Tsolve((T11, T21), 'T:1', p_splitter[ptr]); err += d_err
                mesh.p_crossing[ptr, :] = (theta, phi)  # Set theta for crossing & phi to *lower-right* of crossing.
                T = X.T((theta, phi), p_splitter[ptr])
                Upost[:, ind:ind+2] = Upost[:, ind:ind+2].dot(T)
            # Set input phases at the top of the diagonal.
            phi_out[ind:ind+2] = np.angle((Upost[:, ind:ind+2].conj() * U[:, ind:ind+2]).sum(0))

    mesh.phi_out[:] = phi_out
    if (warn and err):
        warnings.warn(
            "Mesh calibration: {:d}/{:d} matrix values could not be set.".format(err, len(mesh.p_crossing)))



class IdentityNetwork(StructuredMeshNetwork):
    def __init__(self, N: int, X: Crossing=MZICrossing(), phi_pos='out'):
        super(IdentityNetwork, self).__init__(N, [], [], 0., 0., X=X, phi_pos=phi_pos)


class ClippedNetwork(MeshNetwork):
    full: MeshNetwork
    _M: int
    _N: int
    idx_in: Any
    idx_out: Any

    @property
    def L(self) -> int:
        return self.full.L
    @property
    def M(self) -> int:
        return self._M
    @property
    def N(self) -> int:
        return self._N

    # These methods require a StructuredMeshNetwork base class, which is usually the case, so they're implemented here.
    @property
    def X(self) -> Crossing:
        assert isinstance(self.full, StructuredMeshNetwork)
        return self.full.X
    @property
    def p_crossing(self) -> np.ndarray:
        assert isinstance(self.full, StructuredMeshNetwork)
        return self.full.p_crossing
    @p_crossing.setter
    def p_crossing(self, x):
        assert isinstance(self.full, StructuredMeshNetwork)
        self.p_crossing[:] = x
    @property
    def phi_out(self) -> np.ndarray:
        assert isinstance(self.full, StructuredMeshNetwork)
        return self.full.phi_out[self.idx_out if (self.phi_pos == 'out') else self.idx_in]
    @phi_out.setter
    def phi_out(self, x):
        assert isinstance(self.full, StructuredMeshNetwork)
        self.full.phi_out[self.idx_out if (self.phi_pos == 'out') else self.idx_in] = x
    @property
    def phi_pos(self):
        assert isinstance(self.full, StructuredMeshNetwork)
        return self.full.phi_pos

    def __init__(self, full: MeshNetwork, idx_in, idx_out):
        r"""
        A MeshNetwork where only a fraction (idx_in, idx_out) of the inputs and outputs are accessible.  This class can
        be instantiated by indexing a MeshNetwork class, i.e. mesh[idx_out, idx_in].
        :param full: The mesh to be clipped.
        :param idx_in: [array | list | slice | None] Input ports
        :param idx_out: [array | list | slice | None] Output ports
        """
        self.full = full
        idx_in  = slice(None, None, None) if (idx_in  is None) else idx_in
        idx_out = slice(None, None, None) if (idx_out is None) else idx_out
        self._N = len(idx_in  if ('__len__' in type(idx_in ).__dict__) else np.arange(full.N)[idx_in ])
        self._M = len(idx_out if ('__len__' in type(idx_out).__dict__) else np.arange(full.M)[idx_out])
        self.idx_in  = idx_in
        self.idx_out = idx_out
        self.p_phase    = full.p_phase
        self.p_splitter = full.p_splitter

    def dot(self, v, p_phase=None, p_splitter=None) -> np.ndarray:
        if (type(self.idx_in) == slice and self.idx_in == slice(None)):
            V = v
        else:
            V = np.zeros((self.full.N,) + v.shape[1:])
            V[self.idx_in] = v
        return self.full.dot(V)[self.idx_out]
