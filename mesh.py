# meshes/mesh.py
# Ryan Hamerly, 7/11/20
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


import numpy as np
import warnings
from typing import List, Any, Tuple, Callable
from .crossing import Crossing, MZICrossing, MZICrossingOutPhase


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
    def matrix(self, p_phase=None, p_splitter=None) -> np.ndarray:
        r"""
        Computes the input-output matrix.  Equivalent to self.dot(np.eye(self.N))
        :param p_phase: Phase parameters.  Defaults to stored values.
        :param p_splitter: Splitter angle parameters (deviation from pi/4).  Defaults to stored values.
        :return: NxN matrix.
        """
        return self.dot(np.eye(self.N), p_phase, p_splitter)
    def grad_phi(self, v, w, p_phase=None, p_splitter=None) -> np.ndarray:
        r"""
        Computes the gradient with respect to phase shifts phi.
        :param v: Input matrix / vector (normally np.eye(M)), size (M, ...)
        :param w: Output gradient matrix / vector dJ/d(U*), size (N, ...)
        :param p_phase: Phase parameters (phi_k).  Scalar or vector of size N^2.
        :param p_splitter: Beamsplitter parameters (beta_k - pi/4).  Scalar of vector of size N(N-1).
        :return:
        """
        raise NotImplementedError()
    def grad_phi_target(self, U_target, p_phase=None, p_splitter=None) -> Tuple[float, np.ndarray]:
        r"""
        Gets the gradient of the L2 norm |U - U_target|^2 with respect to the phase parameters.
        :param U_target: Target matrix, size (M, N)
        :param p_phase: Phase parameters (phi_k).  Scalar or vector of size N^2.
        :param p_splitter: Beamsplitter parameters (beta_k - pi/4).  Scalar of vector of size N(N-1).
        :return:
        """
        J = [0]
        def f(U): V = U - U_target; J[0] = np.linalg.norm(V)**2; return V
        grad = 2*np.real(self.grad_phi(np.eye(self.M), f, p_phase, p_splitter)[0])
        return (J[0], grad)




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

    def dot(self, v, p_phase=None, p_splitter=None, p_crossing=None, phi_out=None):
        v = np.array(v, dtype=np.complex)
        (p_crossing, phi_out, p_splitter) = self._defaults(p_phase, p_splitter, p_crossing, phi_out)
        # Loop through the crossings, one row at a time.  Then apply the final phase shifts.
        if (self.is_phase and self.phi_pos == 'in'): v *= np.exp(1j*phi_out).reshape((self.N,) + (1,)*(v.ndim-1))
        for (i, i1, i2, L, s) in zip(range(self.L), self.inds[:-1], self.inds[1:], self.lens, self.shifts):
            v = v if (self.perm[i] is None) else v[self.perm[i]]
            v[s:s+2*L] = self.X.dot(p_crossing[i1:i2], p_splitter[i1:i2], v[s:s+2*L])
        v = v if (self.perm[-1] is None) else v[self.perm[-1]]
        if (self.is_phase and self.phi_pos == 'out'): v *= np.exp(1j*phi_out).reshape((self.N,) + (1,)*(v.ndim-1))
        return v

    def _L2norm_fn(self, J):
        def f(U):
            V = U - np.eye(self.N, dtype=np.complex)
            if not self.is_phase: V -= np.diag(np.diag(V))
            J[0] = np.linalg.norm(V)**2; return 2*V
        return f

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
        grad = np.real(self.grad_phi(U_target.T.conj(), f, p_phase, p_splitter)[0])
        return (J[0], grad)

    def grad_phi(self, v, w, p_phase=None, p_splitter=None, p_crossing=None, phi_out=None):
        assert np.all([p is None for p in self.perm])  # TODO -- handle permutations.
        assert self.phi_pos == 'out'  # TODO -- handle phase shifts on inputs too.
        (p_crossing, phi_out, p_splitter) = self._defaults(p_phase, p_splitter, p_crossing, phi_out)
        v = v + 0j; Nc = self.n_cr*self.X.n_phase; grad = np.zeros(Nc+self.N*self.is_phase)
        grad_phiout = grad[Nc:]; grad_xing = grad[:Nc].reshape((self.n_cr, self.X.n_phase))

        # Forward pass
        for (n, i1, i2, L, s) in zip(range(self.L), self.inds[:-1], self.inds[1:], self.lens, self.shifts):
            v[s:s+2*L] = self.X.dot(p_crossing[i1:i2], p_splitter[i1:i2], v[s:s+2*L])
        if self.is_phase:
            v *= np.exp(1j*phi_out).reshape((self.N,) + (1,)*(v.ndim-1))
        # Can use callable w to specify a derivative based on the output matrix v.
        w = w(v) if callable(w) else np.array(w)
        # Reverse pass.  Back-propagate v rather than storing it to save memory (arithmetic intensity is low).
        if self.is_phase:
            grad_phiout[:] = np.real(-1j * w * v.conj()).sum(-1)
            w *= np.exp(-1j*phi_out).reshape((self.N,) + (1,)*(v.ndim-1))
            v *= np.exp(-1j*phi_out).reshape((self.N,) + (1,)*(v.ndim-1))
        for n in range(len(self.lens)-1, -1, -1):
            (i1, i2, L, s) = (self.inds[n], self.inds[n+1], self.lens[n], self.shifts[n])
            v[s:s+2*L] = self.X.dot(p_crossing[i1:i2], p_splitter[i1:i2], v[s:s+2*L], True)
            grad_xing[i1:i2] = self.X.grad(p_crossing[i1:i2], p_splitter[i1:i2], v[s:s+2*L], w[s:s+2*L])
            w[s:s+2*L] = self.X.dot(p_crossing[i1:i2], p_splitter[i1:i2], w[s:s+2*L], True)
        return (grad, w)

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

    def flip_crossings(self, inplace=False) -> 'StructuredMeshNetwork':
        r"""
        Flips the mesh, i.e. from one with output phase shifters to input phase shifters and vice versa.  Only works
        for MZI meshes.
        :param inplace: Performs the operation inplace or returns a copy.
        :return: The flipped StructuredMeshNetwork object.
        """
        if not inplace:
            self = self.copy()

        if (self.phi_pos == 'out'):
            assert isinstance(self.X, MZICrossing)
            self.X = MZICrossingOutPhase()
            for (m, len, shift, ind) in list(zip(range(self.L), self.lens, self.shifts, self.inds))[::-1]:
                phi = self.p_crossing[ind:ind+len, 1]
                (psi1, psi2) = self.phi_out[shift:shift+2*len].reshape([len, 2]).T
                (phi[:], psi1[:], psi2[:]) = np.array([psi2-psi1, phi+psi1, psi1])
            self.phi_pos = 'in'
        else:
            assert isinstance(self.X, MZICrossingOutPhase)
            self.X = MZICrossing()
            for (m, len, shift, ind) in list(zip(range(self.L), self.lens, self.shifts, self.inds)):
                phi = self.p_crossing[ind:ind+len, 1]
                (psi1, psi2) = self.phi_out[shift:shift+2*len].reshape([len, 2]).T
                (phi[:], psi1[:], psi2[:]) = np.array([psi1-psi2, psi2, phi+psi2])
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
        out = np.zeros([self.L, (self.N+1)//2, self.X.n_splitter]) * np.nan
        for (i, j0, nj, ind) in zip(range(self.L), self.shifts, self.lens, self.inds):
            out[i, j0//2:j0//2+nj, :] = self.p_splitter[ind:ind+nj, :]
        return out

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
        s_out = p_splitter * np.zeros([self.n_cr, X.n_splitter])
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
