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


import autograd.numpy as npa
import numpy as np
from typing import List, Any
from .crossing import Crossing, MZICrossing


# The base class MeshNetwork.

class MeshNetwork:
    p_phase: npa.array
    p_splitter: npa.array

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
    def dot(self, v, p_phase=None, p_splitter=None):
        r"""
        Computes the dot product between the splitter and a vector v.
        :param v: Input vector / matrix.
        :param p_phase: Phase parameters.  Defaults to stored values.
        :param p_splitter: Splitter angle parameters (deviation from pi/4).  Defaults to stored values.
        :return: Output vector / matrix.
        """
        raise NotImplementedError()
    def matrix(self, p_phase=None, p_splitter=None):
        r"""
        Computes the input-output matrix.  Equivalent to self.dot(np.eye(self.N))
        :param p_phase: Phase parameters.  Defaults to stored values.
        :param p_splitter: Splitter angle parameters (deviation from pi/4).  Defaults to stored values.
        :return: NxN matrix.
        """
        return self.dot(np.eye(self.N), p_phase, p_splitter)
    def grad_phi(self, v, w, p_phase=None, p_splitter=None):
        r"""
        Computes the gradient with respect to phase shifts phi.
        :param v: Input matrix / vector (normally np.eye(M)), size (M, ...)
        :param w: Output gradient matrix / vector dJ/d(U*), size (N, ...)
        :param p_phase: Phase parameters (phi_k).  Scalar or vector of size N^2.
        :param p_splitter: Beamsplitter parameters (beta_k - pi/4).  Scalar of vector of size N(N-1).
        :return:
        """
        raise NotImplementedError()
    def grad_phi_target(self, U_target, p_phase=None, p_splitter=None):
        r"""
        Gets the gradient of the L2 norm |U - U_target|^2 with respect to the phase parameters.
        :param U_target: Target matrix, size (M, N)
        :param p_phase: Phase parameters (phi_k).  Scalar or vector of size N^2.
        :param p_splitter: Beamsplitter parameters (beta_k - pi/4).  Scalar of vector of size N(N-1).
        :return:
        """
        J = [0]
        def f(U):
            V = U - U_target; J[0] = np.linalg.norm(V)**2; return V
        grad = 2*np.real(self.grad_phi(np.eye(self.M), f, p_phase, p_splitter))
        return (J[0], grad)



class StructuredMeshNetwork(MeshNetwork):
    _N: int
    inds: List[int]
    lens: List[int]
    shifts: List[int]
    X: Crossing

    def __init__(self,
                 N: int,
                 lens: List[int],
                 shifts: List[int],
                 p_phase: Any=0.,
                 p_splitter: Any=0.,
                 p_crossing=None,
                 phi_out=None,
                 X: Crossing=MZICrossing()):
        r"""
        Manages the beamsplitters for a Clements beamsplitter mesh.
        :param N: Number of inputs / outputs.
        :param lens: List of number of MZM's in each column.
        :param shifts: List of the shifts for each column.
        :param p_phase: Parameters [phi_i] for phase shifters.
        :param p_splitter: Parameters [beta_i - pi/4] for beam splitters.
        """
        self.X = X
        self._N = N
        #assert N%2 == 0
        self.lens = lens; n_cr = sum(lens)
        self.shifts = shifts
        self.inds = np.cumsum([0] + list(lens)).tolist()
        self.p_phase = p_phase * np.ones(n_cr*X.n_phase + N, dtype=np.float)
        self.p_splitter = np.array(p_splitter)
        if not (p_crossing is None): self.p_crossing[:] = p_crossing
        if not (phi_out is None): self.phi_out[:] = phi_out
        assert len(shifts) == len(lens)
        assert self.p_phase.shape in [(), (n_cr*X.n_phase + self.N,)]
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
    def _get_p_crossing(self, p_phase=None):
        return (self.p_phase if (p_phase is None) else p_phase)[:-self.N].reshape([self.n_cr, self.X.n_phase])
    @property
    def phi_out(self):
        r"""
        Returns the segment of self.p_phase that encodes the output phases.
        :return: array of size (self.N,)
        """
        return self._get_phi_out(None)
    def _get_phi_out(self, p_phase=None):
        return (self.p_phase if (p_phase is None) else p_phase)[-self.N:]

    def _defaults(self, p_phase, p_splitter, p_crossing, phi_out):
        # Gets default values assuming certain inputs and current state of the mesh.
        assert (p_crossing is None) == (phi_out is None)
        p_splitter = (self.p_splitter if (p_splitter is None) else p_splitter) * np.ones([self.n_cr, self.X.n_splitter])
        if (p_crossing is not None):
            assert (p_crossing.shape == (self.n_cr, self.X.n_phase)) and (phi_out.shape == (self.N,))
            return (p_crossing, phi_out, p_splitter)
        else:
            return (self._get_p_crossing(p_phase), self._get_phi_out(p_phase), p_splitter)

    def dot(self, v, p_phase=None, p_splitter=None, p_crossing=None, phi_out=None):
        v = np.array(v, dtype=np.complex)
        (p_crossing, phi_out, p_splitter) = self._defaults(p_phase, p_splitter, p_crossing, phi_out)
        # Loop through the crossings, one row at a time.  Then apply the final phase shifts.
        for (i1, i2, L, s) in zip(self.inds[:-1], self.inds[1:], self.lens, self.shifts):
            v[s:s+2*L] = self.X.dot(p_crossing[i1:i2], p_splitter[i1:i2], v[s:s+2*L])
        v *= np.exp(1j*phi_out).reshape((self.N,) + (1,)*(v.ndim-1))
        return v

    def grad_phi(self, v, w, p_phase=None, p_splitter=None, p_crossing=None, phi_out=None):
        (p_crossing, phi_out, p_splitter) = self._defaults(p_phase, p_splitter, p_crossing, phi_out)
        vList = np.zeros((len(self.lens)+1,) + v.shape, dtype=np.complex); vList[0] = v
        grad = np.zeros(len(self.p_phase), dtype=np.float)
        grad_crossing = grad[:-self.N].reshape((self.n_cr, self.X.n_phase)); grad_phiout = grad[-self.N:]
        # Forward pass
        for (n, i1, i2, L, s) in zip(range(self.L), self.inds[:-1], self.inds[1:], self.lens, self.shifts):
            vList[n+1, s:s+2*L] = self.X.dot(p_crossing[i1:i2], p_splitter[i1:i2], vList[n, s:s+2*L])
            vList[n+1, :s] = vList[n, :s]; vList[n+1, s+2*L:] = vList[n, s+2*L:]
        v_out = vList[-1] * np.exp(1j*phi_out).reshape((self.N,) + (1,)*(v.ndim-1))
        # Can use callable w to specify a derivative based on the output matrix v_out.
        w = w(v_out) if callable(w) else np.array(w)
        # Reverse pass
        grad_phiout[:] = np.real(-1j * w * v_out.conj()).sum(-1)
        w *= np.exp(-1j*phi_out).reshape((self.N,) + (1,)*(v.ndim-1))
        for n in range(len(self.lens)-1, -1, -1):
            (i1, i2, L, s) = (self.inds[n], self.inds[n+1], self.lens[n], self.shifts[n])
            grad_crossing[i1:i2] = self.X.grad(p_crossing[i1:i2], p_splitter[i1:i2], vList[n, s:s+2*L], w[s:s+2*L])
            w[s:s+2*L] = self.X.dot(p_crossing[i1:i2], p_splitter[i1:i2], w[s:s+2*L], True)
        return (grad, w)
