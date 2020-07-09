# Meshes.py
# Ryan Hamerly, 6/19/20
#
# Implements the MeshNetwork class, which can be used for describing Reck and Clements meshes (or any other
# beamsplitter mesh where couplings occur between neighboring waveguides).  Clements decomposition is also
# implemented.

import autograd.numpy as npa
import numpy as np
from typing import List


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
    _p0_splitter: npa.array

    def __init__(self, N: int, lens: List[int], shifts: List[int], p_phase=0, p_splitter=0):
        r"""
        Manages the beamsplitters for a Clements beamsplitter mesh.
        :param N: Number of inputs / outputs.
        :param lens: List of number of MZM's in each column.
        :param shifts: List of the shifts for each column.
        :param p_phase: Parameters [phi_i] for phase shifters.
        :param p_splitter: Parameters [beta_i - pi/4] for beam splitters.
        """
        self._N = N
        assert N%2 == 0
        self.lens = lens
        self.shifts = shifts
        self.inds = np.cumsum([0] + np.outer(lens, [1, 1]).flatten().tolist()).tolist()
        self.p_phase = np.array(p_phase)
        self.p_splitter = np.array(p_splitter)
        self._p0_splitter = np.ones(2 * sum(self.lens)) * np.pi / 4
        assert len(shifts) == len(lens)
        assert self.p_phase.shape in [(), 2 * sum(self.lens) + self.N]
        assert self.p_splitter.shape in [(), 2 * sum(self.lens)]

    @property
    def L(self):
        return len(self.lens)
    @property
    def M(self):
        return self._N
    @property
    def N(self):
        return self._N

    def _reshape_params(self, p, p_bk: npa.ndarray, p0: float, ind: int, len: int, len_pad: int):
        if (p is None): p = p_bk
        elif not (type(p) is npa.ndarray): p = npa.array(p)
        if   p.ndim == 0:  out = npa.ones(len)*p
        elif p.ndim == 1:  out = p[ind:ind+len]
        else:              raise ValueError("Dimension of [p]")
        if type(p0) is np.ndarray:  out = out + p0[ind:ind+len]
        else:                       out = out + p0
        return npa.pad(out, (0, len_pad-len), mode='constant')

    def splitter_angles(self, n: int, p_splitter=None):
        r"""
        Returns of splitter angles (deviations from pi/4) for the n-th column of splitters in the mesh.  Note that
        there are 2 columns of splitters for each 2x2 block.
        :param n: Column index.
        :param p_splitter: One can specify the parameters in a vector (or scalar).  Default: use stored values.
        :return: A length-(N/2) vector.  If self.shift(n//2) == 1, the last entry will be zero.
        """
        return self._reshape_params(p_splitter, self.p_splitter, self._p0_splitter,
                                    self.inds[n], self.lens[n // 2], self.N // 2)
    def phase_shifts(self, n: int, p_phase=None):
        r"""
        Returns the phase shifts for the n-th column of the beamsplitter mesh.  Note that only the first channel of
        each block gets a phase shift.
        :param n: Column index.
        :param p_phase: One can specify the parameters in a vector (or scalar).  Default: use stored values.
        :return: A length-(N/2) vector.  If self.shift(n//2) == 1, the last entry will be zero.
        """
        return self._reshape_params(p_phase, self.p_phase, 0, self.inds[n], self.lens[n // 2], self.N // 2)
    def final_phase_shifts(self, p_phase=None):
        r"""
        Returns the final N phase shifts.  Unlike elsewhere in the mesh, here every channel gets a unique phase.
        :param p_phase: One can specify the parameters in a vector (or whatever).  Default: use stored values.
        :return: A length-N vector.
        """
        return self._reshape_params(p_phase, self.p_phase, 0, self.inds[-1], self.N, self.N)

    def dot(self, v, p_phase=None, p_splitter=None):
        for n in range(len(self.lens)):
            v = self._column_dot(v, n, p_phase, p_splitter)
        v = self._final_column_dot(v, p_phase)
        return v

    def _column_dot(self, v, n: int, p_phase=None, p_splitter=None):
        r"""
        Computes the dot product of the n-th splitter column with a vector v.  The splitter column consists of 2x2
        blocks S2*P2*S1*P1, where:
        P_i = [[e^{i theta_i}, 0], [0, 1]]
        S_i = [[cos(beta_i), i*sin(beta_i)], [i*sin(beta_i), cos(beta_i)]]
        Here the theta_i are phase shifts and beta_i are splitter angles (default pi/4).  The parameters are unique
        for each block and given by self.splitter_angles(...), self.phase_shifts(...).  Depending on self.shift(),
        the first block starts at index 0 or 1.
        :param v: Vector (or matrix) of inputs.
        :param n: Column index.
        :param p_phase: Phase parameters.  Defaults to stored values.
        :param p_splitter: Splitter angle parameters (deviation from pi/4).  Defaults to stored values.
        :return: Output vector / matrix.
        """
        shift = self.shifts[n]
        def arr_phi(x): return npa.outer(x, [1, 0]).reshape((self.N//2, 2) + (1,)*(v.ndim-1))
        def arr_beta(x): return x.reshape((self.N//2, 1) + (1,)*(v.ndim-1))
        (phi1,  phi2)  = [arr_phi(self.phase_shifts(m, p_phase))        for m in [2*n, 2*n+1]]
        (beta1, beta2) = [arr_beta(self.splitter_angles(m, p_splitter)) for m in [2*n, 2*n+1]]
        # First shift and reshape the input vector.
        v_shifted = npa.roll(v, -shift, axis=0).reshape((self.N//2, 2) + v.shape[1:])
        # Four steps: phase shift, beamsplitter, phase shift, beamsplitter.
        v_shifted = npa.exp(1j*phi1) * v_shifted
        v_shifted = (npa.cos(beta1)*v_shifted + 1j*npa.sin(beta1)*npa.roll(v_shifted, 1, axis=1))
        v_shifted = npa.exp(1j*phi2) * v_shifted
        v_shifted = (npa.cos(beta2)*v_shifted + 1j*npa.sin(beta2)*npa.roll(v_shifted, 1, axis=1))
        # Finally, shift back and reshape the output vector.
        return npa.roll(v_shifted.reshape(v.shape), shift, axis=0)

    def _column_dot_vjp(self, v, w, n:int, p_phase=None, p_splitter=None):
        assert v.shape == w.shape
        shift = self.shifts[n]
        def arr_phi(x): return npa.outer(x, [1, 0]).reshape((self.N//2, 2) + (1,)*(v.ndim-1))
        def arr_beta(x): return x.reshape((self.N//2, 1) + (1,)*(v.ndim-1))
        (phi1,  phi2)  = [arr_phi(self.phase_shifts(m, p_phase))        for m in [2*n, 2*n+1]]
        (beta1, beta2) = [arr_beta(self.splitter_angles(m, p_splitter)) for m in [2*n, 2*n+1]]
        # Forward pass: V -> U_n V
        v1 = npa.roll(v, -shift, axis=0).reshape((self.N//2, 2) + v.shape[1:])
        v2 = npa.exp(1j*phi1)*v1
        v3 = npa.cos(beta1)*v2 + 1j*npa.sin(beta1)*npa.roll(v2, 1, axis=1)
        v4 = npa.exp(1j*phi2)*v3
        v5 = npa.cos(beta2)*v4 + 1j*npa.sin(beta2)*npa.roll(v4, 1, axis=1)
        # Reverse pass: W -> U_n^dag W
        w5 = npa.roll(w, -shift, axis=0).reshape((self.N//2, 2) + w.shape[1:])
        w4 = npa.cos(beta2)*w5 - 1j*npa.sin(beta2)*npa.roll(w5, 1, axis=1)
        w3 = npa.exp(-1j*phi2)*w4
        w2 = npa.cos(beta1)*w3 - 1j*npa.sin(beta1)*npa.roll(w3, 1, axis=1)
        w1 = npa.exp(-1j*phi1)*w2
        # Gradients
        grad_phi1 = 1j*(w2[:,0].conj() * v2[:,0]).sum(axis=tuple(range(1, v.ndim)))
        grad_phi2 = 1j*(w4[:,0].conj() * v4[:,0]).sum(axis=tuple(range(1, v.ndim)))
        w0 = npa.roll(w1.reshape(w.shape), shift, axis=0)
        return (w0, grad_phi1[:self.lens[n]], grad_phi2[:self.lens[n]])

    def _final_column_dot(self, v, p_phase=None):
        r"""
        Applies the final column of phase shifters to the field.
        :param v: Vector (or matrix) of inputs.
        :param p_phase: Phase parameters.  Defaults to stored values.
        :return: Output vector / matrix.
        """
        phi_f = self.final_phase_shifts(p_phase).reshape((self.N,) + (1,)*(v.ndim-1))
        return npa.exp(1j*phi_f) * v

    def _final_column_dot_vjp(self, v, w, p_phase=None):
        assert v.shape == w.shape
        phi_f = self.final_phase_shifts(p_phase).reshape((self.N,) + (1,)*(v.ndim-1))
        w_out = npa.exp(-1j*phi_f) * w
        grad_phi = 1j*(w_out.conj() * v).sum(axis=tuple(range(1, v.ndim)))
        return (w_out, grad_phi)

    def grad_phi(self, v, w, p_phase=None, p_splitter=None):
        vList = np.zeros((len(self.lens)+1,) + v.shape, dtype=np.complex); vList[0] = v
        grad_phi = np.zeros(self.N**2, dtype=np.complex)
        # Forward pass
        for n in range(len(self.lens)):
            vList[n+1] = self._column_dot(vList[n], n, p_phase, p_splitter)
        # Can use w_func to specify a derivative based on the output matrix v.
        if callable(w):
            w = w(self._final_column_dot(vList[-1], p_phase))
        # Reverse pass
        (w, grad_phi[self.inds[-1]:]) = self._final_column_dot_vjp(vList[-1], w, p_phase)
        for n in range(len(self.lens)-1, -1, -1):
            (i1, i2, L) = (self.inds[2*n], self.inds[2*n + 1], self.lens[n])
            (w, grad_phi[i1:i1+L], grad_phi[i2:i2+L]) = self._column_dot_vjp(vList[n], w, n, p_phase, p_splitter)
        return grad_phi

    def flip_splitter_symmetry(self):
        r"""
        In the default configuration, when p_splitter = 0, all the splitters are identical and have angles pi/4.
        This function switches between the default configuration and the asymmetric configuration where the splitter
        angles are [+pi/4, -pi/4, +pi/4, -pi/4, ...].
        2x2 block matrix depends on configuration:
        Default:     i e^{i*phi1/2} [[sin(phi1/2), cos(phi1/2)], [cos(phi1/2), -sin(phi1/2)]] . diag([e^{i phi2}, 1])
        Alternating:   e^{i*phi1/2} [[cos(phi1/2), -sin(phi1/2)], [sin(phi1/2), cos(phi1/2)]] . diag([e^{i phi2}, 1])
        We convert between them as follows:
        T(phi1, phi2) = diag([-1, 1]).T'(phi1 + pi, phi2)
        :return:
        """
        phases = np.zeros(self.N); (inds, lens, shifts) = (self.inds, self.lens, self.shifts)
        self._p0_splitter = self._p0_splitter * np.concatenate([[1]*l + [-1]*l for (i, l) in enumerate(lens)])
        self.p_phase = self.p_phase * np.ones(2 * sum(self.lens) + self.N)
        for n in range(len(self.lens)):
            active_phases = phases[shifts[n]:shifts[n]+2*lens[n]].reshape((lens[n], 2))
            self.p_phase[inds[2 * n]:inds[2 * n] + lens[n]] += active_phases[:, 0] - active_phases[:, 1]
            self.p_phase[inds[2 * n + 1]:inds[2 * n + 1] + lens[n]] += np.pi
            active_phases[:] = np.outer(active_phases[:, 1], [1, 1]) + np.array([[np.pi, 0]])
        self.p_phase[inds[-1]:] += phases
        self.p_phase = np.mod(self.p_phase + np.pi, 2 * np.pi) - np.pi
        self.p_splitter = np.mod(self.p_splitter + np.pi, 2 * np.pi) - np.pi
