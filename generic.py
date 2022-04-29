# meshes/generic.py
# Ryan Hamerly, 4/27/22
#
# Implement a "generic" form of the diagonalization algorithm described in the paper "Accurate Self-Configuration of
# Rectangular Multiport Interferometers" [arXiv:2106.03249].  The algorithm runs by explicitly computing mesh.dot()
# and adjusting phase shifts at each step.  This is much slower than the method used in configure.py, but is compatible
# with a broader set of systems / errors.  In particular, it can be used for:
#   * Nonunitary errors, e.g. unbalanced losses
#   * Thermal crosstalk
#   * Self-configuration on actual hardware
#
# History
#   04/27/22: Created this file.

import numpy as np
from typing import List
from scipy.optimize import minimize
from .crossing import MZICrossingGeneric, MZICrossingGenericOutPhase
from .mesh import MeshNetwork, StructuredMeshNetwork


# Generic symmetric 2x2 matrix (used for diagonalization algorithm).
def Tsym(theta, phi):
    t = np.sin(theta) * np.exp(1j*phi)
    r = np.cos(theta)
    return np.array([[t, r], [r, -t.conj()]])
# Complex arctan2: calculate (theta, phi) so that (x, y) ~ (cos(theta), sin(theta) e^(i*phi))
def arctan2c(y, x):
    phi = np.angle(y) - np.angle(x)
    theta = np.arctan2(np.abs(y), np.abs(x))
    return (theta, phi)
# Clip to [0, 2*pi] with a margin (to prevent repeated large jumps of phase shift when updating close to 0 or 2*pi)
def clip(phase, margin=0.2):
    if (phase > -margin) and (phase < 2*np.pi + margin):
        return phase
    else:
        return np.mod(phase, 2*np.pi)

# Line search to optimize a 2D function
def linesearch(fn, get, set, inds):
    dp = 0
    for ind in inds:
        # Zero-phase & pi-phase measurements
        p = get(); x_0 = p[ind]; y_0 = fn()
        p[ind] = clip(x_0+np.pi); set(p); y_pi = fn()
        # Set to optimal angle
        y = (y_0 + y_pi)/2; dy = (y_0 - y_pi)/2
        dp = np.angle(-y * dy.conj()); p[ind] = clip(x_0+dp); set(p)

# A function of the form f(x, y) = |A + B e^ix + C e^iy + D e^i(x+y)|^2
def f_biphase(p, args):
    (theta, phi) = p; (A, B, C, D) = args
    (T, P) = (np.exp(1j*theta), np.exp(1j*phi))
    out = np.abs(A + B*T + C*P + D*T*P)**2
    return out
def df_biphase(p, args):
    (theta, phi) = p; (A, B, C, D) = args
    (T, P) = (np.exp(1j*theta), np.exp(1j*phi))
    g = A + B*T + C*P + D*T*P
    dg = np.array([1j*(B*T + D*T*P), 1j*(C*P + D*T*P)])
    return 2*np.real(dg.conj() * g)


def opt_mzi(fn, get, set):
    r"""
    Routine to optimize the parameters (theta, phi) of an MZI to zero f = <v|U|w*>.  Works as follows:
    (1) Calculate f(theta, phi), f(theta+pi, phi), f(theta, phi+pi), and f(theta+pi, phi+pi)
    (2) Fit f(theta, phi) = A + B e^{i*theta} + C e^{i*phi} + D e^{i(theta+phi)}
    (3) Minimize the fitted f(theta, phi) using scipy.optimize.minimize
    (4) Set the MZI to the optimal (theta, phi)
    :param fn: Function that returns f = <v|U|w*>
    :param get: Callable to obtain current (theta, phi) of active MZI
    :param set: Callable to set (theta, phi) of active MZI
    """
    # First, obtain fn at phases (0, pi)
    (x1,  x2 ) = get(); y00 = fn()
    (x1p, x2p) = (clip(x1+np.pi), clip(x2+np.pi))
    set((x1p, x2 ));    y10 = fn()
    set((x1p, x2p));    y11 = fn()
    set((x1 , x2p));    y01 = fn()
    # From these, fit to get (A, B, C, D) parameters of f_biphase
    args = np.array([[+1, +1, +1, +1],
                     [+1, +1, -1, -1],
                     [+1, -1, +1, -1],
                     [+1, -1, -1, +1]]) @ [y00, y01, y10, y11] / 4
    # SciPy optimizer.  Fast because it's only 2 variables, ~1 ms.
    soln = minimize(f_biphase, np.array([x1, x2]), jac=df_biphase, args=(args,), method='l-bfgs-b')
    (dx1, dx2) = soln.x
    set((clip(x1+dx1), clip(x2+dx2)))


class MeshNetworkGeneric(MeshNetwork):
    W: StructuredMeshNetwork
    V: StructuredMeshNetwork
    D: np.ndarray
    ind: List
    _N: int
    @property
    def M(self) -> int: return self._N
    @property
    def N(self) -> int: return self._N

    @property
    def n_cr(self) -> int: return self.W.n_cr + self.V.n_cr
    @property
    def p_crossing(self) -> np.ndarray:
        return self.p_phase[:-N].reshape([self.n_cr, 2])


    def _item(self, ind):
        name = ind[0].lower()
        if (name == 'd'):
            return (self.D, ind[1])
        else:
            mesh = {'w': self.W, 'v': self.V}[name]
            (col, idx) = ind[1:]; assert (0 <= idx < mesh.lens[col])
            return (mesh.p_crossing, mesh.inds[col]+idx)

    def __getitem__(self, ind):
        (p, idx) = self._item(ind); return p[idx]
    def __setitem__(self, ind, value):
        (p, idx) = self._item(ind); p[idx] = value

    def __init__(self, W, V, D, p_splitter, p_phase, ind=None):
        r"""
        Generic mesh network, implementing unitary U = VDW.
        :param W: Mesh upstream of the phase screen [StructuredMeshNetwork | None].
        :param V: Mesh downsteadm of the phase screen [StructuredMeshNetwork | None].
        :param D: Phase screen.
        :param ind: Self-configuration sequence.
        """
        if (W is None): W = StructuredMeshNetwork(V.N, [], [], X=MZICrossingGeneric(), is_phase=False)
        if (V is None): V = StructuredMeshNetwork(W.N, [], [], X=MZICrossingGenericOutPhase(), is_phase=False)
        assert (W.N == V.N); self.W = W; self.V = V; self._N = W.N
        self.D = D
        self.ind = ind
        self.p_splitter = p_splitter
        self.p_phase = p_phase

    def dot(self, v, p_phase=None, p_splitter=None) -> np.ndarray:
        (ph_w, ph_v, D) = (None, None, self.D) if (p_phase is None) else \
            (p_phase[:2*self.W.n_cr], p_phase[2*self.W.n_cr:-self.N], p_phase[-self.N:])
        (sp_w, sp_v) = (None, None) if (p_splitter is None) else (p_splitter[:self.W.n_cr], p_splitter[self.W.n_cr:])
        #assert (p_phase is None) and (p_splitter is None)  # No support for these here.
        v = self.W.dot(v, p_phase=ph_w, p_splitter=sp_w)
        v = np.exp(1j*D).reshape((self.N,)+(1,)*(v.ndim-1)) * v
        v = self.V.dot(v, p_phase=ph_v, p_splitter=sp_v)
        return v

    def config(self, M: np.ndarray, ind=None):
        r"""
        Configure the mesh using the diagonalization algorithm.
        :param M: Target matrix.
        :param ind: Self-configuration sequence.  If None, use the default sequence for the mesh.
        :return:
        """
        N = len(M)
        X = np.array(M, dtype=complex)
        V = np.eye(N, dtype=complex)
        W = np.eye(N, dtype=complex)
        if (ind is None): assert (self.ind is not None); ind = self.ind

        # Setter, getter, and function callback.
        def set(p):   self[block, col, row] = p
        def get():    return list(self[block, col, row])
        def opt_fn(): return np.vdot(v, self.dot(w))

        # Iterate through the MZIs in the mesh.
        for (block, col, row, i, zero) in ind:
            m = {'w': self.W, 'v': self.V}[block.lower()]
            j = m.shifts[col] + row*2
            if (block == 'w'):
                # For W updates, find Givens rotation X -> X T* that zeros the target index X_ij.  Then update
                # W -> T W, and given w* (j-th column of W*) and v (i-th column of V), zero <v|U|w*> by inputting
                # w* and measuring <v|out>, adjusting (theta, phi).
                # print ((i, j), (i, j+1))
                (u, v) = X[i, j:j+2]
                p = arctan2c(u.conj(), v.conj()) if zero else arctan2c(-v, u)
                Tdag = Tsym(*p)
                T    = Tdag.T.conj()
                X[:, j:j+2] = X[:, j:j+2] @ Tdag            # X -> X T*
                W[j:j+2, :] = T @ W[j:j+2, :]               # W -> T W
                w = W[j+zero, :].conj()                     # Input jth column of W* (j: column to zero)
                v = V[:, i]                                 # Target ith column of V

                opt_mzi(opt_fn, get, set)                   # Adjust (theta, phi) to set <v|U|w*> -> 0
            else:
                (i, j) = (j, i)
                # print ((i, j), (i+1, j))
                raise NotImplementedError()                 # TODO -- implement this for Clements.

        for i in range(N):
            # Adjust phase shifter psi_i so that <v_i|U|w_i*> = arg(X_ii)
            w = W[i, :].conj()
            v = V[:, i]
            self['d', i] += clip(np.angle(X[i, i]) - np.angle(opt_fn()))


# Initialize (but do not program) a ReckNetwork.
def ReckNetworkGeneric(N, p_splitter=None, p_phase=None):
    r"""
    Initialize a ReckNetwork as a MeshNetworkGeneric.
    :param N: Mesh size.
    :param p_splitter: Splitter imperfections.
    :return:
    """
    L = 2*N-3; i = np.arange(L)
    lens = np.minimum(i//2, (L-1-i)//2) + 1
    shifts = np.abs(i-N+2)
    if (p_splitter is None): p_splitter = np.zeros([N*(N-1)//2, 14])
    if (p_phase is None): p_phase = np.zeros(N**2)
    p_crossing = p_phase[:-N].reshape([N*(N-1)//2, 2])
    W = StructuredMeshNetwork(N, lens, shifts, X=MZICrossingGeneric(), is_phase=False)
    W.p_splitter = p_splitter; W.p_phase = p_phase[:-N]
    D = p_phase[-N:]
    V = None
    indReck = [('w', i+j-1, min(j, N-i-1), j, 1) for i in range(1, N) for j in range(i)]
    return MeshNetworkGeneric(W, V, D, p_splitter, p_phase, ind=indReck)
