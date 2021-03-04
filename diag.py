# meshes/diag.py
# Ryan Hamerly, 3/3/21
#
# Implements the matrix diagonalization method for self-calibrating MZI meshes.
#
# History
#   01/05/21: Method invented (part of meshes/clements.py)
#   03/03/21: Extended to general meshes, including Reck.  Moved to this file.

import numpy as np
from numba import njit
from typing import Callable
from .mesh import MeshNetwork, StructuredMeshNetwork, IdentityNetwork
from .crossing import MZICrossing, MZICrossingOutPhase

# Iterative (theta, phi) optimization to solve: <a|T(theta, phi)|b> = c.  JITted to speed up the for loop.
@njit
def Tsolve_abc(p_splitter, a, b, c, n):
    (Cp, C) = np.cos(p_splitter + np.pi/4); (Sp, S) = np.sin(p_splitter + np.pi/4)
    (a1, a2) = (a[0]*C + 1j*a[1]*S, 1j*a[0]*S + a[1]*C); (b1, b2) = (b[0], b[1]); (theta, phi) = (np.pi/2, 0.)
    for i in range(n):
        a1p = a1 * np.exp(1j*theta); (u1, u2) = (a1p*Cp + 1j*a2*Sp, 1j*a1p*Sp + a2*Cp)
        phi = np.angle(c - b2*u2) - np.angle(b1*u1)
        #print (np.abs(c - b1*u1*np.exp(1j*phi) - b2*u2))
        b1p = b1 * np.exp(1j*phi); (u1, u2) = (b1p*Cp + 1j*b2*Sp, 1j*b1p*Sp + b2*Cp)
        theta = np.angle(c - u2*a2) - np.angle(u1*a1)
        #print (np.abs(c - u1*a1*np.exp(1j*theta) - u2*a2))
    return np.array([theta/2, phi])
@njit
def Tsolve_abc_out(p_splitter, a, b, c, n):
    (Cp, C) = np.cos(p_splitter + np.pi/4); (Sp, S) = np.sin(p_splitter + np.pi/4)
    (a1, a2) = (a[0], a[1]); (b1, b2) = (b[0]*Cp + 1j*b[1]*Sp, 1j*b[0]*Sp + b[1]*Cp); (theta, phi) = (np.pi/2, 0.)
    for i in range(n):
        b1p = b1 * np.exp(1j*theta); (u1, u2) = (b1p*C + 1j*b2*S, 1j*b1p*S + b2*C)
        phi = np.angle(c - u1*a1) - np.angle(u2*a2)
        #print (np.abs(c - u2*a2*np.exp(1j*phi) - u1*a1))
        a2p = a2 * np.exp(1j*phi); (u1, u2) = (a1*C + 1j*a2p*S, 1j*a1*S + a2p*C)
        theta = np.angle(c - b2*u2) - np.angle(b1*u1)
        #print (np.abs(c - b1*u1*np.exp(1j*theta) - b2*u2))
    return np.array([theta/2, phi])
# Numba-accelerated functions for T(theta, phi).
@njit
def T_mzi(p, s):
    (theta, phi) = p; psi = np.array([s[0]+s[1], s[0]-s[1], theta])
    (Cp, Cm, C) = np.cos(psi); (Sp, Sm, S) = np.sin(psi); f = np.exp(1j*phi)
    return np.exp(1j*theta) * np.array([[f * (1j*S*Cm - C*Sp),    1j*C*Cp - S*Sm],
                                        [f * (1j*C*Cp + S*Sm),   -1j*S*Cm - C*Sp]])
@njit
def T_mzi_o(p, s):
    (theta, phi) = p; psi = np.array([s[0]+s[1], s[0]-s[1], theta])
    (Cp, Cm, C) = np.cos(psi); (Sp, Sm, S) = np.sin(psi); f = np.exp(1j*phi)
    return np.exp(1j*theta) * np.array([[    (1j*S*Cm - C*Sp),       ( 1j*C*Cp - S*Sm)],
                                        [f * (1j*C*Cp + S*Sm),   f * (-1j*S*Cm - C*Sp)]])

def diagClements(m, U: np.ndarray):
    r"""
    Self-configures a Clements mesh according to the diagonalization method.
    :param m: Instance of SymClementsNetwork.
    :param U: Target matrix.
    :return:
    """
    N = m.N
    def nn_cl(m):
        return m+1
    def ijxyp_cl(m, n):
        return [N-1-m+n, n, -n, N-2-m+n, m*0+1] if (m[0]%2) else [N-1-n, m-n, n, m-n, m*0]
    diag(m.m1, m.m2, m.phi_diag, U, N-1, nn_cl, ijxyp_cl)

def diagReck(m, U: np.ndarray):
    r"""
    Self-configures a Reck mesh according to the diagonalization method.
    :param m: Instance of ReckNetwork.
    :param U: Target matrix.
    :return:
    """
    N = m.N; out = (m.phi_pos == 'out')
    def nn_rk(m): return N-1-m
    if out:
        def ijxyp_rk(m, n): return [m, N-1-n, m*2+n, N-2-n, m*0 + 1 - out]
        diag(m, None, m.phi_out, U, N-1, nn_rk, ijxyp_rk)
    else:
        def ijxyp_rk(m, n): return [N-1-n, m, -m*2-n, N-2-n, m*0 + 1 - out]
        diag(None, m, m.phi_out, U, N-1, nn_rk, ijxyp_rk)

def diag(m1: StructuredMeshNetwork, m2: StructuredMeshNetwork, phi_diag: np.ndarray, U: np.ndarray,
                  nm: int, nn: Callable, ijxyp: Callable):
    r"""
    Calibrates a beamsplitter the Clements mesh to the diagonalization method, detailed in my note.
    Accelerated by Numba JIT.
    :param m1: Left MZI mesh
    :param m2: Right MZI mesh
    :param phi_diag: Diagonal phase mask
    :param U: Target matrix.
    :param nm: Number of m's
    :param nn: Number of n's (function of m)
    :param ijxyp: Returns [i, j, x, y, p], function of (m, n).  (i, j): Matrix element zeroed.  (x, y): MZI position.
    p: which mesh (0 for m1, 1 for m2).
    :return:
    """
    assert (m1 is not None) or (m2 is not None)
    if (m1 is None): m1 = IdentityNetwork(m2.N, X=m2.X.flip())
    if (m2 is None): m2 = IdentityNetwork(m1.N, X=m1.X.flip())
    assert isinstance(m1.X, MZICrossing) and isinstance(m2.X, MZICrossingOutPhase)
    U = np.array(U, dtype=np.complex); N = m1.N; assert (U.shape == (N, N))
    # (V0)* V and W (W0)*, where V0 is ideal, V is with real components.  Goal: match U = V*W
    VdV = np.eye(N, dtype=np.complex); WWd = np.eye(N, dtype=np.complex); Z = np.eye(N, dtype=np.complex)
    p = []; ind = []; c = []
    for m in [m1, m2]:
        m.phi_out[:] = 0
        m.p_crossing[:, 0] = 0; m.p_crossing[:, 1] = 2*np.pi*np.random.rand(m.n_cr); Z = m.dot(Z)
        p.append(m.p_splitter*np.ones([m.n_cr, m.X.n_splitter])); ind.append(np.array(m.inds)); c.append(m.p_crossing)

    (i, j, x, y, r) = np.concatenate([np.array(ijxyp(np.repeat(m, nn(m)), np.arange(nn(m)))) for m in range(nm)], 1)
    [[i1, s1], [i2, s2]] = [map(np.array, [m.inds, m.shifts]) for m in [m1, m2]]
    (w1, w2) = (np.where(1-r)[0], np.where(r)[0]); (x1, y1, x2, y2) = (x[w1], y[w1], x[w2]+m2.L-1, y[w2]); ind = x*0
    ind[w1] = i1[x1] + (y1-s1[x1])//2
    ind[w2] = i2[x2] + (y2-s2[x2])//2

    #print (np.array([i, j, x, y, ind, r]).T)
    #print (U.shape); print (VdV.shape); print (WWd.shape)
    diagHelper(U, Z, VdV, WWd, *p, *c, np.array([i, j, ind, r]).T)
    phi_diag[:] = np.angle(np.diag(U)) - np.angle(np.sum(VdV * WWd.T, axis=1))

# Super-fast Numba JIT-accelerated Clements calibration code.  Runs 10-20x faster than pure Python version.
@njit
def diagHelper(U, Z, VdV, WWd, p1, p2, c1, c2, ijzp):
    #X = np.array(U)
    for k in range(len(ijzp)):
        (i, j, ind, p) = ijzp[k]  # (i, j): element to zero.  ind: MZI index.  p: 0 (left mesh) or 1 (right mesh)
        upper = (i < j)           # (i, j) in upper triangle
        if (p == 0):
            j1 = (j-1 if upper else j); (u, v) = U[i, j1:j1+2]
            if (u == 0. and v == 0.): (u, v) = (1.+0.j, 0.+0.j) if upper else (0.+0.j, 1.+0.j)
            T0dag = np.array([[v, u.conjugate()], [-u, v.conjugate()]]) / np.sqrt(u*u.conjugate() + v*v.conjugate())
            if upper: T0dag[:, :] = T0dag[:, ::-1]
            for M in [U, WWd]: M[:, j1:j1+2] = M[:, j1:j1+2].dot(T0dag)
            Z[:, j1:j1+2] = Z[:, j1:j1+2].dot(T_mzi(c1[ind], p1[ind]).conj().T)
            wj = WWd[:, j]; vi = VdV[i, :].dot(Z); res = wj.dot(vi) - wj[j1:j1+2].dot(vi[j1:j1+2])
            c1[ind] = Tsolve_abc(p1[ind], vi[j1:j1+2], wj[j1:j1+2], -res, 10)
            T = T_mzi(c1[ind], p1[ind])
            WWd[j1:j1+2, :] = T.dot(WWd[j1:j1+2, :])
            #X[:, j1:j1+2] = X[:, j1:j1+2].dot(T.conj().T)
        else:
            i1 = (i if upper else i-1); (u, v) = U[i1:i1+2, j]
            if (u == 0. and v == 0.): (u, v) = (0.+0.j, 1.+0.j) if upper else (1.+0.j, 0.+0.j)
            T0dag = np.array([[u.conjugate(), v.conjugate()], [ v, -u]]) / np.sqrt(u*u.conjugate() + v*v.conjugate())
            if upper: T0dag[:, :] = T0dag[::-1, :]
            for M in [U, VdV]: M[i1:i1+2, :] = T0dag.dot(M[i1:i1+2, :])
            Z[i1:i1+2, :] = T_mzi_o(c2[ind], p2[ind]).conj().T.dot(Z[i1:i1+2, :])
            wj = Z.dot(WWd[:, j]); vi = VdV[i, :]; res = wj.dot(vi) - wj[i1:i1+2].dot(vi[i1:i1+2])
            c2[ind] = Tsolve_abc_out(p2[ind], vi[i1:i1+2], wj[i1:i1+2], -res, 10)
            T = T_mzi_o(c2[ind], p2[ind])
            VdV[:, i1:i1+2] = VdV[:, i1:i1+2].dot(T)
            #X[i1:i1+2, :] = T.conj().T.dot(X[i1:i1+2, :])
        #print ((i, j), ':', ind, p)#, ' *'[err_i])
        #print ((np.abs(X) > 1e-6).astype(int))
