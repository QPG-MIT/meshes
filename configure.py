# meshes/configure.py
# Ryan Hamerly, 3/3/21
#
# Self-configuration routines for programming MZI meshes.  Currently implements the direct and diagonalization methods.
#
# History
#   01/05/21: Method invented (part of meshes/clements.py)
#   03/03/21: Extended to general meshes, including Reck.  Moved to this file.
#   03/06/21: JIT-ed my direct method code and added it here.
#   03/22/21: Added Saumil's local EC method (see arXiv:2103.04993).
#   04/13/21: Replaced 2*theta -> theta in phase shifters for consistency in notation.
#   04/28/21: Generalized diagHelper and associated helper functions.
#   08/27/22: Added support for 3-MZI.
#   09/13/22: Performance improvements to the diag* self-configuration method

import numpy as np
from numba import njit
from typing import Callable, Union
from .mesh import MeshNetwork, StructuredMeshNetwork, IdentityNetwork
from .crossing import MZICrossing, MZICrossingOutPhase, MZICrossing3, MZICrossing3OutPhase, \
    MZICrossingGeneric, MZICrossingGenericOutPhase, MZICrossingBell


T = dict()
Tsolve_abc = dict()
Tsolve_11 = dict()
INIT = dict()
diagHelper = dict()
directHelper = dict()

@njit(cache=True)
def inv_2x2(M):
    detM = M[0,0]*M[1,1] - M[0,1]*M[1,0]
    return np.array([[ M[1,1], -M[0,1]],
                     [-M[1,0],  M[0,0]]]) / detM

# Numba-accelerated functions for T(theta, phi).
@njit(cache=True)
def T_mzi(p, s):
    # MZICrossing
    if (len(s) == 3):
        ((T11, T12), (T21, T22)) = T_mzi(p, s[:2]); (C, S) = (np.cos(s[2]), np.sin(s[2]))  # If 3rd splitter
        return np.array([[C*T11 + 1j*S*T12, 1j*S*T11 + C*T12], [C*T21 + 1j*S*T22, 1j*S*T21 + C*T22]])
    (theta, phi) = p; psi = np.array([s[0]+s[1], s[0]-s[1], theta/2])
    (Cp, Cm, C) = np.cos(psi); (Sp, Sm, S) = np.sin(psi); f = np.exp(1j*phi); t = np.exp(1j*theta/2)
    return t * np.array([[f*(1j*S*Cm - C*Sp), 1j*C*Cp - S*Sm], [f*(1j*C*Cp + S*Sm), -1j*S*Cm - C*Sp]])
@njit(cache=True)
def T_mzi_o(p, s):
    # MZICrossingOutPhase
    if (len(s) == 3):
        ((T11, T12), (T21, T22)) = T_mzi_o(p, s[:2]); (C, S) = (np.cos(s[2]), np.sin(s[2]))  # If 3rd splitter
        return np.array([[C*T11 + 1j*S*T21, C*T12 + 1j*S*T22], [1j*S*T11 + C*T21, 1j*S*T12 + C*T22]])
    (theta, phi) = p; psi = np.array([s[0]+s[1], s[0]-s[1], theta/2])
    (Cp, Cm, C) = np.cos(psi); (Sp, Sm, S) = np.sin(psi); f = np.exp(1j*phi); t = np.exp(1j*theta/2)
    return t * np.array([[1j*S*Cm - C*Sp, 1j*C*Cp - S*Sm], [f*(1j*C*Cp + S*Sm), f*(-1j*S*Cm - C*Sp)]])
@njit(cache=True)
def T_gmzi(p, s):
    # MZICrossingGeneric
    (t_theta, t_phi) = np.exp(1j*p); (t11, t21, t12, t22) = np.exp(1j*(s[2:6] + s[6:10]) + s[10:14]/2)
    (Ca, Cb) = np.cos(s[:2] + np.pi/4); (Sa, Sb) = np.sin(s[:2] + np.pi/4)
    # T = V*U.  U: phi+bs1, V: theta+bs2
    U11 =    t11*Ca*t_phi;  U12 = 1j*t21*Sa;  V11 =    t12*Cb*t_theta;  V12 = 1j*t22*Sb
    U21 = 1j*t11*Sa*t_phi;  U22 =    t21*Ca;  V21 = 1j*t12*Sb*t_theta;  V22 =    t22*Cb
    return np.array([[V11*U11 + V12*U21, V11*U12 + V12*U22],
                     [V21*U11 + V22*U21, V21*U12 + V22*U22]])
@njit(cache=True)
def T_gmzi_o(p, s):
    # MZICrossingGenericOutPhase
    return T_gmzi(p, s).T[::-1,::-1]
@njit(cache=True)
def T_bell(p, s):
    # Symmetric (Bell) Crossing
    (theta, phi) = p; psi = np.array([s[0]+s[1], s[0]-s[1], (theta-phi)/2])
    (Cp, Cm, C) = np.cos(psi); (Sp, Sm, S) = np.sin(psi); t = np.exp(1j*(theta+phi)/2)
    return t * np.array([[-C*Sp + 1j*Cm*S, -Sm*S + 1j*Cp*C], [Sm*S + 1j*Cp*C, -C*Sp - 1j*Cm*S]])


T[MZICrossing]                = T_mzi
T[MZICrossingOutPhase]        = T_mzi_o
T[MZICrossing3]               = T_mzi
T[MZICrossing3OutPhase]       = T_mzi_o
T[MZICrossingGeneric]         = T_gmzi
T[MZICrossingGenericOutPhase] = T_gmzi_o
T[MZICrossingBell]            = T_bell


# Minimize f(x, y) = |A + B e^ix + C e^iy + D e^i(x+y)| by line search.
@njit(cache=True)
def linesearch(A, B, C, D, n, init): # sign):
    (theta, phi) = init # (np.pi/2*sign, 0)
    for i in range(n):
        temp  = np.exp(1j*theta); phi   = np.angle(-(A + temp*B)) - np.angle(C + temp*D)
        temp  = np.exp(1j*phi);   theta = np.angle(-(A + temp*C)) - np.angle(B + temp*D)
        # print (i, theta, phi)
    return np.array([theta, phi])

# Iterative (theta, phi) optimization to solve: <a|T(theta, phi)|b> = c.  JITted to speed up the for loop.
@njit(cache=True)
def Tsolve_abc_mzi(sp, a, b, c, n, init): #, sign):
    # MZICrossing, MZICrossing3
    (Ca, Cb) = np.cos(sp[:2] + np.pi/4); (Sa, Sb) = np.sin(sp[:2] + np.pi/4)
    (a1p, a2p) = (a[0]*Cb + 1j*a[1]*Sb, a[1]*Cb + 1j*a[0]*Sb)
    if len(sp) == 3: (Sc, Cc) = (np.sin(sp[2]), np.cos(sp[2])); (b1, b2) = (b[0]*Cc + 1j*b[1]*Sc, b[1]*Cc + 1j*b[0]*Sc)
    else: (b1, b2) = b
    A =     a2p*b2*Ca - c
    B =  1j*a1p*b2*Sa
    C =  1j*a2p*b1*Sa
    D =     a1p*b1*Ca
    return linesearch(A, B, C, D, n, init) #, sign)
@njit(cache=True)
def Tsolve_abc_mzi_o(sp, a, b, c, n, init): #, sign):
    # MZICrossingOutPhase, MZICrossing3OutPhase
    (Ca, Cb) = np.cos(sp[:2] + np.pi/4); (Sa, Sb) = np.sin(sp[:2] + np.pi/4)
    (b1p, b2p) = (b[0]*Ca + 1j*b[1]*Sa, b[1]*Ca + 1j*b[0]*Sa)
    if len(sp) == 3: (Sc, Cc) = (np.sin(sp[2]), np.cos(sp[2])); (a1, a2) = (a[0]*Cc + 1j*a[1]*Sc, a[1]*Cc + 1j*a[0]*Sc)
    else: (a1, a2) = a
    A =  1j*a1*b2p*Sb - c
    B =     a1*b1p*Cb
    C =     a2*b2p*Cb
    D =  1j*a2*b1p*Sb
    return linesearch(A, B, C, D, n, init) #, sign)
@njit(cache=True)
def Tsolve_abc_gmzi(p_splitter, a, b, c, n, init): #, sign):
    # MZICrossingGeneric
    (t11, t21, t12, t22) = np.exp(1j*(p_splitter[2:6] + p_splitter[6:10]) + p_splitter[10:14]/2)
    (Ca, Cb) = np.cos(p_splitter[:2] + np.pi/4); (Sa, Sb) = np.sin(p_splitter[:2] + np.pi/4)
    (a1p, a2p) = ((a[0]*Cb + 1j*a[1]*Sb)*t12, (a[1]*Cb + 1j*a[0]*Sb)*t22); (b1p, b2p) = (b[0]*t11, b[1]*t21)
    A =     a2p*b2p*Ca - c
    B =  1j*a1p*b2p*Sa
    C =  1j*a2p*b1p*Sa
    D =     a1p*b1p*Ca
    return linesearch(A, B, C, D, n, init) #, sign)
@njit(cache=True)
def Tsolve_abc_gmzi_o(p_splitter, a, b, c, n, init): #, sign):
    # MZICrossingGenericOutPhase
    # Easy since T_out = T^tr[::-1, ::-1], so <a|T_out|b> = [b2,b1]*T*[a2,a1]
    return Tsolve_abc_gmzi(p_splitter, b[::-1], a[::-1], c, n, init) #, sign)
@njit(cache=True)
def Tsolve_abc_bell(sp, a, b, c, n, init): #, sign):
    # MZICrossingBell
    (Ca, Cb) = np.cos(sp + np.pi/4); (Sa, Sb) = np.sin(sp + np.pi/4)
    (a1p, a2p) = (a[0]*Cb + 1j*a[1]*Sb, a[1]*Cb + 1j*a[0]*Sb)
    (b1p, b2p) = (b[0]*Ca + 1j*b[1]*Sa, b[1]*Ca + 1j*b[0]*Sa)
    A = -c; B = a1p*b1p; C = a2p*b2p
    # Solve A + B e^ix + C e^iy = 0, or find best approximate (x, y).  This can be done analytically.
    (absA, absB, absC) = (np.abs(A), np.abs(B), np.abs(C))
    cos_q = (absA**2 - absB**2 - absC**2) / (2*absB*absC)
    if   (cos_q >= +1): z = np.angle(B) - np.angle(C)
    elif (cos_q <= -1): z = np.angle(B) - np.angle(C) + np.pi
    else:               z = np.angle(B) - np.angle(C) + np.sign(init[0])*np.arccos(cos_q) #sign*np.arccos(cos_q)
    BC = B + C*np.exp(1j*z); theta = np.angle(-A) - np.angle(BC); phi = theta + z
    return np.array([theta, phi])

Tsolve_abc[MZICrossing]                = Tsolve_abc_mzi
Tsolve_abc[MZICrossingOutPhase]        = Tsolve_abc_mzi_o
Tsolve_abc[MZICrossing3]               = Tsolve_abc_mzi
Tsolve_abc[MZICrossing3OutPhase]       = Tsolve_abc_mzi_o
Tsolve_abc[MZICrossingGeneric]         = Tsolve_abc_gmzi
Tsolve_abc[MZICrossingGenericOutPhase] = Tsolve_abc_gmzi_o
Tsolve_abc[MZICrossingBell]            = Tsolve_abc_bell

# Optimization to solve T(theta, phi)[0, 0] = T
@njit(cache=True)
def Tsolve_11_mzi(T, p_splitter):
    (alpha, beta) = p_splitter
    Cp = np.cos(alpha+beta); Cm = np.cos(alpha-beta); Sp = np.sin(alpha+beta); Sm = np.sin(alpha-beta)
    # Input target T = T[0, 0]
    S2 = (np.abs(T)**2 - Sp**2) / (Cm**2 - Sp**2)
    theta2 = np.arcsin(np.sqrt(np.minimum(np.maximum(S2, 0.), 1.)))
    if (np.isnan(theta2)): theta2 = 0.
    phi = np.angle(T) - np.angle(1j*Cm*np.sin(theta2) - np.cos(theta2)*Sp) - theta2
    return np.array([theta2*2, phi])
Tsolve_11[MZICrossing] = Tsolve_11_mzi
Tsolve_11[MZICrossingOutPhase] = Tsolve_11_mzi

INIT[MZICrossing]                = ([np.pi/2, 0], [-np.pi/2, 0])
INIT[MZICrossingOutPhase]        = ([np.pi/2, 0], [-np.pi/2, 0])
INIT[MZICrossing3]               = ([np.pi/2, -np.pi/2], [-np.pi/2, np.pi/2])
INIT[MZICrossing3OutPhase]       = ([np.pi/2, -np.pi/2], [-np.pi/2, np.pi/2])
INIT[MZICrossingGeneric]         = ([np.pi/2, 0], [-np.pi/2, 0])
INIT[MZICrossingGenericOutPhase] = ([np.pi/2, 0], [-np.pi/2, 0])
INIT[MZICrossingBell]            = ([np.pi/2, 0], [-np.pi/2, 0])


JIT = True

def diag(m1: Union[StructuredMeshNetwork, None], m2: Union[StructuredMeshNetwork, None],
         phi_diag: np.ndarray, U: np.ndarray, nm: int, nn: Callable, ijxyp: Callable,
         improved: bool, sigp: float=0., init='rand'):
    r"""
    Calibrates a beamsplitter mesh using the diagonalization method, detailed in my note.
    Accelerated by Numba JIT.
    :param m1: Left MZI mesh
    :param m2: Right MZI mesh
    :param phi_diag: Diagonal phase mask
    :param U: Target matrix.
    :param nm: Number of m's
    :param nn: Number of n's (function of m)
    :param ijxyp: Returns [i, j, x, y, p], function of (m, n).  (i, j): Matrix element zeroed.  (x, y): MZI position.
    p: which mesh (0 for m1, 1 for m2).
    :param improved: Whether to use the improved method (better updates to V, W when nulling is imperfect)
    :param sigp: Programming error for each phase shifter (modeled as a Gaussian)
    :return:
    """
    assert (m1 is not None) or (m2 is not None)
    if (m1 is None): m1 = IdentityNetwork(m2.N, X=m2.X.flip())
    if (m2 is None): m2 = IdentityNetwork(m1.N, X=m1.X.flip())
    U = np.array(U, dtype=np.complex, order="C"); N = m1.N; assert (U.shape == (N, N))
    # (V0)* V and W (W0)*, where V0 is ideal, V is with real components.  Goal: match U = V*W
    VdV = np.eye(N, dtype=np.complex); WWd = np.eye(N, dtype=np.complex); Z = np.eye(N, dtype=np.complex)
    p = []; ind = []; c = []
    for m in [m1, m2]:
        m.phi_out[:] = 0; sp = m.p_splitter*np.ones([m.n_cr, m.X.n_splitter])
        if (type(m.X) in [MZICrossing, MZICrossingOutPhase]):
            m.p_crossing[:, 0] = 0; m.p_crossing[:, 1] = 2*np.pi*np.random.rand(m.n_cr);
        elif (type(m.X) == MZICrossing3) and (m.n_cr > 0):
            m.p_crossing[:] = np.array(m.X.Tsolve((1e-15, 1.), 'T1:', sp.T)[0]).T
        elif (type(m.X) == MZICrossing3OutPhase) and (m.n_cr > 0):
            m.p_crossing[:] = np.array(m.X.Tsolve((1., 1e-15), 'T:2', sp.T)[0]).T

        Z = m.dot(Z); p.append(sp); ind.append(np.array(m.inds)); c.append(m.p_crossing)

    (i, j, x, y, r) = np.concatenate([np.array(ijxyp(np.repeat(m, nn(m)), np.arange(nn(m)))) for m in range(nm)], 1)
    [[i1, s1], [i2, s2]] = [map(np.array, [m.inds, m.shifts]) for m in [m1, m2]]
    (w1, w2) = (np.where(1-r)[0], np.where(r)[0]); (x1, y1, x2, y2) = (x[w1], y[w1], x[w2]+m2.L-1, y[w2]); ind = x*0
    ind[w1] = i1[x1] + (y1-s1[x1])//2
    ind[w2] = i2[x2] + (y2-s2[x2])//2

    #print (np.array([i, j, x, y, ind, r]).T)
    #print (U.shape); print (VdV.shape); print (WWd.shape)
    #print (np.round(np.abs(Z), 2))
    ijzp = np.array([i, j, ind, r]).T;
    (init_p, init_m) = INIT[type(m1.X)]
    if (callable(init)): init = init(x, y)
    elif (init == 'rand'): init = np.random.randint(0, 2, len(ijzp))
    elif (init == 'uniform'): init = np.ones(len(ijzp), dtype=int)
    else: raise ValueError(init)
    if (init.ndim == 1): init = np.outer(init, init_p) + np.outer(1-init, init_m)

    diagHelper = get_diagHelper(type(m1.X), type(m2.X))
    diagHelper(U, Z, VdV, WWd, *p, *c, ijzp, init, improved, sigp)
    phi_diag[:] = np.angle(np.diag(U)) - np.angle(np.sum(VdV * WWd.T, axis=1))

# Super-fast Numba JIT-accelerated self-configuration code that implements the matrix diagonalization method.
# Runs 10-20x faster than pure Python version.
def get_diagHelper(type_i, type_o):
    if (type_i in diagHelper):
        return diagHelper[type_i]

    print ("Creating diagHelper for type ", type_i, " and ", type_o)

    T_i = T[type_i];  Tsolve_abc_i = Tsolve_abc[type_i]
    T_o = T[type_o];  Tsolve_abc_o = Tsolve_abc[type_o]

    def diagHelper_fn(U, Z, VdV, WWd, p1, p2, c1, c2, ijzp, init, improved, sigp):
        U_tr = U; WWd_tr = WWd; Z_tr = Z  # Inplace storage of transpose mat.T (improve cache hits on the 2x2 matmul's)

        #X = np.array(U)
        p_prev = 1
        for k in range(len(ijzp)):
            (i, j, ind, p) = ijzp[k]  # (i, j): element to zero.  ind: MZI index.  p: 0 (left mesh) or 1 (right mesh)
            upper = (i < j)           # (i, j) in upper triangle
            if (p == 0):
                if (p != p_prev):
                    if not improved: WWd_tr[:] = WWd.T
                    Z_tr[:] = Z.T; U_tr[:] = U.T; p_prev = p

                j1 = (j-1 if upper else j); (u, v) = U_tr[j1:j1+2, i] #U[i, j1:j1+2]
                if (u == 0. and v == 0.): (u, v) = (1.+0.j, 0.+0.j) if upper else (0.+0.j, 1.+0.j)
                T0dag = np.array([[v, u.conjugate()], [-u, v.conjugate()]]) / np.sqrt(u*u.conjugate() + v*v.conjugate())
                if upper: T0dag[:, :] = T0dag[:, ::-1]
                Z_tr[j1:j1+2, :] = inv_2x2(T_i(c1[ind], p1[ind])).T @ Z_tr[j1:j1+2, :]
                #Z[:, j1:j1+2] = Z[:, j1:j1+2] @ inv_2x2(T_i(c1[ind], p1[ind]))
                if improved:
                    wj2 = T0dag[:, j-j1]; vi2 = Z_tr[j1:j1+2, i]  # Simplified since VdV = WWd = I
                    c1[ind] = Tsolve_abc_i(p1[ind], vi2, wj2, 0., 10, init[k])
                else:
                    wj = T0dag[:, j-j1].T @ WWd_tr[j1:j1+2, :] #wj = WWd[:, j1:j1+2] @ T0dag[:, j-j1]
                    vi = Z_tr @ VdV[i, :] #vi = VdV[i, :] @ Z
                    res = wj @ vi - wj[j1:j1+2] @ vi[j1:j1+2]
                    c1[ind] = Tsolve_abc_i(p1[ind], vi[j1:j1+2], wj[j1:j1+2], -res, 10, init[k])
                c1[ind] += np.random.randn(2)*sigp
                T = T_i(c1[ind], p1[ind])
                if improved:
                    U_tr[j1:j1+2, :] = T.conj() @ U_tr[j1:j1+2, :]    # U[:, j1:j1+2]   = U[:, j1:j1+2] @ T.T.conj()
                    #WWd[:, j1:j1+2] = WWd[:, j1:j1+2] @ T.T.conj()   # (these cancel out)
                    #WWd[j1:j1+2, :] = T @ WWd[j1:j1+2, :]
                else:
                    U_tr[j1:j1+2, :] = T0dag.T @ U_tr[j1:j1+2, :]     # U[:, j1:j1+2]   = U[:, j1:j1+2] @ T0dag
                    WWd_tr[j1:j1+2, :] = T0dag.T @ WWd_tr[j1:j1+2, :] # WWd[:, j1:j1+2] = WWd[:, j1:j1+2] @ T0dag
                    WWd_tr[:, j1:j1+2] = WWd_tr[:, j1:j1+2] @ T.T     # WWd[j1:j1+2, :] = T @ WWd[j1:j1+2, :]
                #X[:, j1:j1+2] = X[:, j1:j1+2] @ T.conj().T

            else:
                if (p != p_prev):
                    if not improved: WWd[:] = WWd_tr.T
                    Z[:] = Z_tr.T; U[:] = U_tr.T; p_prev = p

                i1 = (i if upper else i-1); (u, v) = U[i1:i1+2, j]
                if (u == 0. and v == 0.): (u, v) = (0.+0.j, 1.+0.j) if upper else (1.+0.j, 0.+0.j)
                T0dag = np.array([[u.conjugate(), v.conjugate()], [ v, -u]]) / np.sqrt(u*u.conjugate() + v*v.conjugate())
                if upper: T0dag[:, :] = T0dag[::-1, :]
                Z[i1:i1+2, :] = inv_2x2(T_o(c2[ind], p2[ind])) @ Z[i1:i1+2, :]
                if improved:
                    wj2 = Z[i1:i1+2, j]; vi2 = T0dag[i-i1, :]  # Simplified since VdV = WWd = I
                    c2[ind] = Tsolve_abc_o(p2[ind], vi2, wj2, 0., 10, init[k])
                else:
                    wj = Z @ WWd[:, j]
                    vi = T0dag[i-i1, :] @ VdV[i1:i1+2, :]
                    res = wj @ vi - wj[i1:i1+2] @ vi[i1:i1+2]
                    c2[ind] = Tsolve_abc_o(p2[ind], vi[i1:i1+2], wj[i1:i1+2], -res, 10, init[k])
                c2[ind] += np.random.randn(2)*sigp
                T = T_o(c2[ind], p2[ind])
                if improved:
                    U[i1:i1+2, :]   = T.T.conj() @ U[i1:i1+2, :]
                    #VdV[i1:i1+2, :] = T.T.conj() @ VdV[i1:i1+2, :]  # (these cancel out)
                    #VdV[:, i1:i1+2] = VdV[:, i1:i1+2] @ T
                else:
                    U[i1:i1+2, :]   = T0dag @ U[i1:i1+2, :]
                    VdV[i1:i1+2, :] = T0dag @ VdV[i1:i1+2, :]
                    VdV[:, i1:i1+2] = VdV[:, i1:i1+2] @ T
                #X[i1:i1+2, :] = T.conj().T @ X[i1:i1+2, :]
            #print ((i, j), ':', ind, p)#, ' *'[err_i])
            #print ((np.abs(X) > 1e-6).astype(int))
    diagHelper_fn = njit(diagHelper_fn, cache=True) if JIT else diagHelper_fn
    diagHelper[type_i] = diagHelper_fn
    return diagHelper_fn


def direct(m: StructuredMeshNetwork, U: np.ndarray, diag: str, dk_max=1):
    r"""
    Calibrates a beamsplitter mesh to the direct method, detailed in my note.  Accelerated by Numba JIT.
    :param m: Mesh to configure.
    :param U: Target matrix.
    :param diag: Direction of diagonals ['up' or 'down']
    :param dk_max: Normally 1.  Use dk_max=2 for some divided Clements rectangles (see note).
    :return:
    """
    if (diag == 'up') and (m.phi_pos == 'out'): m.flip(True); direct(m, U, 'down'); m.flip(True); return
    assert (diag == 'down') and (m.phi_pos == 'in')
    assert isinstance(m.X, MZICrossingOutPhase)
    N = len(U); Upost = np.eye(N, dtype=np.complex); assert U.shape == (N, N)
    if np.isfortran(U): U = np.array(U, order="C")

    # Divide mesh into diagonals.  Make sure each diagonal can be uniquely targeted with a certain input waveguide.
    pos = np.concatenate([[[i, j, 2*j+j0, i-(2*j+j0)] for j in range(m.lens[i])] for (i, j0) in enumerate(m.shifts)])
    pos = pos[np.lexsort(pos[:, ::3].T)][::-1]; pos = np.split(pos, np.where(np.roll(pos[:,3], 1) != pos[:,3])[0])[1:]
    assert (np.diff(np.concatenate([pos_i[-1:] for pos_i in pos])[:, 2]) > 0).all()  # Diagonals have distinct inputs.
    p_splitter = m.p_splitter * np.ones([m.n_cr, 2])
    phi_out = 0*m.phi_out

    # Prepare data for the JIT-accelerated helper function, below.
    p = np.array([[2*j+j0, i+(2*j+j0)] for (i, j0) in enumerate(m.shifts) for j in range(m.lens[i])])
    p = p[np.lexsort(p.T)]; fld = dict(p[np.where(np.roll(p[:, 1], 1) != p[:, 1])[0]][:, ::-1])  # out-fields
    (k, v) = np.array(list(fld.items())).T; of = np.repeat(0, max(2*k)+5); is_of = of*0; of[k] = v; is_of[k] = 1
    env = np.maximum.accumulate((np.array(m.shifts) + np.array(m.lens)*2 - 2)[::-1])[::-1]
    Psum = np.cumsum(np.abs(U[::-1]**2), axis=0)[::-1]  # Psum[i, j] = norm(U[i:, j])^2
    Tlist = np.zeros([N+5, 2, 2], dtype=np.complex); w = np.zeros([N], dtype=np.complex)

    # Numba JIT-accelerated helper function.
    directHelper = get_directHelper(type(m.X))
    directHelper(np.concatenate(pos), np.array([len(p) for p in pos]), np.array(m.inds), p_splitter, m.p_crossing,
                 phi_out, of, is_of, env, m.L, U, Upost, Tlist, w, dk_max)

    m.phi_out[:] = phi_out

# Numba JIT helper function for the direct method.
def get_directHelper(type):
    assert type == MZICrossingOutPhase   # TODO -- make this generalizable like get_diagHelper
    if (type in directHelper):
        return directHelper[type]
    #
    # T_fn = T[type];  Tsolve_11_fn = Tsolve_11[type]

    def directHelper_fn(pos, lpos, inds, p_splitter, p_crossing, phi_out, outfield, is_of, env, L,
                        U, Upost, Tlist, w, dkmax):
        ipos = np.roll(np.cumsum(lpos), 1); ipos[0] = 0
        for (ip, lp) in zip(ipos, lpos):
            pos_i = pos[ip:ip+lp]
            N = len(U); E_in = 1.0; ptr_last = 0; l = pos_i[-1, 2]; w[:] = 0
            (i, ind, ptr) = (0, 0, 0)
            for (m, (i, j, ind, _)) in enumerate(pos_i[::-1]):  # Adjust MZIs *down* the diagonal one by one.
                ptr = inds[i] + j
                k = outfield[i+ind] if (is_of[i+ind]) else ind  # Output index (or two) k:k+dk
                dk = (min(dkmax, outfield[i+ind+2]-k) if (is_of[i+ind+2]) else dkmax) if (i < L-1) else 1
                U_kl = U[k:k+dk, l]
                T11 = (U_kl - w[k:k+dk]).sum()/(E_in*Upost[k:k+dk, ind]).sum()
                (theta, phi) = Tsolve_11_mzi(T11, p_splitter[ptr])
                p_crossing[ptr, 0] = theta
                if m: p_crossing[ptr_last, 1] = phi  # Set theta for crossing & phi to *upper-left* of crossing.
                else: phi_out[ind] = phi
                T = Tlist[m] = T_mzi(np.array([theta, phi]), p_splitter[ptr])   # TODO -- why T_mzi not T_mzi_o?
                w += E_in * T[0, 0] * Upost[:, ind]
                E_in *= T[1, 0]
                phi_out[ind+1] = np.angle(U[k, ind+1]) - np.angle(Upost[k, ind]*T[0, 1])
                ptr_last = ptr
            k = (ind+1) if (i == L-1 or ind > env[i+1]) else outfield[i+ind+2]
            dk = (min(dkmax, outfield[i+ind+4]-k) if (is_of[i+ind+4]) else dkmax) if (i < L-1) else 1
            # Set final phase shift.
            p_crossing[ptr,1] = phi = \
                (np.angle((U[k:k+dk,l]-w[k:k+dk]).sum()) - np.angle((E_in*Upost[k:k+dk,ind+1]).sum()))
            Upost[:, ind+1] *= np.exp(1j*phi)
            for (ind, T) in zip(pos_i[:, 2], Tlist[len(pos_i)-1::-1]):
                Upost[:, ind:ind+2] = Upost[:, ind:ind+2].dot(T)   # Multiply Upost by diagonal's T.
    directHelper_fn = njit(directHelper_fn, cache=True)
    directHelper[type] = directHelper_fn
    return directHelper_fn

def errcorr_local(mesh: StructuredMeshNetwork, p_splitter: np.ndarray) -> StructuredMeshNetwork:
    r"""
    Performs local error correction routine to match a target mesh's matrix.  Works for any single-pass mesh structure.
    For details, see: S. Bandyopadhyay et al., "Hardware error correction for programmable photonics" [2103.04993]
    :param mesh: The target (ideal) mesh.
    :param p_splitter: Splitter imperfections, dim=(mesh.n_cr, X.n_splitter)
    :return:
    """
    X = mesh.X; lens = mesh.lens; shifts = mesh.shifts; inds = mesh.inds; p_phase = mesh.p_phase
    out = mesh.copy(); phase = np.zeros([mesh.N])
    out.p_splitter = p_splitter; mesh.p_splitter = mesh.p_splitter * np.ones(out.p_splitter.shape)

    for (i, l, s, ind) in zip(range(out.L), lens, shifts, inds):
        ph0 = mesh.p_crossing[ind:ind+l]; ph = out.p_crossing[ind:ind+l]
        sp0 = mesh.p_splitter[ind:ind+l]; sp = out.p_splitter[ind:ind+l]
        phase = (phase if (mesh.perm[i] is None) else phase[mesh.perm[i]]); psi = phase[s:s+2*l]
        T0 = X.T(ph0, sp0); T0[:, 0, :] *= np.exp(1j*psi[::2]); T0[:, 1, :] *= np.exp(1j*psi[1::2])  # Target Tij
        ph[:] = np.array(X.Tsolve((T0[0, 0], T0[0, 1]), 'T1:', sp.T)[:1])[0].T
        psi[:] = np.angle(T0/X.T(ph, sp))[:, 0, :].T.flatten()
    phase = phase if (mesh.perm[-1] is None) else phase[mesh.perm[-1]]
    out.phi_out += phase
    return out
