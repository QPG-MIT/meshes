# meshes/clements.py
# Ryan Hamerly, 7/9/20
#
# Implements ClementsNetwork (subclass of MeshNetwork) with code to handle the Clements decomposition.
#
# History
#   11/26/19: Wrote code for function clemdec() (part of Note4/meshes.py), Clements decomposition.
#   06/18/20: Defined class ClementsNetwork (part of module meshes.py)
#   07/09/20: Moved to this file.
#   07/11/20: Added compatibility with custom crossings in crossing.py
#   12/15/20: Added Ratio Method tuning strategy for Clements.
#   01/05/21: Added new Clements tuning method.
#   03/03/21: Moved "new" tuning method (diagonalization method) to meshes/diag.py

import numpy as np
import warnings
from numba import njit
from typing import Any, Tuple
from .mesh import MeshNetwork, StructuredMeshNetwork, calibrateTriangle
from .crossing import Crossing, MZICrossing, SymCrossing
from .configure import diag

class ClementsNetwork(StructuredMeshNetwork):
    def __init__(self,
                 p_crossing: Any=0.,
                 phi_out: Any=0.,
                 p_splitter: Any=0.,
                 X: Crossing=MZICrossing(),
                 M: np.ndarray=None,
                 N: int=None,
                 warn=True,
                 phi_pos='out',
                 is_phase=True):
        r"""
        Mesh network based on the Reck (triangular) decomposition.
        :param N: Number of inputs / outputs.
        :param p_phase: Phase parameters (phi_k).  Scalar or vector of size N^2.
        :param p_splitter: Beamsplitter imperfection parameters.  Scalar or size (N(N-1)/2, X.n_splitter).
        :param M: A unitary matrix.  If specified, runs clemdec() to find the mesh realizing this unitary.
        """
        if (M == 'haar'):
            assert not np.any(list(map(np.iterable, [p_crossing, phi_out, p_splitter])))
            assert type(X) in [MZICrossing, SymCrossing]
            x = (np.outer(1, np.array([0]*(N//2) + [1]*(N//2-1))) + np.outer(np.arange(0, N, 2), 1)).flatten()
            y = np.outer([1]*(N//2), np.concatenate([np.arange(0, N, 2), np.arange(1, N-1, 2)])).flatten()
            k = (np.minimum(2*np.minimum(x, N-1-x)+1, 2*np.minimum(y, N-2-y)+2))
            theta = 2*np.arccos(np.random.uniform(0, np.ones(len(k)))**(1/(2*k)))*(np.random.randint([2]*len(k))*2-1)
            phi = 2*np.pi*np.random.uniform(0, np.ones(len(k)))
            p_crossing = np.array([theta, phi]).T
            phi_out = 2*np.pi*np.random.uniform(0, np.ones(N))[:N*is_phase]
        elif (M is not None): N = len(M)  # Start by getting N, shifts, lens, p_splitter
        elif (np.iterable(phi_out)): N = len(phi_out)
        else: assert N != None
        lens = list((N - np.arange(N) % 2)//2)
        shifts = list(np.arange(N) % 2)
        p_splitter = np.array(p_splitter); assert p_splitter.shape in [(), (N*(N-1)//2, X.n_splitter)]
        if (M is None):
            # Initialize from parameters.  Check parameters first.
            p_crossing = p_crossing * np.ones([N*(N-1)//2, X.n_phase]); phi_out = phi_out * np.ones(N)
        elif (type(M) != str):
            # Initialize from a matrix.  Calls clemdec() after correctly ordering the crossings.
            assert phi_pos == 'out'
            (pars_S, ch_S, pars_Smat, phi_out) = clemdec(M, p_splitter, X, warn)  # <-- The hard work is done here.
            p_crossing = (pars_Smat if N%2 else
                          pars_Smat.reshape([N//2, N, X.n_phase])[:, :-1, :]).reshape(N*(N-1)//2, X.n_phase)
        if (type(M) != str) and (not is_phase) and (np.iterable(phi_out)):
            warnings.warn("Cannot realize mesh exactly due to lack of external phase shifters: is_phase=False")
        super(ClementsNetwork, self).__init__(N, lens, shifts, p_splitter=p_splitter,
                                              p_crossing=p_crossing, phi_out=phi_out, X=X, phi_pos=phi_pos, is_phase=is_phase)

    def split(self) -> Tuple[StructuredMeshNetwork, StructuredMeshNetwork]:
        r"""
        Splits the mesh into two triangles: upper-left and lower-right.
        :return: The triangle parts (M1, M2), where the output of M1 feeds into M2.
        """
        N = self.N; inds = np.array(self.inds)
        assert (N > 2)
        (L1, L2) = [N//2*2-1,                    (N-1)//2*2                        ]
        shifts   = [np.arange(L1)%2,             N-2 - np.arange(L2)               ]
        lens     = [(N//2*2-np.arange(L1))//2,   (np.arange(L2)+2)//2              ]
        d_inds   = [np.array([0]*L1),            (np.arange(L2)[::-1]+((N+1)%2))//2]
        inds     = [inds[np.arange(L1)],         inds[np.arange(self.L-L2, self.L)]]
        phi_outs = np.outer([1, 0] if self.phi_pos == 'in' else [0, 1], self.phi_out)
        meshes   = []
        for (L, shift, len, d_ind, ind, phi_out) in zip([L1, L2], shifts, lens, d_inds, inds, phi_outs):
            p_crossing = np.concatenate([self.p_crossing[i+di:i+di+l] for (i, di, l) in zip(ind, d_ind, len)])
            p_splitter = (np.concatenate([self.p_splitter[i+di:i+di+l] for (i, di, l) in zip(ind, d_ind, len)])
                          if self.p_splitter.ndim else self.p_splitter)
            meshes.append(StructuredMeshNetwork(self.N, len.tolist(), shift.tolist(),
                                                p_crossing=p_crossing, p_splitter=p_splitter, phi_out=phi_out,
                                                X=self.X, phi_pos=self.phi_pos))
        return tuple(meshes)

    def copy(self) -> 'ClementsNetwork':
        return ClementsNetwork(N=self.N, p_splitter=np.array(self.p_splitter), p_crossing=np.array(self.p_crossing),
                               phi_out=np.array(self.phi_out), X=self.X, phi_pos=self.phi_pos)

    def flip_crossings(self, inplace=False) -> 'ClementsNetwork':
        return super(ClementsNetwork, self).flip_crossings(inplace)




# Miscellaneous functions and the Clements decomposition.

# Transfer matrix for the 2x2 MZI:
# -->--[phi]--| (pi/4   |--[2*theta]--| -(pi/4   |-->--
# -->---------|  +beta) |-------------|   +beta) |-->--
def _T(theta, phi, beta=0.):
    #return np.exp(1j*theta) * np.array([[np.exp(1j*phi)*np.cos(theta), -np.sin(theta)],
    #                                    [np.exp(1j*phi)*np.sin(theta),  np.cos(theta)]])
    cosTh = np.cos(theta); sinTh = np.sin(theta); cos2B = np.cos(beta); sin2B = np.sin(beta); eiPh = np.exp(1j*phi)
    return np.exp(1j*theta) * np.array([[eiPh*(cosTh - 1j*sin2B*sinTh), -cos2B*sinTh],
                                        [eiPh*(cos2B*sinTh),            cosTh + 1j*sin2B*sinTh]])
def _Tdag(theta, phi, beta=0.):
    return _T(theta, phi, beta).conj().transpose()

def clemdec(U: np.ndarray, p_splitter: Any=0., X: Crossing=MZICrossing(), warn=True):
    r"""
    Computes the Clements decomposition of a unitary matrix.  This code is called when instantiating a Clements network
    from a matrix.
    :param U: The unitary matrix.
    :return: A tuple (pars_S, ch_S, pars_Smat, phi_out)
        pars_S: A size-((N-1)*N/2, 2) array.  Each row contains a pair (theta, phi) for each 2x2 block
        ch_S: A size-((N-1)*N/2) vector.  Location of each block.
        pars_Smat: A size-(N, N/2, 2) array.  Gives pairs (theta, phi) ordered on the Clements grid.
        phi_out: Output phases.
    """
    U = np.array(U)
    n = len(U)
    n_even = (n-1)//2 * 2
    n_odd = n//2 * 2 - 1
    num_Tdag = ((n_odd+1)//2) ** 2
    num_T = n_even * (n_even+2)//4
    p_splitter = np.ones([num_T+num_Tdag, X.n_splitter]) * p_splitter
    beta_T = np.zeros([num_T, X.n_splitter], dtype=float)
    pars_Tdag = np.zeros([num_Tdag, X.n_phase], dtype=float)
    pars_T = np.zeros([num_T, X.n_phase], dtype=float)
    ch_Tdag = np.zeros([num_Tdag], dtype=int)
    ch_T = np.zeros([num_T], dtype=int)
    ind_Tdag = 0
    ind_T = 0
    ind_cols = np.pad(np.cumsum((n - np.arange(n) % 2)//2), (1, 0))
    err = 0
    for i in range(n - 1):
        for j in range(i + 1):
            if (i % 2 == 0):
                # U -> U T_{i-j,i-j+1}^dag
                u = U[n-1-j, i-j]
                v = U[n-1-j, i-j+1]
                beta_i = p_splitter[ind_cols[j] + (i-j)//2]      # print (ind_cols[j] + (i-j)//2)  <-- reorder_clements
                (pars_i, err_i) = X.Tsolve((np.conj(v), -np.conj(u)), 'T1:', beta_i)
                pars_Tdag[ind_Tdag] = pars_i
                ch_Tdag[ind_Tdag] = i-j
                ind_Tdag += 1
                err += err_i
                U[:, i-j:i-j+2] = U[:, i-j:i-j+2].dot(X.Tdag(pars_i, beta_i))
            else:
                # U -> T_{i-j,i-j+1} U
                u = U[n-2-i+j, j]
                v = U[n-1-i+j, j]
                beta_i = p_splitter[ind_cols[n-1-j] + (n-2-i+j)//2]  # print (ind_cols[n-1-j] + (n-2-i+j)//2)
                (pars_i, err_i) = X.Tsolve((v, -u), 'T2:', beta_i)
                pars_T[ind_T] = pars_i
                beta_T[ind_T] = beta_i
                ch_T[ind_T] = n-2-i+j
                ind_T += 1
                err += err_i
                U[n-2-i+j:n-i+j, :] = X.T(pars_i, beta_i).dot(U[n-2-i+j:n-i+j, :])
            #print (np.round(np.abs(U), 4))
    diag_phi = np.angle(np.diag(U))
    diag_phi2 = np.array(diag_phi)
    # We can express U = S1* S2* S3* ... D T1 T2 T3 ...
    # S1, ... come from pars_T.  T1, ... come from pars_Tdag
    # Need to convert this to U = D' S1' S2' S3' ... T1' T2' T3' ...
    # Each step: Si* D = D' Si.  Tdag(theta, phi)*diag(psi1, psi2) = diag(psi2-phi, psi2)*T(-theta, psi1-psi2)
    # Finally, combine the T's and S's.
    pars_S = np.zeros([num_T + num_Tdag, 2], dtype=float)
    ch_S = np.zeros([num_T + num_Tdag], dtype=int)
    ind_S = 0
    pars_S[:num_Tdag] = pars_Tdag
    ch_S[:num_Tdag] = ch_Tdag
    for (ind_S, (pars_i, beta_i, i)) in enumerate(zip(pars_T[::-1], beta_T[::-1], ch_T[::-1])):
        phi_i = diag_phi2[i:i+2]
        ch_S[num_Tdag + ind_S] = i
        (pars_S[num_Tdag + ind_S], diag_phi2[i:i+2]) = X.clemshift(phi_i, pars_i, beta_i)
    # Convert to a 2D matrix
    w = n // 2
    d = n
    ind = 0
    pars_Smat = np.zeros([d, w, X.n_phase], dtype=float)
    for j0 in range(0, 2 * n - 2, 2):
        for i in range(n):
            j = j0 - i
            if (j >= 0 and j < n - 1):
                pars_Smat[i, j // 2] = pars_S[ind]
                ind += 1
    if (warn and err):
        warnings.warn(
            "Clements calibration: {:d}/{:d} beam splitters could not be set correctly.".format(err, n*(n-1)//2))
    return (pars_S, ch_S, pars_Smat, np.mod(diag_phi2, 2 * np.pi))


class SymClementsNetwork(MeshNetwork):
    m1: StructuredMeshNetwork
    m2: StructuredMeshNetwork

    @property
    def n_cr(self):
        return self.N*(self.N-1)//2

    def __init__(self,
                 M: np.ndarray=None,
                 p_splitter: np.ndarray=0.,
                 clem: ClementsNetwork=None,
                 X: Crossing=MZICrossing(),
                 method='ratio',
                 warn=True):
        r"""
        Constructs a symmetric Clements network.  This is the product of two triangle meshes with the phase shifts
        along the rising center diagonal.
        :param M: Target matrix.
        :param p_splitter: Splitter imperfections.
        :param clem: If specified, converts a ClementsNetwork to SymClementsNetwork.
        :param X: The crossing type.  Currently only MZICrossing supported.
        :param method: Calibration method used: 'direct', 'ratio', 'mod', 'new'.
        :param warn: Throws a warning if mesh cannot be calibrated correctly.
        """
        assert (M is None) ^ (clem is None)
        assert (method in ['direct', 'ratio', 'mod', 'diag'])
        N = len(M) if (clem is None) else clem.N

        if (clem is not None):
            # Just split the existing Clements matrix into its triangles.
            (m1, m2) = clem.split(); (M1, M2) = (m1.matrix(), m2.matrix())
            m2.flip_crossings(inplace=True)
        elif (method != 'diag'):
            # Get the Clements decomposition of M and break it up into upper & lower triangles M = M2*M1.
            clem = ClementsNetwork(M=M, X=X)
            (m1, m2) = clem.split(); (M1, M2) = (m1.matrix(), m2.matrix())

        # Calibrate each triangle independently.
        clem = ClementsNetwork(N=N, X=X.flip(), phi_pos='in')
        (m1, m2) = clem.split();
        self.p_splitter = s = np.array(p_splitter) * np.ones([clem.n_cr, clem.X.n_splitter]);
        m1.flip_crossings(True)

        (m1.p_splitter, m2.p_splitter) = np.split(s, [m1.n_cr])

        if (method != 'diag'):
            m1.flip(True);
            calibrateTriangle(m2, M2, 'down', method, warn=warn)               #  <-- Hard work done here.
            calibrateTriangle(m1, M1[::-1,::-1].T, 'down', method, warn=warn)  #  <-- Hard work done here.
            m1.flip(True)
            if (clem.N%2): m1.phi_out[-1] = np.angle(M1[-1,-1])
            else: m2.phi_out[0] = np.angle(M2[0,0])

        m2.phi_out += m1.phi_out; m1.phi_out = 0
        self.p_phase = np.concatenate([m1.p_phase, m2.p_phase])
        (m1.p_phase, m2.p_phase) = np.split(self.p_phase, [len(m1.p_phase)])
        self.m1 = m1
        self.m2 = m2
        self.X = X

        if (method == 'diag'):
            diagClements(self, M)       #  <-- Hard work done here.

    def clements(self, phi_pos='out') -> ClementsNetwork:
        r"""
        Converts to ordinary Clements form.
        :return: A ClementsNetwork object.
        """
        (m1, m2) = (self.m1, self.m2); N = self.N
        # Preprocess the triangles to have the same crossings, output phase conventions.
        if (phi_pos == 'in'):
            m1 = m1.copy(); m2 = m2.copy(); (m1.phi_out, m2.phi_out) = (m2.phi_out, 0)
            m1 = m1.flip_crossings(); out = ClementsNetwork(N=self.N, phi_pos=phi_pos, X=self.X.flip())
        else:
            m2 = m2.flip_crossings(); out = ClementsNetwork(N=self.N, phi_pos=phi_pos, X=self.X)

        # Collate the columns of the two triangles to form the Clements mesh.
        p1 = np.split(m1.p_crossing, m1.inds[1:-1] + [m1.n_cr]*(N-m1.L))
        p2 = np.split(m2.p_crossing, [0]*(N-m2.L) + m2.inds[1:-1])
        out.p_crossing = np.concatenate([p for p12 in zip(p1, p2) for p in p12])
        if m1.p_splitter.ndim:
            s1 = np.split(m1.p_splitter, m1.inds[1:-1] + [m1.n_cr]*(N-m1.L))
            s2 = np.split(m2.p_splitter, [0]*(N-m2.L) + m2.inds[1:-1])
            out.p_splitter = np.concatenate([s for s12 in zip(s1, s2) for s in s12])
        else:
            out.p_splitter = np.array(m1.p_splitter)
        out.phi_out = m2.phi_out if (phi_pos == 'out') else m1.phi_out
        return out

    @property
    def phi_diag(self) -> np.ndarray:
        return self.m2.phi_out
    @phi_diag.setter
    def phi_diag(self, p):
        self.m2.phi_out = p

    @property
    def L(self) -> int:
        return self.m1.N
    @property
    def M(self) -> int:
        return self.m1.N
    @property
    def N(self) -> int:
        return self.m1.N
    def dot(self, v, p_phase=None, p_splitter=None) -> np.ndarray:
        (p1, p2) = np.split(p_phase, [len(self.m1.p_phase)]) if (np.iterable(p_phase)) else [p_phase]*2
        (s1, s2) = np.split(p_splitter, [self.m1.n_cr]) if (np.iterable(p_splitter)) else [p_splitter]*2
        return self.m2.dot(self.m1.dot(v, p1, s1), p2, s2)

    # TODO -- implement grad_phi

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

