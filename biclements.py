# meshes/piloss.py
# Ryan Hamerly, 5/30/23
#
# Implements the BiClements mesh, a rectangular mesh with 2x the length of a Clements and open output ports on the top,
# that can realize arbitrary non-unitary matrices.
#
# History
#   05/30/23: Created this file.

import numpy as np
from scipy.linalg import eigvalsh, cholesky, qr
from numba import njit
from typing import Any
from .mesh import StructuredMeshNetwork, ClippedNetwork
from .crossing import Crossing, MZICrossing
from .configure import diag


class BiClementsNetwork(ClippedNetwork):
    fact = 1.0

    def __init__(self,
                 M: np.ndarray   = None,
                 eig: float      = None,
                 N: int          = None,
                 p_splitter: Any = 0.,
                 p_crossing      = 0.,
                 phi_out         = 0,
                 X: Crossing     = MZICrossing(),
                 phi_pos         = 'out',
                 method: str     = 'diag',
                 is_phase        = True):
        r"""
        Mesh that realizes a non-unitary matrix with the rectangular (BiClements) structure.
        :param M: Target matrix (N*N), not necesssarily unitary.
        :param eig: If specified, scales matrix to match the maximum eigenvalue of M*M.
        :param N: Mesh size (physical size is 2N*2N, can realize an N*N matrix)
        :param p_phase: Parameters [phi_i] for phase shifters.
        :param p_splitter: Parameters [alpha, beta] for beam splitters.
        :param p_crossing: Crossing parameterrs (theta, phi).  Takes place of p_phase.
        :param phi_out: External phase screen.  Takes place of p_phase.
        :param X: Crossing type.
        :param phi_pos: Position of external phase screen.
        :param is_phase: Whether phase screen exists.
        """
        if (N is None): N = len(M)
        L = N*2; n_cr = N*(3*N-1)//2
        shifts = np.abs(np.arange(L) - N); lens = N - (np.abs(np.arange(L) - N)+1)//2
        p_splitter = p_splitter + np.zeros([N*N, X.n_splitter])
        p_crossing = p_crossing + np.zeros([N*N, X.n_phase])
        phi_out    = phi_out    + np.zeros([N])

        full = StructuredMeshNetwork(2*N, lens, shifts, p_splitter=np.zeros([n_cr, X.n_splitter]),
                                     p_crossing=np.zeros([n_cr, X.n_phase]), phi_out=np.zeros([2*N]), X=X,
                                     phi_pos=phi_pos, is_phase=is_phase)
        # Map the input p_splitter, p_crossing, phi_out to the values of the mesh (indices are different due to
        # dummy crossings at the top of the mesh).
        for (i, ind, ln) in zip(range(full.L), full.inds, full.lens):
            full.p_splitter[ind+ln-N//2:ind+ln] = p_splitter[i*N//2:(i+1)*N//2]
            full.p_crossing[ind+ln-N//2:ind+ln] = p_crossing[i*N//2:(i+1)*N//2]
            full.phi_out[N:] = phi_out

        super(BiClementsNetwork, self).__init__(full, slice(N, None), slice(N, None))

        # Self-configure the mesh to match M, if provided.
        if (M is not None):
            assert (method in ['diag', 'diag*']) and (full.phi_pos == 'out') and (full.is_phase)
            (U, self.fact) = embed(M, eig)
            nm = 3*N//2-1
            def nn(m): return min(N+m+1, 2*N-m-1, 3*N-2*m-2)
            def ijxyp(m, n): return (np.maximum(m, 2*m-N+1), np.minimum(N+1+2*m, 2*N-1) - n,
                                     2*np.maximum(0, m-N//2+1) + n, np.minimum(2*m+N, 2*N-2) - n, 0*m)
            diag(full, None, full.phi_out, U, nm, nn, ijxyp, method == 'diag')


@njit
def givens(U, i, j, rt):
    (u, v) = U[i, j:j+2] if rt else U[i:i+2, j]
    T = np.array([[-v, u.conjugate()], [u, v.conjugate()]]) / np.sqrt(np.abs(u)**2 + np.abs(v)**2)
    if rt: U[:,j:j+2] = U[:,j:j+2].dot(T)
    else:  U[i:i+2,:] = T.T.dot(U[i:i+2,:])

@njit
def embed_helper(U):
    N = len(U)//2
    for m in range(N-1):
        for n in range(N-1-m):
            givens(U, n, 2*N-1-m, 0)
            givens(U, 2*N-1-m, n, 1)
    for m in range(N-1):
        for n in range(N-1-m):
            if (m % 2): givens(U, m//2, m//2+1+n, 1)
            else:       givens(U, m//2+n, m//2, 0)

def embed(M, eig):
    # First embed the N*N matrix M into a larger 2N*2N unitary
    #     [ U11 | U12 ]
    # U = [ ----+---- ]
    #     [ U21 |  M  ]
    M11 = np.array(M); (M, N) = M11.shape; assert (M == N)
    MtM = M11.T.conj().dot(M11); fact = np.sqrt(eig / eigvalsh(MtM, subset_by_index=[N-1, N-1])[0]) if eig else 1.0
    MtM *= fact**2; M11 *= fact; M21 = cholesky(np.eye(N) - MtM) #.T.conj()
    U = np.zeros([M+N, M+N], dtype=complex); U[:M, :N] = M11; U[M:, :N] = M21
    U[:, N:] = np.random.randn(M+N, M, 2).dot([1, 1j])
    (Q, R) = qr(U); Q *= np.diag(np.exp(1j*(np.angle(U)-np.angle(Q))))
    U = np.zeros([M+N, M+N], dtype=complex)
    U[N:, M:] = Q[:M, :N];  U[:N, M:] = Q[M:, :N]
    U[N:, :M] = Q[:M, N:];  U[:N, :M] = Q[M:, N:]

    # Next, perform the Givens rotations to make U11 lower-right triangular, while keeping 1/4 of the entries of U12
    # and U21 zero.
    embed_helper(U)
    return (U, fact)
