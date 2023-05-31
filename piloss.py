# meshes/piloss.py
# Ryan Hamerly, 5/30/23
#
# Implements the Path-Independent Loss (PILOSS) mesh.
#
# History
#   05/30/23: Created this file.

import numpy as np
from scipy.linalg import eigvalsh
from typing import Any
from .mesh import StructuredMeshNetwork, ClippedNetwork
from .crossing import Crossing, MZICrossing
from .configure import T, Tsolve_abc


class PilossNetwork(ClippedNetwork):
    fact = 1

    def __init__(self,
                 M: np.ndarray   = None,
                 eig: float      = None,
                 N: int          = None,
                 p_phase: Any    = 0.,
                 p_splitter: Any = 0.,
                 p_crossing      = None,
                 phi_out         = None,
                 X: Crossing     = MZICrossing(),
                 phi_pos         = 'out',
                 config_iters    = 10,
                 is_phase        = True):
        r"""
        Mesh that realizes a non-unitary matrix with the Path-Independent Loss (PILOSS) structure.
        :param M: Target matrix (N*N), not necesssarily unitary.
        :param N: Mesh size (physical size is 2N*2N, can realize an N*N matrix)
        :param p_phase: Parameters [phi_i] for phase shifters.
        :param p_splitter: Parameters [alpha, beta] for beam splitters.
        :param p_crossing: Crossing parameterrs (theta, phi).  Takes place of p_phase.
        :param phi_out: External phase screen.  Takes place of p_phase.
        :param X: Crossing type.
        :param phi_pos: Position of external phase screen.
        :param config_iters: Number of iterative self-configuration steps employed (only if M is specified)
        :param is_phase: Whether phase screen exists.
        """
        if (N is None): N = len(M)
        lens = [N]*N; shifts = [0]*N;
        perm_int = np.concatenate([[0], (np.arange(2*N-2) ^ 1) + 1, [2*N-1]])
        perm_mzi = [np.arange(2*N) ^ (np.arange(2*N) >> 1 & 1) ^ i for i in [0, 1]]
        perm = np.array([perm_mzi[0]] + [perm_mzi[i%2][perm_int][perm_mzi[(i+1)%2]] for i in range(N-1)] +
                        [perm_mzi[(N-1)%2]])
        full = StructuredMeshNetwork(2*N, lens, shifts, p_phase=p_phase, p_splitter=p_splitter,
                 p_crossing=p_crossing, phi_out=phi_out, perm=perm, X=X, phi_pos=phi_pos, is_phase=is_phase)
        idx = np.arange(N)
        super(PilossNetwork, self).__init__(full, 2*idx + idx%2, 2*idx + (1-idx%2))
        if (M is not None):
            self.fact = np.sqrt(eig / eigvalsh(M.conj().T @ M, subset_by_index=[N-1, N-1])[0]) if eig else 1.0
            M = M * self.fact
            self.config(M, config_iters)

    def config(self, M: np.ndarray, ct=1):
        r"""
        Configures the PILOSS mesh to realize a target matrix using the iterative direct method.
        :param M: Target matrix.
        :param ct: Number of iterative self-configuration steps.  Usually converges in under 10 steps.
        :return:
        """
        full = self.full
        N = full.N//2
        assert (M.shape == (N, N))
        (x, y) = np.meshgrid(np.arange(N), np.arange(N))
        z = ((y-x&1)*(y+x) + (y-x&1 == 0)*(y-x))
        k_in = z*(z>0)*(N>z) - (z+1)*(z<0) + (2*N-1-z)*(z>N)
        z = ((y-x&1)*(y-x+N-1) + (y-x&1 == 0)*(y+x-N+1))
        k_out = z*(z>0)*(N>z) - (z+1)*(z<0) + (2*N-1-z)*(z>=N)

        for k in range(ct):
            U_post = full.matrix()
            U_pre  = np.eye(2*N, dtype=complex)
            sp = full.p_splitter * np.ones([full.n_cr, full.X.n_splitter])
            ph = full.p_crossing
            T_fn = T[type(full.X)]
            Tsolve = Tsolve_abc[type(full.X)]
            m = 0
            for m in range(N):
                U_post[:] = U_post[:, full.perm[m]]
                U_pre[:] = U_pre[full.perm[m]]
                for n in range(N):
                    (i, j) = (k_out[n,m], k_in[n,m]); (ii, jj) = (2*i + (1-i%2), 2*j + j%2)
                    Tmn = T_fn(ph[m*N+n], sp[m*N+n])
                    U_post[:, 2*n:2*n+2] = U_post[:, 2*n:2*n+2] @ np.linalg.inv(Tmn)
                    u_post = U_post[ii, :]; a = u_post[2*n:2*n+2]
                    u_pre  = U_pre[:, jj];  b = u_pre[2*n:2*n+2]
                    c = M[i, j] - (u_post @ u_pre - a @ b)
                    # print ((m, n), (i, j), (ii, jj), a, b, c)
                    ph1 = Tsolve(sp[m*N+n], a, b, c, 10, +1); T1 = T_fn(ph1, sp[m*N+n])
                    ph2 = Tsolve(sp[m*N+n], a, b, c, 10, -1); T2 = T_fn(ph2, sp[m*N+n])
                    ph[m*N+n] = ph1 if np.linalg.norm(T1 - Tmn) < np.linalg.norm(T2 - Tmn) else ph2
                    Tmn = T_fn(ph[m*N+n], sp[m*N+n])
                    # print (a @ T @ b - c)
                    U_pre[2*n:2*n+2, :] = Tmn @ U_pre[2*n:2*n+2]
