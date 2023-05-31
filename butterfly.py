# meshes/butterfly.py
# Ryan Hamerly, 3/22/21
#
# FFT butterfly fractal mesh.
#
# History
#   03/22/21: Created ButterflyNetwork class.
#   11/04/22: Improved stability of decomposition, added option to permute singular values.

import numpy as np
from numpy.linalg import svd
from scipy.linalg import cossin
from .mesh import StructuredMeshNetwork
from .crossing import Crossing, MZICrossing

class ButterflyNetwork(StructuredMeshNetwork):
    def __init__(self,
                 N:          int=None,
                 p_phase:    np.ndarray=0.,
                 p_splitter: np.ndarray=0.,
                 p_crossing: np.ndarray=None,
                 phi_out:    np.ndarray=None,
                 M:          np.ndarray=None,
                 X:          Crossing=MZICrossing(),
                 phi_pos:    str='out',
                 order:      bool=True):
        r"""
        Mesh based on the generalized FFT butterfly fractal.  This mesh has layers of nonlocal crossings (stride 2^k).
        As a result, the distribution of splitting angles is not tightly concentrated near the cross state as in the
        Reck or Clements mesh.
        :param N: Mesh size.  Not needed if matrix M specified.
        :param p_phase: Phase shifts, dim=(N(N-1)/2*X.n_phase + N)
        :param p_splitter: Splitter imperfections, dim=(N(N-1)/2, X.n_splitter)
        :param p_crossing: Crossing parameters, dim=(N(N-1)/2, X.n_phase)
        :param phi_out: Output phases, dim=(N)
        :param M: Target matrix.
        :param X: Crossing type.
        :param phi_pos: Position of phase screen.  Currently only 'out' is supported.
        """
        assert (N is None) ^ (M is None)
        assert (phi_pos == 'out')   # TODO -- phi_pos='in'
        N = (N if N else len(M))
        assert N == 2**int(np.log2(N))  # Size must be a power of 2.
        if (M is not None) and (M.dtype != np.complex): M = M.astype(np.complex)

        # Set up the mesh parameters and permutations for crossings with stride s > 1.
        lens = [N//2]*(N-1); shifts = [0]*(N-1); perm = [None]*N
        for i in range(1, N-1, 2):
            s = 2**np.binary_repr(i+1)[::-1].index('1')
            perm[i:i+2] = [(np.outer(1, x) + np.outer(np.arange(0, N, 2*s), 1)).flatten() for x in
                    [np.outer(np.arange(s),1)+np.array([[0,s]]), np.outer(1,np.arange(0,2*s,2))+np.array([[0],[1]])]]

        super(ButterflyNetwork, self).__init__(N, lens, shifts, p_phase=p_phase, p_splitter=p_splitter,
                                               p_crossing=p_crossing, phi_out=phi_out, perm=perm, X=X, phi_pos=phi_pos)

        # Configure to match target matrix, if provided.
        if (M is not None):
            assert isinstance(self.X, MZICrossing)  # TODO -- generalize.

            # Perform the recursive block-wise SVD of U to get the crossing amplitudes Dij.
            def configButterfly(U, Dij):
                N = len(U)
                if (N > 2):
                    (U11, U12, U21, U22) = (U[:N//2, :N//2], U[:N//2, N//2:], U[N//2:, :N//2], U[N//2:, N//2:])
                    (V, D, W) = cossin([U11, U12, U21, U22])
                    p = np.arange(N//2) if order else np.argsort(np.random.randn(N//2))
                    V1 = V[:N//2, p]; V2 = V[N//2:, p+N//2]; W1 = W[p, :N//2]; W2 = W[p+N//2, N//2:]
                    D11 = D[p, p]; D12 = D[p, p+N//2]; D21 = D[p+N//2, p]; D22 = D[p+N//2, p+N//2]
                    Dij[0, 0, N//2-1, :] = D11; Dij[0, 1, N//2-1, :] = D12
                    Dij[1, 0, N//2-1, :] = D21; Dij[1, 1, N//2-1, :] = D22

                    configButterfly(W1, Dij[:, :, :N//2, :N//4]); configButterfly(W2, Dij[:, :, :N//2, N//4:])
                    configButterfly(V1, Dij[:, :, N//2:, :N//4]); configButterfly(V2, Dij[:, :, N//2:, N//4:])

                    D = np.block([[np.diag(D11), np.diag(D12)], [np.diag(D21), np.diag(D22)]])
                    V = np.block([[V1, V1*0], [V2*0, V2]]); W = np.block([[W1, W1*0], [W2*0, W2]])
                    err = np.linalg.norm(V @ D @ W - U)
                else:
                    Dij[:, :, 0, 0] = U
            Dij = np.zeros([2, 2, N-1, N//2], dtype=np.complex); configButterfly(M, Dij)

            # Convert the crossing amplitudes Dij into phase shifts (theta, phi).
            p_crossing = self.p_crossing.reshape([N-1, N//2, 2]); phi_out = self.phi_out
            for i in range(N-1):
                s = 2**np.binary_repr(i+1)[::-1].index('1')   # Permutation stride
                (p1, p2) = [(np.outer(1, x) + np.outer(np.arange(0, N, 2*s), 1)).flatten() for x in
                        [np.outer(np.arange(s),1)+np.array([[0,s]]), np.outer(1,np.arange(0,2*s,2))+np.array([[0],[1]])]]
                phi_out[:] = phi_out[p1]
                Dij[:, 0, i, :] *= np.exp(1j*phi_out[::2]); Dij[:, 1, i, :] *= np.exp(1j*phi_out[1::2])
                p_crossing[i] = np.array(self.X.Tsolve((Dij[0, 0, i], Dij[0, 1, i]), 'T1:')[:1])[0].T
                #phi_out[:] = np.angle(Dij[:, :, i]/self.X.T(p_crossing[i]))[:, 0, :].T.flatten()[p2]
                phi_out[:] = (np.angle(Dij[:, :, i]) - np.angle(self.X.T(p_crossing[i])))[:, 0, :].T.flatten()[p2]
