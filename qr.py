# meshes/diag.py
# Ryan Hamerly, 5/30/23
#
# Implements the triangular QR mesh, which encodes a general NxN matrix using a QR decomposition.
#
# History
#   03/06/21: Wrote class for QR mesh and self-configuration code for the matrix diagonalization method.
#   05/30/23: Simplified using the new ClippedNetwork class.

import numpy as np
from .configure import diag
from .crossing import Crossing, MZICrossing
from .mesh import MeshNetwork, ClippedNetwork, StructuredMeshNetwork
from typing import Any
from scipy.linalg import eigvalsh
from numpy.linalg import cholesky, qr   # Don't work if imported from SciPy.  Weird that SciPy & NumPy versions differ.


class QRNetwork(ClippedNetwork):
    fact = 1.0

    def __init__(self,
                 p_crossing: Any=0.,
                 phi_out: Any=0.,
                 p_splitter: Any=0.,
                 X: Crossing=MZICrossing(),
                 M: np.ndarray=None,
                 eig: float=None,
                 N: int=None,
                 method: str='diag',
                 phi_pos='out'):
        r"""
        Mesh network based on the SquareNet (diamond) geometry.
        :param N: Number of inputs / outputs.
        :param p_crossing: Crossing parameters (theta, phi).  Scalar of vector of size (N^2, X.n_phase)
        :param phi_out: External phase shifts.  Scalar or vector of size (2*N).
        :param p_phase: Crossing and external phase parameters, size (N^2*X.n_phase + 2*N)
        :param p_splitter: Beamsplitter imperfection parameters.  Scalar or size (N(N-1)/2, X.n_splitter).
        :param M: A unitary matrix.  If specified, runs clemdec() to find the mesh realizing this unitary.
        :param eig: If specified, normalizes M so that max_eigval(M*M) = eig.
        :param N: Size.  Used for initializing a blank Reck mesh.
        :param method: Method used to program the Reck mesh in presence of errors: 'direct' or 'ratio'.
        :param phi_pos: Position of phase shifts: 'in' or 'out'.
        """
        assert (N is not None) ^ (M is not None)
        if (M is not None): 
            if (M.dtype != complex): M = M.astype(complex)
            N = len(M); assert (M.shape == (N, N))
        out = (phi_pos == 'out')
        p_splitter = np.zeros((N**2, X.n_splitter)) + np.array(p_splitter)
        p_crossing = np.zeros((N**2, X.n_phase   )) + np.array(p_crossing)
        phi_out_f = np.zeros([2*N]); phi_out_f[slice(N, None) if out else slice(None, None, 2)] = phi_out

        shifts = np.arange(2*N-1); lens = (2*N-shifts)//2

        full = StructuredMeshNetwork(2*N, list(lens), list(shifts), p_splitter=p_splitter,
                                  p_crossing=p_crossing, phi_out=phi_out_f, X=X if out else X.flip(), phi_pos=phi_pos)
        super(QRNetwork, self).__init__(full, slice(None, None, 2), slice(N, None, None))

        if (M is not None):
            if (method in ['diag', 'diag*']):
                diagQR(self, M, eig, method == 'diag*')
            else:
                raise NotImplementedError(method)  # TODO -- program matrix.


def diagQR(m: QRNetwork, M: np.ndarray, eig: float=None, improved=False):
    r"""
    Self-configures a QR mesh according to the diagonalization method.
    :param m: Instance of QRNetwork.
    :param M: Target matrix.
    :param eig: maximum eigenvector of M*M, after rescaling (set t = None to prevent rescaling)
    :return:
    """
    # Stuff for QR decomposition mesh.  Embed M into a larger unitary matrix U.
    N = len(M); assert M.shape == (N, N)
    MtM = M.T.conj().dot(M); fact = m.fact = np.sqrt(eig / eigvalsh(MtM, subset_by_index=[N-1, N-1])[0]) if eig else 1.0
    MtM *= fact**2; M = M * fact; L = cholesky(np.eye(N) - MtM[::-1,::-1])[::-1,::-1].T.conj()
    M1 = np.concatenate([L, M], 0); M2 = np.random.randn(2*N, N)
    M2 = qr(qr(np.concatenate([M1, M2], 1))[0][:, N:].T)[1].T
    U = np.zeros([2*N, 2*N], dtype=complex); U[:, ::2] = M1; U[:, 1::2] = M2

    if (m.phi_pos == 'out'):
        def nn_qr(m): return min(m+1, 2*N-(m+1))
        def ijxyp_qr(m, n): return [m, min(2*m[0]+1, 2*N-1)-n, n+max(0, 2*(m[0]-N+1)), min(2*N-2,2*m[0])-n, m*0]
        diag(m.full, None, m.full.phi_out, U, 2*N-1, nn_qr, ijxyp_qr, improved)
    else:
        def nn_qr(m): return 2*(N-m)-1
        def ijxyp_qr(m, n): return [2*N-1-n, 2*m, -n-2*m, 2*N-2-n, m*0+1]
        diag(None, m.full, m.full.phi_out, U, N, nn_qr, ijxyp_qr, improved)
