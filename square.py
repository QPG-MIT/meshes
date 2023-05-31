# meshes/square.py
# Ryan Hamerly, 12/10/20
#
# Implements SquareNetwork (subclass of MeshNetwork) with code to handle the SquareNet decomposition.
#
# History
#   11/16/19: Conceived SquareNet, wrote decomposition code (Note3/main.py)
#   07/09/20: Moved to this file.  Created classes SquareNetwork, SquareNetworkMZI.
#   12/10/20: Added Ratio Method tuning strategy for SquareNet and the associated Reck.
#   03/06/21: Harmonized notation with other modules.  Support for fast JIT-ed direct and diagonalization routines.
#   05/30/23: Simplified using the new ClippedNetwork class.

import numpy as np
from scipy.linalg import eigvalsh
from numpy.linalg import cholesky
from typing import Any, Tuple
from .crossing import Crossing, MZICrossing
from .mesh import MeshNetwork, StructuredMeshNetwork, ClippedNetwork
from .configure import diag, direct


class SquareNetwork(ClippedNetwork):
    fact = 1.0

    def __init__(self,
                 p_crossing: Any=0.,
                 phi_out: Any=0.,
                 p_splitter: Any=0.,
                 X: Crossing=MZICrossing(),
                 M: np.ndarray=None,
                 eig: float=None,
                 N: int=None,
                 shape: Tuple[int, int]=None,
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
        assert (N is not None) + (M is not None) + (shape is not None) == 1
        if (M is not None) and (M.dtype != complex): M = M.astype(complex)
        if (shape is None): shape = (N, N) if (N is not None) else M.shape
        out = (phi_pos == 'out')
        p_splitter = np.zeros((np.prod(shape), X.n_splitter)) + np.array(p_splitter)
        p_crossing = np.zeros((np.prod(shape), X.n_phase   )) + np.array(p_crossing)
        phi_out = np.concatenate([np.zeros((shape[1-out],)) + phi_out, np.zeros((shape[out],))])
        shifts = np.abs(np.arange(-shape[1]+1, shape[0])); lens = (sum(shape) - shifts - shifts[::-1])//2

        full = StructuredMeshNetwork(sum(shape), list(lens), list(shifts), p_splitter=p_splitter,
                                     p_crossing=p_crossing, phi_out=phi_out, X=X if out else X.flip(), phi_pos=phi_pos)
        super(SquareNetwork, self).__init__(full, slice(shape[1]), slice(shape[0]))

        if (M is not None):
            if (method in ['diag', 'diag*']):
                diagSquare(self, M, eig, method=='diag*')
            elif (method == 'direct'):
                directSquare(self, M, eig)
            else:
                raise NotImplementedError(method)  # TODO -- program matrix.


def diagSquare(m: SquareNetwork, M: np.ndarray, eig=0.9, improved=True):
    r"""
    Self-configures a SquareNet mesh according to the diagonalization method.
    :param m: Instance of SquareNetwork.
    :param M: Target matrix.
    :param eig: Maximum eigenvector of M*M, after rescaling (set t = None to prevent rescaling)
    :param improved: Whether to use improved diagonalization method (better results for imperfect error correction).
    :return:
    """
    M11 = np.array(M); (M, N) = M11.shape; assert (m.shape == M11.shape); out = (m.phi_pos == 'out')

    if out:
        # Set up U = [[M, M']], where MM* + (M')(M'*) = 1
        MMt = M11.dot(M11.conj().T); fact = np.sqrt(eig / eigvalsh(MMt, subset_by_index=[M-1, M-1])[0]) if eig else 1.0
        MMt *= fact**2; M11 *= fact; M12 = cholesky(np.eye(M) - MMt); m.fact = fact
        U = np.zeros([M+N, M+N], dtype=complex); U[:M, :N] = M11; U[:M, N:] = M12
        # Call the subroutine that self-configures meshes by matrix diagonalization.
        def nn_sq(m): return N
        def ijxyp_sq(m, n): return [m, N+m-n, m+n, N-1+m-n, m*0]
        diag(m.full, None, m.full.phi_out, U, M, nn_sq, ijxyp_sq, improved)
    else:
        # Set up U = [[M], [M']], where M*M + (M'*)(M') = 1
        MtM = M11.T.conj().dot(M11); fact = np.sqrt(eig / eigvalsh(MtM, subset_by_index=[N-1, N-1])[0]) if eig else 1.0
        MtM *= fact**2; M11 *= fact; M21 = cholesky(np.eye(N) - MtM).T.conj(); m.fact = fact
        U = np.zeros([M+N, M+N], dtype=complex); U[:M, :N] = M11; U[M:, :N] = M21
        # Call the subroutine that self-configures meshes by matrix diagonalization.
        def nn_sq(m): return M
        def ijxyp_sq(m, n): return [M+m-n, m, -m-n, M-1-n+m, m*0+1]
        diag(None, m.full, m.full.phi_out, U, N, nn_sq, ijxyp_sq, improved)

def directSquare(m: SquareNetwork, M: np.ndarray, eig=0.9):
    r"""
    Self-configures a SquareNet mesh according to the direct method.
    :param m: Instance of SquareNetwork.
    :param M: Target matrix.
    :param eig: maximum eigenvector of M*M, after rescaling (set t = None to prevent rescaling)
    :return:
    """
    fact = np.sqrt(eig / eigvalsh(M.dot(M.conj().T), subset_by_index=[M.shape[0]-1, M.shape[0]-1])[0]) if eig else 1.0
    m.fact = fact; M = M * fact; U = np.zeros([m.full.N, m.full.N], dtype=complex); U[:M.shape[0], :M.shape[1]] = M
    direct(m.full, U, 'down')
