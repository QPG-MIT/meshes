# meshes/diag.py
# Ryan Hamerly, 3/6/21
#
# Implements the triangular QR mesh, which encodes a general NxN matrix using a QR decomposition.
#
# History
#   03/06/21: Wrote class for QR mesh and self-configuration code for the matrix diagonalization method.

import numpy as np
from .configure import diag
from .crossing import Crossing, MZICrossing
from .mesh import MeshNetwork, StructuredMeshNetwork
from typing import Any
from scipy.linalg import eigvalsh
from numpy.linalg import cholesky, qr   # Don't work if imported from SciPy.  Weird that SciPy & NumPy versions differ.

class QRNetwork(MeshNetwork):
    full: StructuredMeshNetwork
    phi_pos: str
    _N: int
    _out: bool
    X: Crossing

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
            if (M.dtype != np.complex): M = M.astype(np.complex)
            N = len(M); assert (M.shape == (N, N))
        out = (phi_pos == 'out')
        p_splitter = np.zeros((N**2, X.n_splitter)) + np.array(p_splitter)
        p_crossing = np.zeros((N**2, X.n_phase   )) + np.array(p_crossing)
        phi_out_f = np.zeros([2*N]); phi_out_f[slice(N, None) if out else slice(None, None, 2)] = phi_out

        shifts = np.arange(2*N-1); lens = (2*N-shifts)//2

        super(QRNetwork, self).__init__()
        self.full = StructuredMeshNetwork(2*N, list(lens), list(shifts), p_splitter=p_splitter,
                                  p_crossing=p_crossing, phi_out=phi_out_f, X=X if out else X.flip(), phi_pos=phi_pos)
        self._out = out; self._N = N
        #self.p_phase = Doesn't fit nicely into a slice.  TODO -- think about this.
        self.p_splitter = self.full.p_splitter
        self.phi_pos = phi_pos
        self.X = self.full.X

        if (M is not None):
            if (method == 'diag'):
                diagQR(self, M, eig)
            else:
                raise NotImplementedError(method)  # TODO -- program matrix.

    @property
    def L(self) -> int:
        return 2*self.N - 1
    @property
    def M(self) -> int:
        return self._N
    @property
    def N(self) -> int:
        return self._N
    @property
    def p_crossing(self) -> np.ndarray:
        return self.full.p_crossing
    @p_crossing.setter
    def p_crossing(self, x):
        self.p_crossing[:] = x
    @property
    def phi_out(self) -> np.ndarray:
        return self.full.phi_out[slice(self.N, None) if self._out else slice(None, None, 2)]
    @phi_out.setter
    def phi_out(self, x):
        self.phi_out[:] = x

    def _fullphase(self, p_phase):
        if p_phase is None: return None
        out = np.zeros([self.full.p_phase.shape]); p_phase = np.array(p_phase); N = self.N
        if (p_phase.ndim):
            out[:-2*N] = p_phase[:-N]; out[slice(-N, None) if self._out else slice(-2*N, None, 2)] = p_phase[-N:]
        else:
            out[:] = p_phase
        return out

    def dot(self, v, p_phase=None, p_splitter=None) -> np.ndarray:
        r"""
        Computes the dot product between the splitter and a vector v.
        :param v: Input vector / matrix.
        :param p_phase: Phase parameters.  Defaults to stored values.
        :param p_splitter: Splitter angle parameters (deviation from pi/4).  Defaults to stored values.
        :return: Output vector / matrix.
        """
        shape = list(v.shape); shape[0] *= 2; v2 = np.zeros(shape, dtype=np.complex); v2[::2] = v
        w = self.full.dot(v2, p_phase=self._fullphase(p_phase), p_splitter=p_splitter)
        return np.array(w[self.N:])


def diagQR(m: QRNetwork, M: np.ndarray, eig: float=None):
    r"""
    Self-configures a QR mesh according to the diagonalization method.
    :param m: Instance of QRNetwork.
    :param M: Target matrix.
    :param eig: maximum eigenvector of M*M, after rescaling (set t = None to prevent rescaling)
    :return:
    """
    # Stuff for QR decomposition mesh.  Embed M into a larger unitary matrix U.
    N = len(M); assert M.shape == (N, N)
    MtM = M.T.conj().dot(M); fact = np.sqrt(eig / eigvalsh(MtM, subset_by_index=[N-1, N-1])[0]) if eig else 1.0
    MtM *= fact**2; M *= fact; L = cholesky(np.eye(N) - MtM[::-1,::-1])[::-1,::-1].T.conj()
    M1 = np.concatenate([L, M], 0); M2 = np.random.randn(2*N, N)
    M2 = qr(qr(np.concatenate([M1, M2], 1))[0][:, N:].T)[1].T
    U = np.zeros([2*N, 2*N], dtype=np.complex); U[:, ::2] = M1; U[:, 1::2] = M2

    if (m.phi_pos == 'out'):
        def nn_qr(m): return min(m+1, 2*N-(m+1))
        def ijxyp_qr(m, n): return [m, min(2*m[0]+1, 2*N-1)-n, n+max(0, 2*(m[0]-N+1)), min(2*N-2,2*m[0])-n, m*0]
        diag(m.full, None, m.full.phi_out, U, 2*N-1, nn_qr, ijxyp_qr)
    else:
        def nn_qr(m): return 2*(N-m)-1
        def ijxyp_qr(m, n): return [2*N-1-n, 2*m, -n-2*m, 2*N-2-n, m*0+1]
        diag(None, m.full, m.full.phi_out, U, N, nn_qr, ijxyp_qr)

