# meshes/reck.py
# Ryan Hamerly, 7/11/20
#
# Implements ReckNetwork (subclass of MeshNetwork) with code to handle the Reck decomposition.
#
# History
#   06/18/20: Defined class ReckNetwork (part of module meshes.py)
#   07/09/20: Moved to this file.
#   07/10/20: Added compatibility with custom crossings in crossing.py
#   03/06/21: Added compatibility with diagonalization method, JIT-ed routine for direct method.

import numpy as np
import warnings
from typing import Any
from .mesh import StructuredMeshNetwork, calibrateTriangle
from .crossing import Crossing, MZICrossing
from .configure import diag, direct

class ReckNetwork(StructuredMeshNetwork):
    def __init__(self,
                 p_crossing: Any=0.,
                 phi_out: Any=0.,
                 p_splitter: Any=0.,
                 X: Crossing=MZICrossing(),
                 M: np.ndarray=None,
                 N: int=None,
                 method: str='diag',
                 warn=False,
                 phi_pos='out'):
        r"""
        Mesh network based on the Reck (triangular) decomposition.
        :param N: Number of inputs / outputs.
        :param p_phase: Phase parameters (phi_k).  Scalar or vector of size N^2.
        :param p_splitter: Beamsplitter imperfection parameters.  Scalar or size (N(N-1)/2, X.n_splitter).
        :param M: A unitary matrix.  If specified, runs clemdec() to find the mesh realizing this unitary.
        :param N: Size.  Used for initializing a blank Reck mesh.
        :param method: Method used to program the Reck mesh in presence of errors: 'direct' or 'ratio'.
        :param phi_pos: Position of phase shifts: 'in' or 'out'.
        """
        assert (M is None) ^ (N is None)
        if (M is not None): N = len(M)  # Start by getting N, shifts, lens, p_splitter
        elif (np.iterable(phi_out)): N = len(phi_out)
        else: assert N != None
        shifts = list(range(N-2, 0, -1)) + list(range(0, N-1, 1))
        lens = ((N-np.array(shifts))//2).tolist()
        p_splitter = np.array(p_splitter); assert p_splitter.shape in [(), (N*(N-1)//2, X.n_splitter)]
        if (method is None):
            if (M is None):
                # Initialize from parameters.  Check parameters first.
                p_crossing = p_crossing * np.ones([N*(N-1)//2, X.n_phase]); phi_out = phi_out * np.ones(N)
            else:
                # Initialize from a matrix.  Calls reckdec() after correctly ordering the crossings.
                if p_splitter.ndim: p_splitter = reorder_reck(N, p_splitter, True)
                (pars_S, ch_S, phi_out) = reckdec(M, p_splitter, X, warn)  # <-- The hard work is all done in here.
                p_crossing = reorder_reck(N, pars_S)
                if p_splitter.ndim: p_splitter = reorder_reck(N, p_splitter)
            super(ReckNetwork, self).__init__(N, lens, shifts, p_splitter=p_splitter,
                                              p_crossing=p_crossing, phi_out=phi_out, X=X)
            if phi_pos == 'in': self.flip_crossings(inplace=True)
        elif (method == 'diag'):
            super(ReckNetwork, self).__init__(N, lens, shifts, p_splitter=p_splitter,
                  p_crossing=np.zeros([N*(N-1)//2, X.n_phase]), phi_out=np.zeros(N),
                  X=X.flip() if (phi_pos == 'in') else X, phi_pos=phi_pos)
            diagReck(self, M)
        elif (method == 'direct'):
            assert (phi_pos == 'in')
            super(ReckNetwork, self).__init__(N, lens, shifts, p_splitter=p_splitter,
                  p_crossing=np.zeros([N*(N-1)//2, X.n_phase]), phi_out=np.zeros(N),
                  X=X.flip() if (phi_pos == 'in') else X, phi_pos=phi_pos)
            direct(self, M, 'down')
        else:
            assert phi_pos == 'in'
            super(ReckNetwork, self).__init__(N, lens, shifts, p_splitter=p_splitter,
                  p_crossing=np.zeros([N*(N-1)//2, X.n_phase]), phi_out=np.zeros(N), X=X.flip(), phi_pos='in')
            calibrateTriangle(self, M, 'down', method, warn)



def reckdec(U: np.ndarray, p_splitter: Any=0., X: Crossing=MZICrossing(), warn=True):
    r"""
    Computes the Reck decomposition of a unitary matrix.  This code is called when instantiating a Reck network from
    a matrix.
    :param U: The unitary matrix.
    :param p_splitter: Splitter imperfections.  A size-(N*(N-1)/2, X.n_splitter) matrix.
    :param X: Crossing class.
    :param warn: Issues warnings if the matrix could not be perfectly realized.
    :return: A tuple (pars_S, ch_S, pars_Smat, phi_out)
        pars_S: A size-((N-1)*N/2, 2) array.  Each row contains a pair (theta, phi) for each 2x2 block
        ch_S: A size-((N-1)*N/2) vector.  Location of each block.
        pars_Smat: A size-(N, N/2, 2) array.  Gives pairs (theta, phi) ordered on the Clements grid.
        phi_out: Output phases.
    """
    U = np.array(U)
    n = len(U)
    num_Tdag = n*(n-1)//2
    p_splitter = np.ones([num_Tdag, X.n_splitter]) * p_splitter
    pars_Tdag = np.zeros([num_Tdag, X.n_phase], dtype=float)
    ch_Tdag = np.zeros([num_Tdag], dtype=int)
    ind_Tdag = 0
    err = 0
    for i in range(n - 1):
        for j in range(n-2, i-1, -1):
            # U -> U T_{i-j,i-j+1}^dag
            u = U[i, j]
            v = U[i, j+1]
            beta_i = p_splitter[ind_Tdag]
            (pars_i, err_i) = X.Tsolve((np.conj(v), -np.conj(u)), 'T2:', beta_i)
            pars_Tdag[ind_Tdag] = pars_i
            ch_Tdag[ind_Tdag] = j
            ind_Tdag += 1
            err += err_i
            U[:, j:j+2] = U[:, j:j+2].dot(X.Tdag(pars_i, beta_i))
    diag_phi = np.mod(np.array(np.angle(np.diag(U))), 2*np.pi)
    # We can express U = D T1 T2 T3 ..., where T1, ... come from pars_Tdag
    pars_S = pars_Tdag
    ch_S = ch_Tdag
    if (warn and err):
        warnings.warn(
            "Reck calibration: {:d}/{:d} beam splitters could not be set correctly.".format(err, n*(n-1)//2))
    return (pars_S, ch_S, diag_phi)

def reorder_reck(N: int, data: np.ndarray, reverse=False) -> np.ndarray:
    r"""
    Used to reorder blocks because reckdec()'s output arranges them along rising diagonals, top to bottom.
    Meanwhile, StructuredMeshNetwork orders them in columns for uniformity and computational efficiency.
    :param N: Size of Reck network (number of channels).
    :param data: Array of size (N(N-1)/2, p)
    :param reverse: If False, start with data ordered along diagonals, reorder them in columns.
    If True, start with data ordered along columns, reorder them in diagonals.
    :return: Array of size (N(N-1)/2, p)
    """
    if data.ndim == 0: return data
    assert len(data) == N*(N-1)//2
    inds = np.cumsum(np.pad((N-(np.abs(np.array(range(-N+2, N-1)))))//2, (1, 0)))
    out = data*0; ind_in = 0
    for i in range(N-1):
        for j in range(N-1-i):
            ind_out = inds[j+2*i]+min(i, N-2-i-j)
            if reverse: out[ind_in ] = data[ind_out]
            else:       out[ind_out] = data[ind_in ]
            ind_in += 1
    return out

def diagReck(m: ReckNetwork, U: np.ndarray):
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

