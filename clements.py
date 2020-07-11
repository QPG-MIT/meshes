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

import numpy as np
import warnings
from typing import Any
from .mesh import StructuredMeshNetwork
from .crossing import Crossing, MZICrossing

class ClementsNetwork(StructuredMeshNetwork):
    def __init__(self,
                 p_crossing: Any=0.,
                 phi_out: Any=0.,
                 p_splitter: Any=0.,
                 X: Crossing=MZICrossing(),
                 M: np.ndarray=None,
                 warn=True):
        r"""
        Mesh network based on the Reck (triangular) decomposition.
        :param N: Number of inputs / outputs.
        :param p_phase: Phase parameters (phi_k).  Scalar or vector of size N^2.
        :param p_splitter: Beamsplitter imperfection parameters.  Scalar or size (N(N-1)/2, X.n_splitter).
        :param M: A unitary matrix.  If specified, runs clemdec() to find the mesh realizing this unitary.
        """
        N = len(phi_out if (M is None) else M)   # Start by getting N, shifts, lens, p_splitter
        lens = list((N - np.arange(N) % 2)//2)
        shifts = list(np.arange(N) % 2)
        p_splitter = np.array(p_splitter); assert p_splitter.shape in [(), (N*(N-1)//2, X.n_splitter)]
        if (M is None):
            # Initialize from parameters.  Check parameters first.
            assert p_crossing.shape == (N*(N-1)//2, X.n_phase) and phi_out.shape == (N,)
        else:
            # Initialize from a matrix.  Calls clemdec() after correctly ordering the crossings.
            (pars_S, ch_S, pars_Smat, phi_out) = clemdec(M, p_splitter, X, warn)  # <-- The hard work is done here.
            p_crossing = (pars_Smat if N%2 else
                          pars_Smat.reshape([N//2, N, X.n_phase])[:, :-1, :]).reshape(N*(N-1)//2, X.n_phase)
        super(ClementsNetwork, self).__init__(N, lens, shifts, p_splitter=p_splitter,
                                              p_crossing=p_crossing, phi_out=phi_out, X=X)


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
