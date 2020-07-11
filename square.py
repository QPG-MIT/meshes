# meshes/square.py
# Ryan Hamerly, 7/11/20
#
# Implements SquareNetwork (subclass of MeshNetwork) with code to handle the SquareNet decomposition.
#
# History
#   11/16/19: Conceived SquareNet, wrote decomposition code (Note3/main.py)
#   07/09/20: Moved to this file.  Created classes SquareNetwork, SquareNetworkMZI.

import numpy as np
import warnings
from typing import Any
from .crossing import Crossing, MZICrossing
from .mesh import MeshNetwork

def calibrateDiag(Acol, Tpost, p_splitter: Any=0., X: Crossing=MZICrossing(), errout=False):
    r"""
    Gets p_phase for the next (falling) diagonal of SquareNet.  Does so by sending a signal 1 into the input and
    going down the diagonal, one MZI at a time, to match the complex amplitude of each output field.
    :param Acol: Desired column of the output matrix A.
    :param Tpost: Transfer matrix from all subsequent columns
    :param p_splitter: Splitter errors or other manufacturing imperfections.
    :param X: Crossing class.
    :param errout: Whether to output the error state.
    :return: (p_phase, err) if errout else (p_phase)
    """
    N = len(Acol); assert (Acol.shape == (N,) and Tpost.shape == (N,N))
    p_splitter = np.ones([N, X.n_splitter]) * p_splitter
    p_phase = np.zeros([N, X.n_phase], dtype=np.float)
    # x_LO: LO input.  y_LO: output from LO.  y_V: output due to fields v[1:k-1]
    # u_k, w_k: intermediate input / output of k'th cell.  See CodeFig1.
    y_V = np.zeros(N, dtype=np.complex); u_k = 1.; err = 0
    # Loop over all elements in the diagonal.
    for k in range(N):
        # The output is: y[k] = yk_avg + r * yk_diff.  Depends on value of (complex) reflection r
        yk_avg = y_V[k]
        yk_diff = Tpost[k, k] * u_k
        # Get z = i*sin(theta)*e^{i(phi+theta)}, then get (theta, phi).
        if (yk_diff == 0):
            (theta, phi) = (0, 0); err += 1
        else:
            z = (Acol[k] - yk_avg) / yk_diff
            (p_phase[k], d_err) = X.Tsolve(z, 0, p_splitter[k]); err += d_err
        # Update u_k, y_V for the next element.  Save (theta, phi).
        T = X.T(p_phase[k], p_splitter[k])
        y_V += Tpost[:, k] * T[0, 0] * u_k
        u_k *= T[1, 0]
    return (p_phase, err) if errout else (p_phase)

# Calibrates the phases needed to realize SquareNet.
def squaredec(mat, p_splitter: Any=0., X: Crossing=MZICrossing(), warn=True):
    r"""
    Performs the SquareNet decomposition.
    :param mat: Input-output matrix.  Can be rectangular or complex-valued.  SquareNet can only realize matrices with
    |A| ≤ 1.
    :param p_splitter: Splitter errors or other manufacturing imperfections.
    :param X: Crossing class.
    :param warn: Issues warnings if the matrix could not be perfectly realized.
    :return: p_phase, an array with size=(M, N, X.n_phase)
    """
    (N, M) = mat.shape; Tv = np.eye(N, N+1, dtype=np.complex)
    p_splitter = p_splitter * np.ones([M, N, X.n_splitter])
    p_phase = np.zeros([M, N, X.n_phase], dtype=np.float)
    theta = np.zeros([M, N], dtype=np.float); phi = np.zeros([M, N], dtype=np.float)
    # Loop over diagonals top to bottom.  Calibrate each diagonal, then right-multiply T_k
    # to get the output transfer matrix (same method as in T_square).
    err = 0
    for i in range(M):
        (p_phase[i], err_i) = calibrateDiag(mat[:, i], Tv[:, :N], p_splitter[i], X, True)
        err += err_i
        X.rmult_falling(Tv, p_phase[i], p_splitter[i])
        Tv[:, :N] = Tv[:, 1:]
        Tv[:, N] = 0
    if (warn and err):
        warnings.warn(
            "SquareNet calibration: {:d}/{:d} matrix values could not be set.".format(err, M*N))
    return p_phase

# Gives the SquareNet transfer matrix A, where y[N] = A[N*M] x[M].
# Inputs (theta, phi) are M*N arrays.  The rows are the diagonals, ordered from top to bottom.
# If aux = True, returns (A, B), where y[N] = A[N*M] x[M] + B[N*N] x'[N]
def squaremat(p_phase, p_splitter: Any=0., X: Crossing=MZICrossing(), aux=False):
    (M, N) = p_phase.shape[:-1]
    p_splitter = p_splitter * np.ones([M, N, X.n_splitter])
    Tv = np.eye(N, N+1, dtype=np.complex)
    A = np.zeros([N, M], dtype=np.complex)
    # For each diagonal k, right-multiply [T_1...T_{k-1} | 0] * T_k = [A_k | T_1...T_k].
    # Save first column to A, shift the columns over, and zero last column to get [T_1...T_k | 0]
    for i in range(M):
        X.rmult_falling(Tv, p_phase[i], p_splitter[i])
        A[:, i] = Tv[:, 0]
        Tv[:, :N] = Tv[:, 1:]
        Tv[:, N] = 0
    if (aux):
        return (A, np.array(Tv[:, :N]))
    else:
        return A

class SquareNetwork(MeshNetwork):
    _M: int
    _N: int
    _X: Crossing

    @property
    def L(self):
        return self._M + self._N - 1
    @property
    def M(self):
        return self._M
    @property
    def N(self):
        return self._N
    @property
    def X(self):
        r"""
        Crossing class object.  Default: MZI crossing.
        """
        return self._X
    @property
    def n_phase(self):
        r"""
        Number of programmable degrees of freedom for crossing (default=2 for MZI).
        """
        return self.X.n_phase
    @property
    def n_splitter(self):
        r"""
        Number of splitter imperfection degrees of freedom for crossing (default=2 for MZI).
        """
        return self.X.n_splitter

    def __init__(self, p_phase=0.0, p_splitter=0.0, X=MZICrossing(), M=None):
        r"""
        Mesh network based on SquareNet decomposition.  SquareNet can represent an arbitrary rectangular matrix up to a
        scaling factor.  For an N*M matrix, the inputs are:
        :param p_phase: Crossing degrees of freedom.  Scalar or size-(M, N, X.n_phase) tensor.
        :param p_splitter: Crossing imperfections.  Scalar or size-(M, N, X.n_splitter) tensor.
        :param M: The matrix to be represented.  Real or complex, size-(N, M).  If specified, runs squaredec() to find
        the SquareNet decomposition, and any information in p_phase is overwritten.
        """
        self._X = X; self.p_phase = np.array(p_phase); self.p_splitter = np.array(p_splitter)
        (self._M, self._N) = M.shape if not (M is None) else self.p_phase.shape[:-1]
        if not (M is None):
            self.p_phase = squaredec(M, self.p_splitter, self.X)  # <-- Where all the hard work is done
        assert self.p_phase.shape in [(), (self.M, self.N, self.n_phase)]
        assert self.p_splitter.shape in [(), (self.M, self.N, self.n_splitter)]

    def matrix(self, p_phase=None, p_splitter=None, aux=False):
        r"""
        Computes the input-output matrix.  Equivalent to self.dot(np.eye(self.N))
        Gives the SquareNet transfer matrix A, where y[N] = A[N*M] x[M].
        Inputs (p_phase, p_splitter) are M*N*2 arrays.  The rows are the falling diagonals, ordered from top to bottom.
        If aux = True, returns (A, B), where y[N] = A[N*M] x[M] + B[N*N] x'[N]
        :param p_phase: Phase parameters.  Defaults to stored values.
        :param p_splitter: Splitter angle parameters (deviation from pi/4).  Defaults to stored values.
        :param aux: Whether to return the auxiliary matrix B.
        :return: (A, B) if aux=True else A
        """
        if p_phase is None: p_phase = self.p_phase
        if p_splitter is None: p_splitter = self.p_splitter
        return squaremat(p_phase, p_splitter, self.X, aux)

class SquareNetworkMZI(SquareNetwork):
    @property
    def theta(self):
        return self.p_phase[:, :, 0] if self.p_phase.ndim else self.p_phase
    @property
    def phi(self):
        return self.p_phase[:, :, 1] if self.p_phase.ndim else self.p_phase

    def __init__(self, p_phase=0.0, p_splitter=0.0, M=None):
        r"""
        SquareNet based on lossless MZI crossings.
        :param p_phase: Phase parameters (phi_k, theta_k).  Scalar or size-(m, n, 2) tensor.
        :param p_splitter: Beamsplitter parameters (beta_k - pi/4).  Scalar or size-(m, n, 2) tensor.
        :param A: The matrix to be represented.  If specified, runs squaredec() to find the SquareNet decomposition,
        and any information in p_phase is overwritten.
        """
        super(SquareNetworkMZI, self).__init__(p_phase, p_splitter, X=MZICrossing(), M=M)