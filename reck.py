# meshes/reck.py
# Ryan Hamerly, 7/9/20
#
# Implements ReckNetwork (subclass of MeshNetwork) with code to handle the Reck decomposition.
#
# History
#   06/18/20: Defined class ReckNetwork (part of module meshes.py)
#   07/09/20: Moved to this file.

import numpy as np
from .mesh import StructuredMeshNetwork

class ReckNetwork(StructuredMeshNetwork):
    def __init__(self, N: int=0, p_phase=0, p_splitter=0, M=None):
        r"""
        Mesh network based on the Reck (triangular) decomposition.
        :param N: Number of inputs / outputs.
        :param p_phase: Phase parameters (phi_k).  Scalar or vector of size N^2.
        :param p_splitter: Beamsplitter parameters (beta_k - pi/4).  Scalar of vector of size N(N-1).
        :param M: A unitary matrix.  If specified, runs clemdec() to find the mesh realizing this unitary.
        """
        if (M is not None): N = len(M)
        assert (N > 0 and N%2 == 0)
        shifts = list(range(N-2, 0, -1)) + list(range(0, N-1, 1))
        lens = ((N-np.array(shifts))//2).tolist()
        super(ReckNetwork, self).__init__(N, lens, shifts, p_phase, p_splitter)
        if not (M is None):
            self._p0_splitter = self._p0_splitter * np.concatenate([[1]*l + [-1]*l for (i, l) in enumerate(lens)])
            (pars_S, ch_S, phi_out) = reckdec(M); self.p_phase = np.zeros(N**2, dtype=np.float)
            (m0, m, n) = (-2, 0, -1)
            for (i, (ch, (theta, phi))) in enumerate(zip(ch_S, pars_S)):
                if (ch == N-2): m0 += 2; m = m0-1
                m += 1; n = (ch - shifts[m])//2
                self.p_phase[self.inds[2*m] + n] = phi
                self.p_phase[self.inds[2*m+1] + n] = 2*theta
            self.p_phase[-N:] = phi_out
            self.flip_splitter_symmetry()


# Miscellaneous functions and the Reck decomposition.

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

def reckdec(U):
    r"""
    Computes the Reck decomposition of a unitary matrix.  This code is called when instantiating a Reck network from
    a matrix.
    :param U: The unitary matrix.
    :return: A tuple (pars_S, ch_S, pars_Smat, phi_out)
        pars_S: A size-((N-1)*N/2, 2) array.  Each row contains a pair (theta, phi) for each 2x2 block
        ch_S: A size-((N-1)*N/2) vector.  Location of each block.
        pars_Smat: A size-(N, N/2, 2) array.  Gives pairs (theta, phi) ordered on the Clements grid.
        phi_out: Output phases.
    """
    U = np.array(U)
    n = len(U)
    num_Tdag = n*(n-1)//2
    pars_Tdag = np.zeros([num_Tdag, 2], dtype=float)
    ch_Tdag = np.zeros([num_Tdag], dtype=int)
    ind_Tdag = 0
    for i in range(n - 1):
        for j in range(n-2, i-1, -1):
            # U -> U T_{i-j,i-j+1}^dag
            u = U[i, j]
            v = U[i, j+1]
            theta = np.arctan(np.abs(v/u))
            phi = -np.angle(-v/u)
            pars_Tdag[ind_Tdag] = [theta, phi]
            ch_Tdag[ind_Tdag] = j
            ind_Tdag += 1
            U[:, j:j+2] = U[:, j:j+2].dot(_Tdag(theta, phi))
    diag_phi = np.mod(np.array(np.angle(np.diag(U))), 2*np.pi)
    # We can express U = D T1 T2 T3 ..., where T1, ... come from pars_Tdag
    pars_S = pars_Tdag
    ch_S = ch_Tdag
    return (pars_S, ch_S, diag_phi)