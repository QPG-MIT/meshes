# meshes/clements.py
# Ryan Hamerly, 7/9/20
#
# Implements ClementsNetwork (subclass of MeshNetwork) with code to handle the Clements decomposition.
#
# History
#   11/26/19: Wrote code for function clemdec() (part of Note4/meshes.py), Clements decomposition.
#   06/18/20: Defined class ClementsNetwork (part of module meshes.py)
#   07/09/20: Moved to this file.

import numpy as np
from .mesh import StructuredMeshNetwork

class ClementsNetwork(StructuredMeshNetwork):
    def __init__(self, N: int=0, p_phase=0.0, p_splitter=0.0, M=None):
        r"""
        Mesh network based on the Clements (rectangular) decomposition.
        :param N: Number of inputs / outputs.
        :param p_phase: Phase parameters (phi_k).  Scalar or vector of size N^2.
        :param p_splitter: Beamsplitter parameters (beta_k - pi/4).  Scalar of vector of size N(N-1).
        or
        :param M: A unitary matrix.  If specified, runs clemdec() to find the mesh realizing this unitary.
        """
        if (M is not None): N = len(M)
        assert (N > 0 and N%2 == 0)
        lens = [N//2, N//2-1]*(N//2)
        shifts = [0, 1]*(N//2)
        super(ClementsNetwork, self).__init__(N, lens, shifts, p_phase, p_splitter)
        if not (M is None):
            # First call clemdec(M).  Then need to rearrange the data to fit into self.p_phase.
            self._p0_splitter = self._p0_splitter * np.concatenate([[1]*l + [-1]*l for (i, l) in enumerate(lens)])
            (pars_S, ch_S, pars_Smat, phi_out) = clemdec(M)
            pars_Smat[:, :, 0] *= 2   # theta -> 2*theta (actual phase of the element)
            self.p_phase = np.concatenate([pars_Smat[i,:lens[i],::-1].T.flatten() for i in range(self.N)] + [phi_out])
            self.flip_splitter_symmetry()


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

def clemdec(U):
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
    n_even = (n - 1) // 2 * 2
    n_odd = n // 2 * 2 - 1
    num_Tdag = ((n_odd + 1) // 2) ** 2
    num_T = n_even * (n_even + 2) // 4
    pars_Tdag = np.zeros([num_Tdag, 2], dtype=float)
    pars_T = np.zeros([num_T, 2], dtype=float)
    ch_Tdag = np.zeros([num_Tdag], dtype=int)
    ch_T = np.zeros([num_T], dtype=int)
    ind_Tdag = 0
    ind_T = 0
    for i in range(n - 1):
        for j in range(i + 1):
            if (i % 2 == 0):
                # U -> U T_{i-j,i-j+1}^dag
                u = U[n - 1 - j, i - j]
                v = U[n - 1 - j, i - j + 1]
                theta = np.arctan(abs(u / v))
                phi = np.angle(u / v)
                pars_Tdag[ind_Tdag] = [theta, phi]
                ch_Tdag[ind_Tdag] = i - j
                ind_Tdag += 1
                U[:, i - j:i - j + 2] = U[:, i - j:i - j + 2].dot(_Tdag(theta, phi))
            else:
                # U -> T_{i-j,i-j+1} U
                u = U[n - 1 - i + j, j]
                v = U[n - 2 - i + j, j]
                theta = np.arctan(abs(u / v))
                phi = np.angle(-u / v)
                pars_T[ind_T] = [theta, phi]
                ch_T[ind_T] = n - 2 - i + j
                ind_T += 1
                U[n - 2 - i + j:n - i + j, :] = _T(theta, phi).dot(U[n - 2 - i + j:n - i + j, :])
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
    for (ind_S, ((theta, phi), i)) in enumerate(zip(pars_T[::-1], ch_T[::-1])):
        (psi1, psi2) = diag_phi2[i:i + 2]
        ch_S[num_Tdag + ind_S] = i
        pars_S[num_Tdag + ind_S] = (-theta, psi1 - psi2)
        diag_phi2[i:i + 2] = (psi2 - phi, psi2)
    # Convert to a 2D matrix
    w = n // 2
    d = n
    ind = 0
    pars_Smat = np.zeros([d, w, 2], dtype=float)
    for j0 in range(0, 2 * n - 2, 2):
        for i in range(n):
            j = j0 - i
            if (j >= 0 and j < n - 1):
                pars_Smat[i, j // 2] = pars_S[ind]
                ind += 1
    return (pars_S, ch_S, pars_Smat, np.mod(diag_phi2, 2 * np.pi))
