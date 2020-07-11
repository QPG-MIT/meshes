# meshes/crossing.py
# Ryan Hamerly, 7/9/20
#
# Implements a generic class to handle 2x2 crossings, and its most common implementation (MZI).  Handling this
# functionality in a separate class allows one to easily switch between different crossing types (MZI, ring,
# double-ring, ring-assisted MZI, lossy MZI, etc.) without rewriting the main code.
#
# History
#   07/09/20: Created this file.  Defined classes Crossing, MZICrossing.
#   07/10/20: Added clemshift() functionality (convert T(p)* psi -> psi' T(p') for Clements decomposition).

import numpy as np
from typing import Any

class Crossing:
    @property
    def n_phase(self) -> int:
        raise NotImplementedError()
    @property
    def n_splitter(self) -> int:
        raise NotImplementedError()

    def T(self, p_phase, p_splitter: Any=0.) -> np.ndarray:
        r"""
        Obtains the 2x2 matrix for the crossing.
        :param p_phase: Phase-shifter (or other d.o.f.) information.  Size (n_phase,) or (k, n_phase)
        :param p_splitter: Splitter errors / imperfections.  Size (n_splitter,) or (k, n_splitter)
        :return: Array of size (2, 2) or (2, 2, k)
        """
        raise NotImplementedError()
    def dT(self, p_phase, p_splitter: Any=0.) -> np.ndarray:
        r"""
        Obtains the derivatives of T for a crossing.
        :param p_phase: Phase shifter (or other d.o.f.) information.  Size (n_phase,) or (k, n_phase)
        :param p_splitter: Splitter errors / imperfections.  Size (n_splitter,) or (k, n_splitter)
        :return: Array of size (n_phase, 2, 2) or (n_phase, 2, 2, k)
        """
        raise NotImplementedError()
    def Tsolve(self, T, ind, p_splitter: Any=0.):
        r"""
        Finds the value of p_phase that sets a certain matrix element to the target T.
        ind=0: targets T[0, 0]
        ind=1: targets T[1, 0]
        TODO implement T[1, 0]/T[0, 0] or something like that...
        TODO implement T[0, 1], T[1, 1].
        :param T: Target value.
        :param ind: Index of the matrix element.
        :param p_splitter: Splitter errors or other manufacturing imperfections.
        :return: (p_phase, err), where err=1 if the target was unreachable.
        """
        raise NotImplementedError()
    def clemshift(self, phi, p_phase, p_splitter: Any=0):
        r"""
        Performs the phase-shifter identity employed in the Clements decomposition: Tdag(p) * PS -> PS' * T(p')
        :param phi: Phase shifts [PS].  Length-2 array.
        :param p_phase: Crossing degrees of freedom [p].  Length-(n_phase) array.
        :param p_splitter: Crossing imperfections.  Length-(n_splitter) array.
        :return: A pair (p', PS') containing new parameters and shifted phases.
        """
        raise NotImplementedError()


    def Tdag(self, p_phase, p_splitter: Any=0.) -> np.ndarray:
        r"""
        Obtains the T.conj().transpose()
        :param p_phase: Phase-shifter (or other d.o.f.) information.
        :param p_splitter: Splitter errors or other manufacturing imperfections.
        :return:
        """
        T = self.T(p_phase, p_splitter)
        return T.conj().transpose((1, 0, 2) if T.ndim == 3 else (1, 0))
    def rmult_falling(self, M, p_phase, p_splitter: Any=0.):
        r"""
        Right-multiplies by matrix T for a falling diagonal, i.e. M -> M T.
        T = T_n T_{n-1} ... T_1.  So we right-multiply in reverse order.
        :param M: Matrix to be multiplied.
        :param p_phase: Phase-shifter (or other d.o.f.) information.
        :param p_splitter: Splitter errors or other manufacturing imperfections.
        :return:
        """
        N = p_phase.shape[0]; assert (N == M.shape[1]-1); p_splitter = p_splitter * np.ones((N, self.n_splitter))
        for (i, p_phase_i, p_splitter_i) in zip(range(N)[::-1], p_phase[::-1], p_splitter[::-1]):
            M[:, i:i+2] = M[:, i:i+2].dot(self.T(p_phase_i, p_splitter_i))
        return M

    def dot(self, p_phase, p_splitter, x, dag=False):
        r"""
        Performs the dot product T*x.  Can be used for a single splitter or an array.
        :param p_phase: Array of size (n_phase,) or (M/2, n_phase)
        :param p_splitter: Array of size (n_splitter,) or (M/2, n_splitter)
        :param x: Array of size (M,) or (M, N)
        :param dag: If True, use the Hermitian conjugate (T^dagger).
        :return: Array of size (M,) or (M, N)
        """
        if (x.ndim == 1):
            return self.dot(p_phase, p_splitter, np.outer(x, 1))[:, 0]
        (m, n) = x.shape; T = (self.Tdag if dag else self.T)(p_phase, p_splitter).reshape([2,2,m//2]).transpose((2,0,1))
        return (T.reshape(T.shape + (1,)) * x.reshape([m//2, 1, 2, n])).sum(axis=2).reshape(x.shape)

    def rdot(self, p_phase, p_splitter, y, dag=False):
        r"""
        Performs the right dot product y*T.  Can be used for a single splitter or an array.
        :param p_phase: Array of size (n_phase,) or (M/2, n_phase)
        :param p_splitter: Array of size (n_splitter,) or (M/2, n_splitter)
        :param x: Array of size (M,) or (N, M)
        :param dag: If True, use the Hermitian conjugate (T^dagger).
        :return: Array of size (M,) or (N, M)
        """
        if (y.ndim == 1):
            return self.rdot(p_phase, p_splitter, np.outer(1, y))[0, :]
        (m, n) = y.shape; T = (self.Tdag if dag else self.T)(p_phase, p_splitter).reshape([2,2,n//2]).transpose((2,0,1))
        return (T.reshape((1,) + T.shape) * y.reshape([m, n//2, 2, 1])).sum(axis=2).reshape(y.shape)

    def grad(self, p_phase, p_splitter, x, gradY):
        r"""
        Obtains the gradient dJ/d[p_phase], using forward- and back-propagating fields x, gradY.
        :param p_phase: Array of size (k, n_phase)
        :param p_splitter: Array of size (k, n_splitter)
        :param x: Forward-propagating field, size (k, q)
        :param gradY: Back-propagating gradient dJ/d[y*], size (k, q)
        :return: Array of size (k/2, n_phase)
        """
        dT = self.dT(p_phase, p_splitter).transpose((0, 3, 1, 2))
        (r, k, m, n) = dT.shape; q = x.shape[1]
        x = x.reshape((k, n, q)); gradY = gradY.reshape((k, m, q))
        dTx = (dT.reshape((r, k, m, n, 1)) * x.reshape((1, k, 1, n, q))).sum(axis=3)
        return np.real(gradY.reshape((1, k, m, q)) * dTx.conj()).sum(axis=(2, 3)).T
        # Gotta love the index manipulation lol


class MZICrossing(Crossing):
    @property
    def n_phase(self) -> int:
        return 2
    @property
    def n_splitter(self) -> int:
        return 2

    def __init__(self):
        r"""
        Class implementing the conventional MZI crossing:
        -->--[phi]--| (pi/4    |--[2*theta]--| (pi/4   |-->--
        -->---------|  +beta') |-------------|  +beta) |-->--
        Here p_phase = (theta, phi) and p_splitter = (beta, beta').
        """
        pass
    def T(self, p_phase, p_splitter: Any=0.):
        (theta, phi) = np.array(p_phase).T; beta = np.array(p_splitter).T if np.iterable(p_splitter) else (p_splitter,)*2
        (Cp, Cm, C, Sp, Sm, S) = [fn(x) for fn in [np.cos, np.sin] for x in [beta[0]+beta[1], beta[0]-beta[1], theta]]
        return np.exp(1j*theta) * np.array([[np.exp(1j*phi) * (1j*S*Cm - C*Sp),    1j*C*Cp - S*Sm],
                                            [np.exp(1j*phi) * (1j*C*Cp + S*Sm),   -1j*S*Cm - C*Sp]])

    def dT(self, p_phase, p_splitter: Any=0.):
        (theta, phi) = np.array(p_phase).T; beta = np.array(p_splitter).T if np.iterable(p_splitter) else (p_splitter,)*2
        (Cp, Cm, C, Sp, Sm, S) = [fn(x) for fn in [np.cos, np.sin] for x in [beta[0]+beta[1], beta[0]-beta[1], theta]]
        return (np.exp(1j*np.array([[[phi+theta, theta]]])) *
                np.array([[[1j*(1j*S*Cm-C*Sp)+( 1j*C*Cm+S*Sp),   1j*( 1j*C*Cp-S*Sm)+(-1j*S*Cp-C*Sm)],
                           [1j*(1j*C*Cp+S*Sm)+(-1j*S*Cp+C*Sm),   1j*(-1j*S*Cm-C*Sp)+(-1j*C*Cm+S*Sp)]],
                          [[1j*(1j*S*Cm-C*Sp),                   0*S                               ],
                           [1j*(1j*C*Cp+S*Sm),                   0*S                               ]]]))

    def Tsolve(self, T, ind, p_splitter: Any=0.):
        beta = p_splitter if np.iterable(p_splitter) else (p_splitter, p_splitter)
        (Cp, Cm, Sp, Sm) = [fn(x) for fn in [np.cos, np.sin] for x in [beta[0]+beta[1], beta[0]-beta[1]]]
        if (ind in [0, (0, 0), 'T11']):
            # Input target T = T[0, 0]
            S2 = (np.abs(T)**2 - Sp**2) / (Cm**2 - Sp**2)
            err = 1*((S2 < 0) or (S2 > 1)); theta = np.arcsin(np.sqrt(np.clip(S2, 0, 1)))
            phi = np.angle(T) - np.angle(1j*Cm*np.sin(theta) - np.cos(theta)*Sp) - theta
        elif (ind in [1, (1, 0), 'T21']):
            # Input target T = T[1, 0]
            S2 = (Cp**2 - np.abs(T)**2) / (Cp**2 - Sm**2)
            err = 1*((S2 < 0) or (S2 > 1)); theta = np.arcsin(np.sqrt(np.clip(S2, 0, 1)))
            phi = np.angle(T) - np.angle(1j*Cp*np.cos(theta) + np.sin(theta)*Sm) - theta
        elif (ind == 'T1:'):
            # Input target T[1, :] = (T11, T12) and try to match the ratio T12/T11.
            R2 = np.abs(T[1])**2/(np.abs(T[0])**2 + 1e-30); S2 = (Cp**2 - R2*Sp**2)/((Cp**2-Sm**2) + R2*(Cm**2-Sp**2))
            err = 1*((S2 < 0) or (S2 > 1)); theta = np.arcsin(np.sqrt(np.clip(S2, 0, 1)))
            (C, S) = (np.cos(theta), np.sin(theta))
            phi = np.angle(1j*C*Cp - S*Sm) - np.angle(1j*Cm*S - C*Sp) + np.angle(T[0]) - np.angle(T[1])
        elif (ind == 'T2:'):
            # Input target T[2, :] = (T21, T22) and try to match the ratio T22/T21.
            R2 = np.abs(T[1])**2/(np.abs(T[0])**2 + 1e-30); S2 = (-Sp**2 + R2*Cp**2)/((Cm**2-Sp**2) + R2*(Cp**2-Sm**2))
            err = 1*((S2 < 0) or (S2 > 1)); theta = np.arcsin(np.sqrt(np.clip(S2, 0, 1)))
            (C, S) = (np.cos(theta), np.sin(theta))
            phi = np.angle(-1j*Cm*S - Sp*C) - np.angle(1j*Cp*C + Sm*S) + np.angle(T[0]) - np.angle(T[1])
        else:
            raise NotImplementedError()
        return ((theta, phi), err)

    def clemshift(self, psi, p_phase, p_splitter: Any=0):
        # Transformation: T(p)* D(psi) -> D(psi') T(p').
        # Since T(p) = t(theta) D([phi, 0]), we have:
        # T(p)* D(psi) = D([-phi, 0]) t(theta)* D([psi1, psi2]) -> D([psi1', psi2']) t(theta') D([phi', 0])
        # Now look at the components: t = [[t11, t12], [t21, t22]].
        # Conjugation relations:
        #   t11(-theta) = t11(theta)*, same with t22.  |t12(theta)| = |t21(theta)| = |t12(-theta)| = |t21(-theta)|
        # In the above transformation, we can set theta' = -theta; thus we have:
        #   phi' + psi1' = psi1 - phi
        #   psi1'        = psi2 - phi - arg(t21) - arg(t12')
        #   phi' + psi2' = psi1 - arg(t12) - arg(t21')
        #   psi2'        = psi2
        # There is some redundancy here.  First solve for psi2', then psi1', then phi'
        (psi1, psi2) = psi; (theta, phi) = p_phase
        t = self.T((theta, 0), p_splitter)
        tp = self.T((-theta, 0), p_splitter)
        theta_p = -theta
        psi2_p  = psi2
        psi1_p  = psi2 - phi - np.angle(t[1, 0]) - np.angle(tp[0, 1])
        phi_p   = psi1 - phi - psi1_p
        return ([theta_p, phi_p], [psi1_p, psi2_p])
