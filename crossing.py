# meshes/crossing.py
# Ryan Hamerly, 7/9/20
#
# Implements a generic class to handle 2x2 crossings, and its most common implementation (MZI).  Handling this
# functionality in a separate class allows one to easily switch between different crossing types (MZI, ring,
# double-ring, ring-assisted MZI, lossy MZI, etc.) without rewriting the main code.
#
# History
#   07/09/20: Created this file.  Defined classes Crossing, MZICrossing.

# TODO -- make the Reck / Clements code call these classes rather than what they currently do.
# TODO -- functionality to convert a crossing to its conjugate (with auxiliary phases)
# TODO -- mechanism to push phases to the front / back of a crossing, used in Clements decomposition.


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
        :param p_phase: Phase-shifter (or other d.o.f.) information.
        :param p_splitter: Splitter errors or other manufacturing imperfections.
        :return:
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

    def Tdag(self, p_phase, p_splitter: Any=0.) -> np.ndarray:
        r"""
        Obtains the T.conj().transpose()
        :param p_phase: Phase-shifter (or other d.o.f.) information.
        :param p_splitter: Splitter errors or other manufacturing imperfections.
        :return:
        """
        return self.T(p_phase, p_splitter).conj().transpose()
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
        (theta, phi) = p_phase; beta = p_splitter if np.iterable(p_splitter) else (p_splitter, p_splitter)
        (Cp, Cm, C, Sp, Sm, S) = [fn(x) for fn in [np.cos, np.sin] for x in [beta[0]+beta[1], beta[0]-beta[1], theta]]
        return np.exp(1j*theta) * np.array([[np.exp(1j*phi) * (1j*Cm*S - C*Sp),    1j*C*Cp - S*Sm],
                                            [np.exp(1j*phi) * (1j*C*Cp + S*Sm),   -1j*Cm*S - C*Sp]])
    def Tsolve(self, T, ind, p_splitter: Any=0.):
        beta = p_splitter if np.iterable(p_splitter) else (p_splitter, p_splitter)
        (Cp, Cm, Sp, Sm) = [fn(x) for fn in [np.cos, np.sin] for x in [beta[0]+beta[1], beta[0]-beta[1]]]
        if (ind == 0):
            S2 = (np.abs(T)**2 - Sp**2) / (Cm**2 - Sp**2)
            err = 1*((S2 < 0) or (S2 > 1)); theta = np.arcsin(np.sqrt(np.clip(S2, 0, 1)))
            phi = np.angle(T) - np.angle(1j*Cm*np.sin(theta) - np.cos(theta)*Sp) - theta
        elif (ind == 1):
            S2 = (Cp**2 - np.abs(T)**2) / (Cp**2 - Sm**2)
            err = 1*((S2 < 0) or (S2 > 1)); theta = np.arcsin(np.sqrt(np.clip(S2, 0, 1)))
            phi = np.angle(T) - np.angle(1j*Cp*np.cos(theta) + np.sin(theta)*Sm) - theta
        else:
            raise NotImplementedError()
        return ((theta, phi), err)
