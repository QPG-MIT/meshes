# meshes/crossing.py
# Ryan Hamerly, 12/10/20
#
# Implements a generic class to handle 2x2 crossings, and its most common implementation (MZI).  Handling this
# functionality in a separate class allows one to easily switch between different crossing types (MZI, ring,
# double-ring, ring-assisted MZI, lossy MZI, etc.) without rewriting the main code.
#
# History
#   07/09/20: Created this file.  Defined classes Crossing, MZICrossing.
#   07/10/20: Added clemshift() functionality (convert T(p)* psi -> psi' T(p') for Clements decomposition).
#   12/10/20: Added MZICrossingOutPhase with phase shifter on the lower output.
#   12/20/20: Added MZICrossingBalanced (symmetric +theta / -theta pairing).
#   03/29/21: Added CartesianCrossing (non-singular parameterization) and crossing conversion utility.
#   04/07/21: Slight speedup using numpy.einsum for dot(), rdot(), grad().
#   04/13/21: Replaced 2*theta -> theta in phase shifters for consistency in notation.

import numpy as np
from typing import Any, Tuple

class Crossing:
    @property
    def n_phase(self) -> int:
        raise NotImplementedError()
    @property
    def n_splitter(self) -> int:
        raise NotImplementedError()
    @property
    def tunable_indices(self) -> Tuple:
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
    def Tsolve(self, T, ind, p_splitter: Any=0.) -> Tuple[Tuple, int]:
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
    def clemshift(self, phi, p_phase, p_splitter: Any=0) -> Tuple[Tuple, Tuple]:
        r"""
        Performs the phase-shifter identity employed in the Clements decomposition: Tdag(p) * PS -> PS' * T(p')
        :param phi: Phase shifts [PS].  Length-2 array.
        :param p_phase: Crossing degrees of freedom [p].  Length-(n_phase) array.
        :param p_splitter: Crossing imperfections.  Length-(n_splitter) array.
        :return: A pair (p', PS') containing new parameters and shifted phases.
        """
        raise NotImplementedError()
    def flip(self) -> 'Crossing':
        r"""
        Flips the location of the phase shifter from the top input to the bottom output.  Used in the Reciprocal RELLIM
        calibration scheme.
        :return: A meshes.Crossing object.
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
    def rmult_falling(self, M, p_phase, p_splitter: Any=0., inplace=True) -> np.ndarray:
        r"""
        Right-multiplies by matrix T for a falling diagonal, i.e. M -> M T.
        T = T_n T_{n-1} ... T_1.  So we right-multiply in reverse order.
        :param M: Matrix to be multiplied.
        :param p_phase: Phase-shifter (or other d.o.f.) information.
        :param p_splitter: Splitter errors or other manufacturing imperfections.
        :param inplace: Whether to multiply M -> M*T in place
        :return: The matrix M*T.
        """
        if not inplace: M = np.array(M)
        N = p_phase.shape[0]; assert (N == M.shape[1]-1); p_splitter = p_splitter * np.ones((N, self.n_splitter))
        for (i, p_phase_i, p_splitter_i) in zip(range(N)[::-1], p_phase[::-1], p_splitter[::-1]):
            M[:, i:i+2] = M[:, i:i+2].dot(self.T(p_phase_i, p_splitter_i))
        return M

    def dot(self, p_phase, p_splitter, x, dag=False, dp=None, dx=None) -> np.ndarray:
        r"""
        Performs the dot product T*x.  Can be used for a single splitter or an array.
        :param p_phase: Array of size (n_phase,) or (M/2, n_phase)
        :param p_splitter: Array of size (n_splitter,) or (M/2, n_splitter)
        :param x: Array of size (M,) or (M, N)
        :param dag: If True, use the Hermitian conjugate (T^dagger).
        :param dp: Derivative of p_phase
        :param dx: Derivative of x
        :return: If no derivatives specified, array [y = Tx] of size (M,) or (M, N).  Otherwise, two arrays [y, dy].
        """
        if (x.ndim == 1):
            out = self.dot(p_phase, p_splitter, np.outer(x, 1), dag, dp)
            return out[:, 0] if (dp is None) else (out[0][:, 0], out[1][:, 0])
        (m, n) = x.shape; T = (self.Tdag if dag else self.T)(p_phase, p_splitter).reshape([2,2,m//2]).transpose((2,0,1))
        y = (np.einsum('ijk,ikl->ijl', T, x.reshape([m//2, 2, n])).reshape(x.shape))
        if (dp is None) and (dx is None):
            return y
        else:
            (dy1, dy2) = (0, 0)
            if (dp is not None):
                tr = (3, 0, 2, 1) if dag else (3, 0, 1, 2); dT = self.dT(p_phase, p_splitter).transpose(*tr)
                dT = np.einsum('ijkl,ij->ikl', dT.conj() if dag else dT, dp)
                dy1 = np.einsum('ijk,ikl->ijl', dT, x.reshape([m//2, 2, n])).reshape(x.shape)
            if (dx is not None):
                dy2 = np.einsum('ijk,ikl->ijl', T, dx.reshape([m//2, 2, n])).reshape(x.shape)
            return (y, dy1+dy2)

    def rdot(self, p_phase, p_splitter, y, dag=False) -> np.ndarray:
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
        return np.einsum('ijk,jkl->ijl', y.reshape([m, n//2, 2]), T).reshape(y.shape)

    def grad(self, p_phase, p_splitter, x, gradY) -> np.ndarray:
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
        return np.real(np.einsum('jkl,ijkl->ji', gradY, dTx.conj()))
        # Gotta love the index manipulation lol

    def convert(self,
                out: 'Crossing',
                p_phase: np.ndarray,
                p_splitter: np.ndarray=None,
                p_splitter_out: np.ndarray=None,
                phi: np.ndarray=None,
                phi_pos: str='out') -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Converts a particular Crossing to a different type.  Given (D, p), solves for (D', p') such that
        D'*T_out(p') = T_self(p)*D (if phi_pos='out'), or T_out(p')*D' = D*T_self(p) (if 'in')
        :param out: The output crossing instance.
        :param p_phase: Crossing parameters.
        :param p_splitter: Splitter imperfections.
        :param p_splitter_out: Splitter imperfections (output)
        :param phi: Phase screen.
        :param phi_pos: Position to propagate the phase screen.
        :return: Tuple (p', phi')
        """
        # Set up arrays.
        def set(p, s): p = np.zeros(s) if (p is None) else p; assert (p.shape == tuple(s)); return p
        n_cr = len(p_phase); p_in = set(p_phase, [n_cr, self.n_phase]); phi = set(phi, [2*n_cr])
        s_in = set(p_splitter, [n_cr, self.n_splitter]); s_out = set(p_splitter_out, [n_cr, out.n_splitter])

        if (phi_pos == 'out'):
            # Compute T0*D0 from old crossing, find T based on amplitude ratios, solve new phases D*T = T0*D0 based on:
            # D = diag(T0*D0*(T*)) (the latter should be a diagonal matrix)
            T0 = self.T(p_in, s_in); T0[:, 0] *= np.exp(1j*phi[::2]); T0[:, 1] *= np.exp(1j*phi[1::2])
            p_out = np.array(out.Tsolve((T0[0, 0], T0[0, 1]), 'T1:', s_out.T)[0]).T
            T = out.T(p_out, s_out)
            phi = np.array([np.angle(T0[i, 0]*T[i, 0].conj() + T0[i, 1]*T[i, 1].conj()) for i in [0, 1]]).T.flatten()
            return (p_out, phi)
        else:
            raise NotImplementedError()  # TODO -- implement the left phase propagation.


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
        -->--[phi]--| (pi/4    |--[theta]--| (pi/4   |-->--
        -->---------|  +beta') |-----------|  +beta) |-->--
        Here p_phase = (theta, phi) and p_splitter = (beta, beta').
        """
        pass
    def T(self, p_phase, p_splitter: Any=0.) -> np.ndarray:
        (theta, phi) = np.array(p_phase).T; (a, b) = (np.array(p_splitter).T if np.iterable(p_splitter) else (p_splitter,)*2)
        (Cp, Cm, C, Sp, Sm, S) = [fn(x) for fn in [np.cos, np.sin] for x in [a+b, a-b, theta/2]]
        return np.exp(1j*theta/2) * np.array([[np.exp(1j*phi) * (1j*S*Cm - C*Sp),    1j*C*Cp - S*Sm],
                                              [np.exp(1j*phi) * (1j*C*Cp + S*Sm),   -1j*S*Cm - C*Sp]])

    def dT(self, p_phase, p_splitter: Any=0.) -> np.ndarray:
        (theta, phi) = np.array(p_phase).T; (a, b) = (np.array(p_splitter).T if np.iterable(p_splitter) else (p_splitter,)*2)
        (Cp, Cm, C, Sp, Sm, S) = [fn(x) for fn in [np.cos, np.sin] for x in [a+b, a-b, theta/2]]
        return (np.exp(1j*np.array([[[phi+theta/2, theta/2]]])) *
                np.array([[[0.5j*(1j*S*Cm-C*Sp)+0.5*( 1j*C*Cm+S*Sp),   0.5j*( 1j*C*Cp-S*Sm)+0.5*(-1j*S*Cp-C*Sm)],
                           [0.5j*(1j*C*Cp+S*Sm)+0.5*(-1j*S*Cp+C*Sm),   0.5j*(-1j*S*Cm-C*Sp)+0.5*(-1j*C*Cm+S*Sp)]],
                          [[  1j*(1j*S*Cm-C*Sp),                   0*S                                 ],
                           [  1j*(1j*C*Cp+S*Sm),                   0*S                                 ]]]))

    def Tsolve(self, T, ind, p_splitter: Any=0.) -> Tuple[Tuple, int]:
        beta = p_splitter if np.iterable(p_splitter) else (p_splitter, p_splitter)
        (Cp, Cm, Sp, Sm) = [fn(x) for fn in [np.cos, np.sin] for x in [beta[0]+beta[1], beta[0]-beta[1]]]
        if (ind in [0, (0, 0), 'T11']):
            # Input target T = T[0, 0]
            S2 = (np.abs(T)**2 - Sp**2) / (Cm**2 - Sp**2)
            err = 1*((S2 < 0) | (S2 > 1)); theta2 = np.nan_to_num(np.arcsin(np.sqrt(np.clip(S2, 0, 1))), 0, 0, 0)
            phi = np.angle(T) - np.angle(1j*Cm*np.sin(theta2) - np.cos(theta2)*Sp) - theta2
        elif (ind in [1, (1, 0), 'T21']):
            # Input target T = T[1, 0]
            S2 = (Cp**2 - np.abs(T)**2) / (Cp**2 - Sm**2)
            err = 1*((S2 < 0) | (S2 > 1)); theta2 = np.nan_to_num(np.arcsin(np.sqrt(np.clip(S2, 0, 1))), 0, 0, 0)
            phi = np.angle(T) - np.angle(1j*Cp*np.cos(theta2) + np.sin(theta2)*Sm) - theta2
        elif (ind == 'T1:'):
            # Input target T[1, :] = (T11, T12) and try to match the ratio T12/T11.
            R2 = np.abs(T[1])**2/(np.abs(T[0])**2 + 1e-30); S2 = (Cp**2 - R2*Sp**2)/((Cp**2-Sm**2) + R2*(Cm**2-Sp**2))
            err = 1*((S2 < 0) | (S2 > 1)); theta2 = np.nan_to_num(np.arcsin(np.sqrt(np.clip(S2, 0, 1))), 0, 0, 0)
            (C, S) = (np.cos(theta2), np.sin(theta2))
            phi = np.angle(1j*C*Cp - S*Sm) - np.angle(1j*Cm*S - C*Sp) + np.angle(T[0]) - np.angle(T[1])
        elif (ind == 'T2:'):
            # Input target T[2, :] = (T21, T22) and try to match the ratio T22/T21.
            R2 = np.abs(T[1])**2/(np.abs(T[0])**2 + 1e-30); S2 = (-Sp**2 + R2*Cp**2)/((Cm**2-Sp**2) + R2*(Cp**2-Sm**2))
            err = 1*((S2 < 0) | (S2 > 1)); theta2 = np.nan_to_num(np.arcsin(np.sqrt(np.clip(S2, 0, 1))), 0, 0, 0)
            (C, S) = (np.cos(theta2), np.sin(theta2))
            phi = np.angle(-1j*Cm*S - Sp*C) - np.angle(1j*Cp*C + Sm*S) + np.angle(T[0]) - np.angle(T[1])
        else:
            raise NotImplementedError(ind)
        return ((theta2*2, phi), err)

    def clemshift(self, psi, p_phase, p_splitter: Any=0) -> Tuple[Tuple, Tuple]:
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
        return ((theta_p, phi_p), (psi1_p, psi2_p))

    def flip(self) -> Crossing:
        return MZICrossingOutPhase()

    @property
    def tunable_indices(self) -> Tuple:
        return ('T11', 'T21', 'T1:', 'T2:')

    
class MZICrossingBalanced(MZICrossing):
    def __init__(self):
        r"""
        Class implementing the symmetric MZI crossing:
        -->--[phi]--| (pi/4    |--[+theta/2]--| (pi/4   |-->--
        -->---------|  +beta') |--[-theta/2]--|  +beta) |-->--
        Here p_phase = (theta, phi) and p_splitter = (beta, beta').
        """
        pass
    def T(self, p_phase, p_splitter: Any=0.) -> np.ndarray:
        (theta, phi) = np.array(p_phase).T; theta = np.asarray(theta)
        out = super(MZICrossingBalanced, self).T(p_phase, p_splitter)
        out *= np.exp(-1j*theta/2).reshape((1, 1) + theta.shape)
        return out
    @property
    def tunable_indices(self) -> Tuple:
        return ('T1:', 'T2:')   # TODO -- also handle T11 and T21
    def Tsolve(self, T, ind, p_splitter: Any=0.) -> Tuple[Tuple, int]:
        assert (ind in ['T1:', 'T2:'])    # TODO -- also handle T11 and T21
        return super(MZICrossingBalanced, self).Tsolve(T, ind, p_splitter)
    def dT(self, p_phase, p_splitter: Any=0.) -> np.ndarray:
        raise NotImplementedError()
    def flip(self) -> Crossing:
        raise NotImplementedError()


class MZICrossingOutPhase(MZICrossing):
    def __init__(self):
        r"""
        Class implementing the MZI crossing with phase shifter on the output:
        -->--| (pi/4    |--[theta]--| (pi/4   |--------->--
        -->--|  +beta') |-----------|  +beta) |--[phi]-->--
        Here p_phase = (theta, phi) and p_splitter = (beta, beta').
        """
        pass
    def _p_splitter(self, p_splitter):
        # Gets the splitter angles for the flipped MZICrossing element.
        beta = np.array(p_splitter if np.iterable(p_splitter) else [p_splitter]*2, dtype=np.float).T
        beta = beta[::-1]; beta[0] += np.pi/2; beta[1] -= np.pi/2; return beta.T

    # Based on the identity:
    # T_out(theta, phi, b1, b2) = T_in(theta, phi, b2+pi/2, b1-pi/2)[::-1,::-1].T
    # where T_out has phase shifter on output 2, T_in on input 1.
    def T(self, p_phase, p_splitter: Any=0.) -> np.ndarray:
        T_in = super(MZICrossingOutPhase, self).T(p_phase, self._p_splitter(p_splitter))
        return T_in[::-1,::-1].T if T_in.ndim == 2 else T_in[::-1,::-1].transpose(1, 0, 2)
    def dT(self, p_phase, p_splitter: Any=0.) -> np.ndarray:
        dT_in = super(MZICrossingOutPhase, self).dT(p_phase, self._p_splitter(p_splitter))
        return dT_in[:,::-1,::-1].transpose(0, 2, 1) if dT_in.ndim == 3 else dT_in[:,:,::-1,::-1].transpose(0, 2, 1, 3)
    def Tsolve(self, T, ind, p_splitter: Any=0.) -> Tuple[Tuple, int]:
        # Just calls Tsolve from MZICrossing with appropriate transformations.
        # I'm really not sure why T transforms as such in the T:1, T:2 cases.  But it works, numerically.
        p_splitter = self._p_splitter(p_splitter)
        if   (ind in [(1, 1), 'T22']): (ind_in, T) = ('T11', T)
        elif (ind in [(1, 0), 'T21']): (ind_in, T) = ('T21', T)
        elif (ind == 'T:1'):           (ind_in, T) = ('T1:', (-T[0].conjugate(), T[1].conjugate()))
        elif (ind == 'T:2'):           (ind_in, T) = ('T2:', (-T[0].conjugate(), T[1].conjugate()))
        else: raise NotImplementedError(ind)
        return super(MZICrossingOutPhase, self).Tsolve(T, ind_in, p_splitter)
    def clemshift(self, psi, p_phase, p_splitter: Any=0) -> Tuple[Tuple, Tuple]:
        raise NotImplementedError()
    def flip(self) -> Crossing:
        return MZICrossing()
    @property
    def tunable_indices(self) -> Tuple:
        return ('T22', 'T21', 'T:1', 'T:2')


class SymCrossing(Crossing):
    @property
    def n_phase(self) -> int:
        return 2
    @property
    def n_splitter(self) -> int:
        return 1

    def __init__(self):
        r"""
        Class implementing the symmetric crossing.
        [[s,                   i sqrt(1 - |s|^2)],
         [i sqrt(1 - |s|^2),   s                ]]
        where s = exp(i*phi) (sin(theta/2) + i cos(theta/2) sin(2*alpha))
        Here p_phase = [theta, phi], p_splitter = [alpha]
        """
        pass
    def T(self, p_phase, p_splitter: Any=0.) -> np.ndarray:
        (theta, phi) = p_phase.T; (beta,) = p_splitter.T
        (C, C_2a, S, S_2a) = [fn(x) for fn in [np.cos, np.sin] for x in [theta/2, 2*beta]]
        return np.array([[np.exp(1j*phi)*(S + 1j*C*S_2a),  1j*C*C_2a],
                         [1j*C*C_2a, np.exp(-1j*phi)*(S - 1j*C*S_2a)]])


class MZICrossing3(Crossing):
    _X: Crossing
    @property
    def n_phase(self) -> int:
        return 2
    @property
    def n_splitter(self) -> int:
        return 3
    @property
    def tunable_indices(self) -> Tuple:
        return ('T1:',)

    def __init__(self):
        r"""
        Class implementing a 3-MZI crossing, p_phase=(theta, phi), p_splitter=(alpha, beta, gamma).
        """
        self._X = MZICrossing()

    def _get_beta(self, p_splitter):
        return (np.array(p_splitter).T if np.iterable(p_splitter) else (p_splitter,)*3)

    def T(self, p_phase, p_splitter: Any=0.) -> np.ndarray:
        sp = self._get_beta(p_splitter)
        ((T11, T12), (T21, T22)) = self._X.T(p_phase, sp[:2].T); eta = sp[2]
        s11 = s22 = np.cos(eta); s12 = s21 = 1j*np.sin(eta)
        return np.array([[T11*s11 + T12*s21,  T11*s12 + T12*s22],
                         [T21*s11 + T22*s21,  T21*s12 + T22*s22]])

    def Tsolve(self, T, ind, p_splitter: Any=0.) -> Tuple[Tuple, int]:
        sp = self._get_beta(p_splitter).T; eta = sp[2]
        if (ind == 'T1:'):
            (T11, T12) = (T[0], T[1]); s = T11/T12
            s = (s - 1j*np.tan(eta)) / (1 - 1j*np.tan(eta)*s)
            return self._X.Tsolve((s, s*0 + 1), ind, sp[:2])


class MZICrossingGeneric(Crossing):
    @property
    def n_phase(self) -> int:
        return 2
    @property
    def n_splitter(self) -> int:
        return 14     # [alpha, beta, d_phi_fab(x4), d_phi_xtalk(x4), d_abs(x4)]

    out_phase: bool

    def __init__(self, out_phase=False):
        r"""
        Class implementing a generic MZI-like crossing with crossing, phase-shifter, and nonunitary errors:
        -->--[phi+ph_11, g_11]--| (pi/4    |--[theta+ph_12, g_12]--| (pi/4   |-->--
        -->--[    ph_21, g_21]--|  +alpha) |--[      ph_22, g_22]--|  +beta) |-->--
        Here p_phase = (theta, phi) and p_splitter = (alpha, beta, [ph_ij_fab], [ph_ij_xtalk], [g_ij])
        where the phase error ph_ij = ph_ij_fab + ph_ij_xtalk is a sum of fabrication- and crosstalk-induced errors.
        Non-unitary errors are set by g_ij.
        :param out_phase: If True, implements the flipped version of the crossing, with output phase shifters.
            -->--| (pi/4   |--[      ph_22, g_22]--| (pi/4    |--[    ph_21, g_21]-->--
            -->--|  +beta) |--[theta+ph_12, g_12]--|  +alpha) |--[phi+ph_11, g_11]-->--
        """
        self.out_phase = out_phase

    def T(self, p_phase, p_splitter: Any=0.) -> np.ndarray:
        (theta, phi) = np.array(p_phase).T; zero = theta*0
        (a, b, p11, p21, p12, p22, q11, q21, q12, q22, g11, g21, g12, g22) = \
            [x+zero for x in (np.array(p_splitter).T if np.iterable(p_splitter) else (p_splitter,)*14)]
        # T1 & T3: phase shifters.  T2 & T4: splitters.
        T1 = np.array([[np.exp(1j*(phi+p11+q11)+g11), zero], [zero, np.exp(1j*(p21+q21)+g21)]])
        T2 = np.array([[   np.cos(np.pi/4+a),  1j*np.sin(np.pi/4+a)],
                       [1j*np.sin(np.pi/4+a),     np.cos(np.pi/4+a)]])
        T3 = np.array([[np.exp(1j*(theta+p12+q12)+g12), zero], [zero, np.exp(1j*(p22+q22)+g22)]])
        T4 = np.array([[   np.cos(np.pi/4+b),  1j*np.sin(np.pi/4+b)],
                       [1j*np.sin(np.pi/4+b),     np.cos(np.pi/4+b)]])

        if self.out_phase:
            T = T4
            T = np.einsum('ij...,jk...->ik...', T3[::-1,::-1], T)
            T = np.einsum('ij...,jk...->ik...', T2, T)
            T = np.einsum('ij...,jk...->ik...', T1[::-1,::-1], T)
            return T
        else:
            T = T1
            T = np.einsum('ij...,jk...->ik...', T2, T)
            T = np.einsum('ij...,jk...->ik...', T3, T)
            T = np.einsum('ij...,jk...->ik...', T4, T)
            return T

    def flip(self) -> Crossing:
        return MZICrossingGeneric(not self.out_phase)





class CartesianCrossing(Crossing):
    @property
    def n_phase(self) -> int:
        return 2
    @property
    def n_splitter(self) -> int:
        return 2
    @property
    def tunable_indices(self) -> Tuple:
        return ('T1:',)

    def __init__(self):
        r"""
        Class implementing an MZI crossing in non-singular Cartesian coordinates.  Representation depends on the angle:
        [ s                i sqrt(1-|s|^2) ]
        [ i sqrt(1-|s|^2)  s*              ]
        for theta < pi/2 (s = sin(theta/2)*e^{i*phi}) and
        [ i sqrt(1-|t|^2)  t               ]
        [ t*               i sqrt(1-|t|^2) ]
        for theta > pi/2 (t = cos(theta/2)*e^{i*phi})
        Here p_phase = (Re[s]+2, Im[s]) or (Re[t]-2, Im[t]) depending on the representation.  Constrained to unit
        disks centered at (2+0j) or (-2+0j).
        and p_splitter = (alpha, beta).  The (+/- 2) is to
        """
        pass

    def T(self, p_phase, p_splitter: Any=0.) -> np.ndarray:
        (x, y) = np.array(p_phase).T; C = (x > 0); B = (1 - C); z = (x + 1j*y) - (4*C - 2)
        r = np.sqrt(1 - np.abs(z**2))
        return np.array([[C*(z   ) + B*(1j*r),       C*(1j*r    ) + B*(z)   ],
                         [C*(1j*r) + B*(z.conj()),   C*(z.conj()) + B*(1j*r)]])

    def dT(self, p_phase, p_splitter: Any=0.) -> np.ndarray:
        (x, y) = np.array(p_phase).T; C = (x > 0); B = (1 - C); x = x - (4*C - 2)
        one = x*0j + 1
        r = np.sqrt(1 - x**2 - y**2)
        return np.array([[[C*(one    ) + B*(-1j*x/r),   C*(-1j*x/r) + B*(one    )],
                          [C*(-1j*x/r) + B*(one    ),   C*(one    ) + B*(-1j*x/r)]],
                         [[C*( 1j*one) + B*(-1j*y/r),   C*(-1j*y/r) + B*(1j*one )],
                          [C*(-1j*y/r) + B*(-1j*one),   C*(-1j*one) + B*(-1j*y/r)]]])

    def Tsolve(self, T, ind, p_splitter: Any=0.) -> Tuple[Tuple, int]:
        (a, b) = np.array(p_splitter) if np.iterable(p_splitter) else (p_splitter,)*2
        if (ind == 'T1:'):
            (T11, T12) = (T[0], T[1]); C = np.abs(T11) <= np.abs(T12); B = (1 - C)
            T1abs = np.sqrt(T11.conj()*T11 + T12.conj()*T12 + 1e-15)
            s = np.abs(T11)/T1abs * np.exp(1j*(np.angle(T11)-np.angle(-1j*T12))); s0 = np.sin(np.abs(a+b))
            t = np.abs(T12)/T1abs * np.exp(1j*(np.angle(T12)-np.angle(-1j*T11))); t0 = np.sin(np.abs(a-b))
            err = 1 * (C & (np.abs(s) < s0 - 1e-8)) | (B & (np.abs(t) < t0 - 1e-8))
            s *= np.maximum(np.abs(s), s0)/(np.abs(s)+1e-30); t *= np.maximum(np.abs(t), t0)/(np.abs(t)+1e-30)
            x = C*(s.real + 2) + B*(t.real - 2)
            y = C*(s.imag    ) + B*(t.imag    )
        else:
            raise NotImplementedError(ind)  # TODO -- fill in the cases I'm too lazy to implement...
        return ((x, y), err)
