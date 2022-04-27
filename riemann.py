# meshes/riemann.py
# Ryan Hamerly, 4/20/22
#
# Riemann-sphere methods for self-configuration.  Presently only useful for benchmarking, i.e. in paper "Infinitely
# scalable multiport interferometers", arXiv:2109.05367.  It would be nice to harmonize this with the other
# self-configuration scripts in the package, as this code is somewhat faster but not as generalizable, e.g. it only
# supports MZI / 3-MZI / MZI+X structures.
#
# History
#   04/20/22: Created this file.


import numpy as np
import meshes as ms
import matplotlib.pyplot as plt
from scipy.stats import unitary_group
from numba import njit

# @njit
# def reckdec(U, mu, sig, eta):
#     r"""
#     Performs the Reck decomposition on a matrix U, assuming Gaussian beamsplitter errors (a, b) ~ N(mu, sig).
#     Returns the normalized error |U - U0| / sqrt(N).
#     :param U: Target matrix
#     :param mu: Mean beamsplitter error
#     :param sig: Std-dev beamsplitter error
#     :param eta: Coupling angle for third splitter.  Prominent choices: 0 -> MZI, pi/2 -> 3-MZI, pi -> MZI+X
#     :return:
#     """
#     N = len(U); z = np.tan(eta)
#     for i in range(N-1):
#         for j in range(N-2, i-1, -1):
#             (a, b) = np.random.randn(2)*sig + mu
#             (x, y) = U[i, j:j+2]
#             s = 1j*x/y
#             s = (s + 1j*z) / (1 + 1j*s*z)
#             abs_s = np.abs(s);
#             s *= min(max(abs_s, np.abs(np.tan(np.abs(a+b)))), np.abs(1/np.tan(np.abs(a-b)+1e-30))) / (abs_s)
#             s = (s - 1j*z) / (1 - 1j*s*z)
#             (theta, phi) = (2*np.arctan(np.abs(s)), np.angle(s))
#             T = np.array([[np.exp(1j*phi)*np.sin(theta/2), 1j*np.cos(theta/2)],
#                           [1j*np.cos(theta/2), np.exp(-1j*phi)*np.sin(theta/2)]])
#             U[:, j:j+2] = U[:, j:j+2].dot(T.conj().T)
#     return np.linalg.norm(U - np.diag(np.diag(U))) / np.sqrt(N)

# @njit
# def clemdec(U, mu, sig, eta):
#     r"""
#     Performs the Clements decomposition on a matrix U, assuming Gaussian beamsplitter errors (a, b) ~ N(mu, sig).
#     Returns the normalized error |U - U0| / sqrt(N).
#     :param U: Target matrix
#     :param mu: Mean beamsplitter error
#     :param sig: Std-dev beamsplitter error
#     :param eta: Coupling angle for third splitter.  Prominent choices: 0 -> MZI, pi/2 -> 3-MZI, pi -> MZI+X
#     :return:
#     """
#     N = len(U); z = np.tan(eta)
#     for i in range(N-1):
#         for j in range(i+1):
#             (a, b) = np.random.randn(2)*sig + mu
#             if (i % 2 == 0):
#                 (k, l) = (N-1-j, i-j)
#                 (x, y) = U[k, l:l+2]
#                 s = -1j*y/x
#                 s = (s + 1j*z) / (1 + 1j*s*z)
#                 abs_s = np.abs(s);
#                 s *= min(max(abs_s, np.abs(np.tan(np.abs(a+b)))), np.abs(1/np.tan(np.abs(a-b)+1e-30))) / (abs_s)
#                 s = (s - 1j*z) / (1 - 1j*s*z)
#                 (theta, phi) = (2*np.arctan(np.abs(s)), np.angle(s))
#                 T = np.array([[np.exp(1j*phi)*np.sin(theta/2), 1j*np.cos(theta/2)],
#                               [1j*np.cos(theta/2), np.exp(-1j*phi)*np.sin(theta/2)]])
#                 U[:, l:l+2] = U[:, l:l+2].dot(T)
#             else:
#                 (k, l) = (N-2-i+j, j)
#                 (x, y) = U[k:k+2, l]
#                 s = 1j*np.conj(x/(y))
#                 s = (s + 1j*z) / (1 + 1j*s*z)
#                 abs_s = np.abs(s);
#                 s *= min(max(abs_s, np.abs(np.tan(np.abs(a+b)))), np.abs(1/np.tan(np.abs(a-b)+1e-30))) / (abs_s)
#                 s = (s - 1j*z) / (1 - 1j*s*z)
#                 (theta, phi) = (2*np.arctan(np.abs(s)), np.angle(s))
#                 T = np.array([[np.exp(1j*phi)*np.sin(theta/2), 1j*np.cos(theta/2)],
#                               [1j*np.cos(theta/2), np.exp(-1j*phi)*np.sin(theta/2)]])
#                 U[k:k+2] = T.dot(U[k:k+2])
#     return np.linalg.norm(U - np.diag(np.diag(U))) / np.sqrt(N)


def get_err(N, mu, sig, eta, ct=20, method='reck', share=True):
    r"""
    Benchmarks error for the Reck/Clements decomposition.  Generates a set of Haar-random target matrices U, and
    for each (mu, sig) computes the mesh matrix error post-correction.

    :paran N: Matrix size
    :param mu: Mean beamsplitter error (list)
    :param sig: Std-dev beamsplitter error (list)
    :param eta: Coupling angle for third splitter.  Prominent choices: 0 -> MZI, pi/2 -> 3-MZI, pi -> MZI+X
    :param ct: Number of instances
    :param method: 'reck' or 'clements'
    :param share: Whether to use the same matrix U0 for each (mu, sig) in an instance.
    :return: Normalized error |U - U0| / sqrt(N).
    """
    if np.iterable(N):
        return np.array([get_err(N_i, mu, sig, eta, ct, method, share) for N_i in N])
    print (f"N = {N}")
    err_c = np.zeros([ct, len(mu)])
    dj = max(len(mu)//50, 1); di = max(ct//50, 1)
    for i in range(ct):
        if (share): U0 = np.array(unitary_group.rvs(N), order="F")
        for (j, (mu_i, sig_i)) in enumerate(zip(mu, sig)):
            U = np.copy(U0, order="F") if share else np.array(unitary_group.rvs(N), order="F")
            err_c[i, j] = {'reck': reckdec, 'clem': clemdec}[method](U, mu_i, sig_i, eta)
            if (len(mu) > 1) and (j % dj == 0): print ('.', end="", flush=True)
        if (len(mu) > 1): print(i)
        elif (i % di == 0): print ('.', end="", flush=True)
    if (len(mu) == 1): print()
    return err_c




@njit
def reckdec_helper(U, V, W, dp, eta, get_vw):
    N = len(U)
    ind = 0
    for i in range(N-1):
        for j in range(N-2, i-1, -1):
            z = np.tan(eta + dp[ind, 2]); (a, b) = (dp[ind, 0], dp[ind, 1])
            ind += 1
            (x, y) = U[i, j:j+2]
            s = 1j*x/y
            s = (s + 1j*z) / (1 + 1j*s*z)
            abs_s = np.abs(s);
            s *= min(max(abs_s, np.abs(np.tan(np.abs(a+b)))), np.abs(1/np.tan(np.abs(a-b)+1e-30))) / (abs_s)
            s = (s - 1j*z) / (1 - 1j*s*z)
            (theta, phi) = (2*np.arctan(np.abs(s)), np.angle(s))
            T = np.array([[np.exp(1j*phi)*np.sin(theta/2), 1j*np.cos(theta/2)],
                          [1j*np.cos(theta/2), np.exp(-1j*phi)*np.sin(theta/2)]])
            U[:, j:j+2] = U[:, j:j+2].dot(T.conj().T)
            if get_vw:
                W[j:j+2, :] = T.dot(W[j:j+2, :])


@njit
def clemdec_helper(U, V, W, dp, eta, get_vw):
    N = len(U)
    ind = 0
    for i in range(N-1):
        for j in range(i+1):
            z = np.tan(eta + dp[ind, 2]); (a, b) = (dp[ind, 0], dp[ind, 1])
            ind += 1
            if (i % 2 == 0):
                (k, l) = (N-1-j, i-j)
                (x, y) = U[k, l:l+2]
                s = -1j*y/x
                s = (s + 1j*z) / (1 + 1j*s*z)
                abs_s = np.abs(s);
                s *= min(max(abs_s, np.abs(np.tan(np.abs(a+b)))), np.abs(1/np.tan(np.abs(a-b)+1e-30))) / (abs_s)
                s = (s - 1j*z) / (1 - 1j*s*z)
                (theta, phi) = (2*np.arctan(np.abs(s)), np.angle(s))
                T = np.array([[np.exp(1j*phi)*np.sin(theta/2), 1j*np.cos(theta/2)],
                              [1j*np.cos(theta/2), np.exp(-1j*phi)*np.sin(theta/2)]])
                U[:, l:l+2] = U[:, l:l+2].dot(T)
                if get_vw:
                    W[l:l+2, :] = T.T.conj().dot(W[l:l+2, :])
            else:
                (k, l) = (N-2-i+j, j)
                (x, y) = U[k:k+2, l]
                s = 1j*np.conj(x/(y))
                s = (s + 1j*z) / (1 + 1j*s*z)
                abs_s = np.abs(s);
                s *= min(max(abs_s, np.abs(np.tan(np.abs(a+b)))), np.abs(1/np.tan(np.abs(a-b)+1e-30))) / (abs_s)
                s = (s - 1j*z) / (1 - 1j*s*z)
                (theta, phi) = (2*np.arctan(np.abs(s)), np.angle(s))
                T = np.array([[np.exp(1j*phi)*np.sin(theta/2), 1j*np.cos(theta/2)],
                              [1j*np.cos(theta/2), np.exp(-1j*phi)*np.sin(theta/2)]])
                U[k:k+2] = T.dot(U[k:k+2])
                if get_vw:
                    V[:, k:k+2] = V[:, k:k+2].dot(T.T.conj())

def matdec(U, dp, eta, method, out='err'):
    r"""
    Performs the Reck / Clements decomposition on a matrix U.  Computes the best realizable matrix U' and the
    normalized error E = |U' - U| / sqrt(N).
    :param U: Target matrix
    :param dp: Splitter errors (alpha, beta, gamma).  Array of dim=(N(N-1)/2, 3)
    :param eta: Coupling angle for third splitter.  Prominent choices: 0 -> MZI, pi/2 -> 3-MZI, pi -> MZI+X
    :param method: 'reck' or 'clements'
    :param out: 'err' or 'matrix'
    :return: Depends on out: 'err' -> (E), 'matrix' -> (E, U')
    """
    # Set up input
    U = np.array(U)
    V = np.eye(len(U), dtype=complex)
    W = np.eye(len(U), dtype=complex)
    N = len(U)
    get_vw = (out == 'matrix')
    matdec_helper = {'reck': reckdec_helper, 'clements': clemdec_helper}[method]

    # Call JIT function
    matdec_helper(U, V, W, dp, eta, get_vw)
    D = np.diag(U)
    err = np.linalg.norm(U - np.diag(D)) / np.sqrt(N)

    # Return required output.
    if out == 'err':
        return err
    elif out == 'matrix':
        return (err, V @ np.diag(D) @ W)
    else:
        raise ValueError(out)


def get_err(N, mu, sig, eta, ct=20, method='reck', share_u=True, share_dp=True):
    r"""
    Benchmarks error for the Reck/Clements decomposition.  Generates a set of Haar-random target matrices U, and
    for each (mu, sig) computes the mesh matrix error post-correction.

    :paran N: Matrix size
    :param mu: Mean beamsplitter error (list)
    :param sig: Std-dev beamsplitter error (list)
    :param eta: Coupling angle for third splitter.  Prominent choices: 0 -> MZI, pi/2 -> 3-MZI, pi -> MZI+X
    :param ct: Number of instances
    :param method: 'reck' or 'clements'
    :param share_u: Whether to use the same matrix U for each (mu, sig) in an instance.
    :param share_dp: Whether to use the same random splitter errors dp for each (mu, sig) in an instnace.
    :return: Normalized error |U' - U| / sqrt(N).
    """
    if np.iterable(N):
        return np.array([get_err(N_i, mu, sig, eta, ct, method, share_u, share_dp) for N_i in N])
    print (f"N = {N}")
    err_c = np.zeros([ct, len(mu)])
    dj = max(len(mu)//50, 1); di = max(ct//50, 1)
    dp_mask = np.array([[1,1,1]]); dp_mask[0,2] = (eta != 0)
    def dp_random():
        return np.random.randn(N*(N-1)//2, 3) * dp_mask

    for i in range(ct):
        if (share_u): U0 = np.array(unitary_group.rvs(N), order="F")
        if (share_dp): dp_r0 = dp_random()
        for (j, (mu_i, sig_i)) in enumerate(zip(mu, sig)):
            U = np.copy(U0, order="F") if share_u else np.array(unitary_group.rvs(N), order="F")
            dp = mu_i*dp_mask + sig_i*(dp_r0 if share_dp else dp_random())
            err_c[i, j] = matdec(U, dp, eta, method, 'err')
            if (len(mu) > 1) and (j % dj == 0): print ('.', end="", flush=True)
        if (len(mu) > 1): print(i)
        elif (i % di == 0): print ('.', end="", flush=True)
    if (len(mu) == 1): print()
    return err_c
