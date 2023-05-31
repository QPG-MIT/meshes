# meshes/svd.py
# Ryan Hamerly, 5/29/23
#
# Containes code for SVDNetwork the class implementing SVD+Reck and SVD+Clements meshes.
#
# History
#   05/29/23: Created this file.

import numpy as np
from typing import Any
from .mesh import MeshNetwork, StructuredMeshNetwork, calibrateTriangle
from .reck import ReckNetwork
from .clements import SymClementsNetwork
from .crossing import Crossing, MZICrossing


class SVDNetwork(MeshNetwork):
    _N: int
    fact: float
    m1: StructuredMeshNetwork
    m2: StructuredMeshNetwork

    def __init__(self,
                 geom: str,
                 M: np.ndarray,
                 eig: float      = None,
                 p_splitter: Any = 0.,
                 X: Crossing     = MZICrossing(),
                 method          = 'diag'):
        r"""
        Mesh that realizes arbitrary non-unitary matrices with the SVD+Reck or SVD+Clements decomposition:
        M = U*D*V.  Here, (U, V) are unitary matrices and D is a diagonal matrix of singular values.  Passive meshes
        can only realize |D| <= 1.
        :param geom: Mesh used to implement the unitary blocks (U, V), 'reck' or 'clements'
        :param M: Target matrix.
        :param p_splitter: Splitter imperfections, an array of size (N*(N-1), X.n_splitter).
        :param X: Crossing type.
        """
        assert (geom in ['reck', 'clements'])
        assert (M.shape == (M.shape[0], M.shape[0])); N = self._N = M.shape[0]
        (v, d, u) = np.linalg.svd(M)
        self.fact = eig / np.max(np.abs(d)) if eig else 1.0; d *= self.fact
        self.p_splitter = p_splitter * np.ones([N*(N-1)+N, X.n_splitter])
        (ps_m1, ps_m2, ps_d) = np.split(self.p_splitter, [N*(N-1)//2, N*(N-1)])
        if (geom == 'reck'):
            m1 = ReckNetwork(M=u[::-1,::-1].T, X=X, p_splitter=ps_m1, method=method).flip().flip_crossings()
            m2 = ReckNetwork(M=v, X=X, p_splitter=ps_m2, method=method)
        else:
            m1 = SymClementsNetwork(M=u, X=X, p_splitter=ps_m1, method=method)
            m2 = SymClementsNetwork(M=v, X=X, p_splitter=ps_m2, method=method)
        self.p_phase = np.zeros([len(m1.p_phase)+len(m2.p_phase) + X.n_phase*N])
        (pp_m1, pp_m2, pp_d) = np.split(self.p_phase, [len(m1.p_phase), len(m1.p_phase)+len(m2.p_phase)])
        ps_m1[:] = m1.p_splitter; ps_m2[:] = m2.p_splitter; pp_m1[:] = m1.p_phase; pp_m2[:] = m2.p_phase
        m1.p_splitter = ps_m1; m2.p_splitter = ps_m2; m1.p_phase = pp_m1; m2.p_phase = pp_m2
        pp_diag = np.array(X.Tsolve(d, 'T11', ps_d.T)[0]).T
        pp_d[:] = np.array(X.Tsolve(d, 'T11', ps_d.T)[0]).T.flatten()
        self.m1 = m1; self.m2 = m2

    @property
    def L(self) -> int:
        return 2*self.N+1
    @property
    def M(self) -> int:
        return self.N
    @property
    def N(self) -> int:
        return self._N

    def dot(self, v, p_phase=None, p_splitter=None) -> np.ndarray:
        if p_phase is None: p_phase = self.p_phase
        if p_splitter is None: p_splitter = self.p_splitter
        N = self.N
        (ps_m1, ps_m2, ps_d) = np.split(self.p_splitter, [N*(N-1)//2, N*(N-1)])
        (pp_m1, pp_m2, pp_d) = np.split(self.p_phase, [len(self.m1.p_phase), len(self.m1.p_phase)+len(self.m2.p_phase)])
        v = self.m1.dot(v, p_splitter=ps_m1, p_phase=pp_m1)
        v *= self.m1.X.T(pp_d.reshape([N, 2]), ps_d)[0, 0].reshape((N,) + (1,)*(v.ndim - 1))
        return self.m2.dot(v, p_splitter=ps_m2, p_phase=pp_m2)

