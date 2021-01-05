# meshes/mesh.py
# Ryan Hamerly, 7/11/20
#
# Implements the MeshNetwork class, which can be used for describing Reck and Clements meshes (or any other
# beamsplitter mesh where couplings occur between neighboring waveguides).  Clements decomposition is also
# implemented.
#
# History
#   06/19/20: Made Reck / Clements code object-oriented using the MeshNetwork class (meshes.py).
#   07/09/20: Turned module into package, renamed file mesh.py.
#   07/10/20: Added compatibility for custom crossings in crossing.py.
#   12/15/20: Added Ratio Method tuning triangular meshes.
#   12/19/20: Added Direct Method for tuning triangular meshes.


import autograd.numpy as npa
import numpy as np
import warnings
from typing import List, Any
from .crossing import Crossing, MZICrossing, MZICrossingOutPhase


# The base class MeshNetwork.

class MeshNetwork:
    p_phase: npa.array
    p_splitter: npa.array

    @property
    def L(self) -> int:
        r"""
        Number of layers in the mesh network.
        """
        raise NotImplementedError()
    @property
    def M(self) -> int:
        r"""
        Output dimension of the mesh network.  Thus, self.matrix().shape == (self.M, self.N)
        """
        raise NotImplementedError()
    @property
    def N(self) -> int:
        r"""
        Output dimension of the mesh network.  Thus, self.matrix().shape == (self.M, self.N)
        """
        raise NotImplementedError()
    def dot(self, v, p_phase=None, p_splitter=None) -> np.ndarray:
        r"""
        Computes the dot product between the splitter and a vector v.
        :param v: Input vector / matrix.
        :param p_phase: Phase parameters.  Defaults to stored values.
        :param p_splitter: Splitter angle parameters (deviation from pi/4).  Defaults to stored values.
        :return: Output vector / matrix.
        """
        raise NotImplementedError()
    def matrix(self, p_phase=None, p_splitter=None) -> np.ndarray:
        r"""
        Computes the input-output matrix.  Equivalent to self.dot(np.eye(self.N))
        :param p_phase: Phase parameters.  Defaults to stored values.
        :param p_splitter: Splitter angle parameters (deviation from pi/4).  Defaults to stored values.
        :return: NxN matrix.
        """
        return self.dot(np.eye(self.N), p_phase, p_splitter)
    def grad_phi(self, v, w, p_phase=None, p_splitter=None) -> np.ndarray:
        r"""
        Computes the gradient with respect to phase shifts phi.
        :param v: Input matrix / vector (normally np.eye(M)), size (M, ...)
        :param w: Output gradient matrix / vector dJ/d(U*), size (N, ...)
        :param p_phase: Phase parameters (phi_k).  Scalar or vector of size N^2.
        :param p_splitter: Beamsplitter parameters (beta_k - pi/4).  Scalar of vector of size N(N-1).
        :return:
        """
        raise NotImplementedError()
    def grad_phi_target(self, U_target, p_phase=None, p_splitter=None) -> np.ndarray:
        r"""
        Gets the gradient of the L2 norm |U - U_target|^2 with respect to the phase parameters.
        :param U_target: Target matrix, size (M, N)
        :param p_phase: Phase parameters (phi_k).  Scalar or vector of size N^2.
        :param p_splitter: Beamsplitter parameters (beta_k - pi/4).  Scalar of vector of size N(N-1).
        :return:
        """
        J = [0]
        def f(U):
            V = U - U_target; J[0] = np.linalg.norm(V)**2; return V
        grad = 2*np.real(self.grad_phi(np.eye(self.M), f, p_phase, p_splitter)[0])
        return (J[0], grad)



class StructuredMeshNetwork(MeshNetwork):
    _N: int
    inds: List[int]
    lens: List[int]
    shifts: List[int]
    X: Crossing
    phi_pos: str

    def __init__(self,
                 N: int,
                 lens: List[int],
                 shifts: List[int],
                 p_phase: Any=0.,
                 p_splitter: Any=0.,
                 p_crossing=None,
                 phi_out=None,
                 X: Crossing=MZICrossing(),
                 phi_pos='out'):
        r"""
        Manages the beamsplitters for a Clements beamsplitter mesh.
        :param N: Number of inputs / outputs.
        :param lens: List of number of MZM's in each column.
        :param shifts: List of the shifts for each column.
        :param p_phase: Parameters [phi_i] for phase shifters.
        :param p_splitter: Parameters [beta_i - pi/4] for beam splitters.
        """
        self.X = X
        self._N = N
        assert phi_pos in ['in', 'out']
        self.lens = lens; n_cr = sum(lens)
        self.shifts = shifts
        self.inds = np.cumsum([0] + list(lens)).tolist()
        self.p_phase = p_phase * np.ones(n_cr*X.n_phase + N, dtype=np.float)
        self.p_splitter = np.array(p_splitter)
        self.phi_pos = phi_pos
        if not (p_crossing is None): self.p_crossing[:] = p_crossing
        if not (phi_out is None): self.phi_out[:] = phi_out
        assert len(shifts) == len(lens)
        assert self.p_phase.shape in [(), (n_cr*X.n_phase + self.N,)]
        assert self.p_splitter.shape in [(), (n_cr, X.n_splitter)]

    @property
    def L(self):
        return len(self.lens)
    @property
    def M(self):
        return self._N
    @property
    def N(self):
        return self._N

    @property
    def n_cr(self):
        r"""
        Number of 2x2 crossings in the beamsplitter mesh.
        :return: int
        """
        return sum(self.lens)
    @property
    def p_crossing(self):
        r"""
        Returns the segment of self.p_phase that encodes the 2x2 crossing parameters.
        :return: array of size (self.n_cr, self.X.n_phase)
        """
        return self._get_p_crossing(None)
    @p_crossing.setter
    def p_crossing(self, p):
        self.p_crossing[:] = p
    def _get_p_crossing(self, p_phase=None):
        return (self.p_phase if (p_phase is None) else p_phase)[:-self.N].reshape([self.n_cr, self.X.n_phase])
    @property
    def phi_out(self):
        r"""
        Returns the segment of self.p_phase that encodes the output phases.
        :return: array of size (self.N,)
        """
        return self._get_phi_out(None)
    @phi_out.setter
    def phi_out(self, p):
        self.phi_out[:] = p
    def _get_phi_out(self, p_phase=None):
        return (self.p_phase if (p_phase is None) else p_phase)[-self.N:]

    def _defaults(self, p_phase, p_splitter, p_crossing, phi_out):
        # Gets default values assuming certain inputs and current state of the mesh.
        assert (p_crossing is None) == (phi_out is None)
        p_splitter = (self.p_splitter if (p_splitter is None) else p_splitter) * np.ones([self.n_cr, self.X.n_splitter])
        if (p_crossing is not None):
            assert (p_crossing.shape == (self.n_cr, self.X.n_phase)) and (phi_out.shape == (self.N,))
            return (p_crossing, phi_out, p_splitter)
        else:
            return (self._get_p_crossing(p_phase), self._get_phi_out(p_phase), p_splitter)

    def dot(self, v, p_phase=None, p_splitter=None, p_crossing=None, phi_out=None):
        v = np.array(v, dtype=np.complex)
        (p_crossing, phi_out, p_splitter) = self._defaults(p_phase, p_splitter, p_crossing, phi_out)
        # Loop through the crossings, one row at a time.  Then apply the final phase shifts.
        if (self.phi_pos == 'in'): v *= np.exp(1j*phi_out).reshape((self.N,) + (1,)*(v.ndim-1))
        for (i1, i2, L, s) in zip(self.inds[:-1], self.inds[1:], self.lens, self.shifts):
            v[s:s+2*L] = self.X.dot(p_crossing[i1:i2], p_splitter[i1:i2], v[s:s+2*L])
        if (self.phi_pos == 'out'): v *= np.exp(1j*phi_out).reshape((self.N,) + (1,)*(v.ndim-1))
        return v

    def grad_phi(self, v, w, p_phase=None, p_splitter=None, p_crossing=None, phi_out=None):
        assert self.phi_pos == 'out'  # TODO -- handle phase shifts on inputs too.
        (p_crossing, phi_out, p_splitter) = self._defaults(p_phase, p_splitter, p_crossing, phi_out)
        vList = np.zeros((len(self.lens)+1,) + v.shape, dtype=np.complex); vList[0] = v
        grad = np.zeros(len(self.p_phase), dtype=np.float)
        grad_crossing = grad[:-self.N].reshape((self.n_cr, self.X.n_phase)); grad_phiout = grad[-self.N:]
        # Forward pass
        for (n, i1, i2, L, s) in zip(range(self.L), self.inds[:-1], self.inds[1:], self.lens, self.shifts):
            vList[n+1, s:s+2*L] = self.X.dot(p_crossing[i1:i2], p_splitter[i1:i2], vList[n, s:s+2*L])
            vList[n+1, :s] = vList[n, :s]; vList[n+1, s+2*L:] = vList[n, s+2*L:]
        v_out = vList[-1] * np.exp(1j*phi_out).reshape((self.N,) + (1,)*(v.ndim-1))
        # Can use callable w to specify a derivative based on the output matrix v_out.
        w = w(v_out) if callable(w) else np.array(w)
        # Reverse pass
        grad_phiout[:] = np.real(-1j * w * v_out.conj()).sum(-1)
        w *= np.exp(-1j*phi_out).reshape((self.N,) + (1,)*(v.ndim-1))
        for n in range(len(self.lens)-1, -1, -1):
            (i1, i2, L, s) = (self.inds[n], self.inds[n+1], self.lens[n], self.shifts[n])
            grad_crossing[i1:i2] = self.X.grad(p_crossing[i1:i2], p_splitter[i1:i2], vList[n, s:s+2*L], w[s:s+2*L])
            w[s:s+2*L] = self.X.dot(p_crossing[i1:i2], p_splitter[i1:i2], w[s:s+2*L], True)
        return (grad, w)

    def copy(self) -> 'StructuredMeshNetwork':
        r"""
        Returns a copy of the mesh.
        :return:
        """
        return StructuredMeshNetwork(self.N, self.lens.copy(), self.shifts.copy(), np.array(self.p_phase),
                                     np.array(self.p_splitter), X=self.X, phi_pos=self.phi_pos)

    def flip_crossings(self, inplace=False) -> 'StructuredMeshNetwork':
        r"""
        Flips the mesh, i.e. from one with output phase shifters to input phase shifters and vice versa.  Only works
        for MZI meshes.
        :param inplace: Performs the operation inplace or returns a copy.
        :return: The flipped StructuredMeshNetwork object.
        """
        if not inplace:
            self = self.copy()

        if (self.phi_pos == 'out'):
            assert isinstance(self.X, MZICrossing)
            self.X = MZICrossingOutPhase()
            for (m, len, shift, ind) in list(zip(range(self.L), self.lens, self.shifts, self.inds))[::-1]:
                phi = self.p_crossing[ind:ind+len, 1]
                (psi1, psi2) = self.phi_out[shift:shift+2*len].reshape([len, 2]).T
                (phi[:], psi1[:], psi2[:]) = np.array([psi2-psi1, phi+psi1, psi1])
            self.phi_pos = 'in'
        else:
            assert isinstance(self.X, MZICrossingOutPhase)
            self.X = MZICrossing()
            for (m, len, shift, ind) in list(zip(range(self.L), self.lens, self.shifts, self.inds)):
                phi = self.p_crossing[ind:ind+len, 1]
                (psi1, psi2) = self.phi_out[shift:shift+2*len].reshape([len, 2]).T
                (phi[:], psi1[:], psi2[:]) = np.array([psi1-psi2, psi2, phi+psi2])
            self.phi_pos = 'out'
        return self

    def flip(self, inplace=False) -> 'StructuredMeshNetwork':
        r"""
        Flips the entire network geometry about both position and time axes.  The resulting transfer matrix
        transforms as U -> U[::-1,::-1].T
        :param self:
        :param inplace: Perform the operation inplace.
        :return:
        """
        if not inplace:
            self = StructuredMeshNetwork(self.N, self.lens.copy(), self.shifts.copy(), np.array(self.p_phase),
                                         np.array(self.p_splitter), X=self.X, phi_pos=self.phi_pos)
        assert isinstance(self.X, MZICrossing) or isinstance(self.X, MZICrossingOutPhase)
        self.X = self.X.flip()
        self.phi_pos = {'in': 'out', 'out': 'in'}[self.phi_pos]
        self.p_splitter[:] = (self.p_splitter + np.zeros([self.n_cr, 2]))[:, ::-1] + np.pi/2*np.array([[1, -1]])
        self.p_splitter[:] = self.p_splitter[::-1]
        self.p_crossing = self.p_crossing[::-1]
        self.phi_out = self.phi_out[::-1]
        self.lens = self.lens[::-1]; self.shifts = (self.N-2*np.array(self.lens)-self.shifts[::-1]).tolist()
        self.inds = np.cumsum([0] + self.lens).tolist()

        return self

    @property
    def grid_phase(self) -> np.ndarray:
        r"""
        Arranges the data of self.p_crossing onto a 2D array.
        :return:
        """
        out = np.zeros([self.L, (self.N+1)//2, self.X.n_phase]) * np.nan
        for (i, j0, nj, ind) in zip(range(self.L), self.shifts, self.lens, self.inds):
            out[i, j0//2:j0//2+nj, :] = self.p_crossing[ind:ind+nj, :]
        return out

    @property
    def grid_splitter(self) -> np.ndarray:
        r"""
        Arranges the data of self.p_splitter onto a 2D array.
        :return:
        """
        out = np.zeros([self.L, (self.N+1)//2, self.X.n_splitter]) * np.nan
        for (i, j0, nj, ind) in zip(range(self.L), self.shifts, self.lens, self.inds):
            out[i, j0//2:j0//2+nj, :] = self.p_splitter[ind:ind+nj, :]
        return out


def calibrateTriangle(mesh: StructuredMeshNetwork, U, diag, method, warn=False):
    r"""
    A general Ratio Method to calibrate an arbitrarily shaped triangular mesh.
    :param mesh: Mesh to configure.
    :param U: Target matrix.
    :param diag: Direction of diagonals ['up' or 'down']
    :param method: Programming method ['direct' or 'ratio']
    :param warn: Warns the presence of errors.
    :return:
    """
    if (diag == 'up') and (mesh.phi_pos == 'out'):
        mesh.flip(True); calibrateTriangle(mesh, U, 'down', method, warn); mesh.flip(True); return
    assert (diag == 'down') and (mesh.phi_pos == 'in')
    X = {'direct': mesh.X.flip(), 'mod': mesh.X.flip(), 'ratio': mesh.X}[method]
    N = len(U); Upost = np.eye(N, dtype=np.complex); assert U.shape == (N, N)

    # Divide mesh into diagonals.  Make sure each diagonal can be uniquely targeted with a certain input waveguide.
    pos = np.concatenate([[[i, j, 2*j+j0, i-(2*j+j0)]
                           for j in range(mesh.lens[i])] for (i, j0) in enumerate(mesh.shifts)])
    pos = pos[np.lexsort(pos[:, ::3].T)][::-1]
    pos = np.split(pos, np.where(np.roll(pos[:,3], 1) != pos[:,3])[0])[1:]
    assert (np.diff(np.concatenate([pos_i[-1:] for pos_i in pos])[:, 2]) > 0).all()  # Diagonals have distinct inputs.
    if not isinstance(mesh.X, MZICrossingOutPhase):
        raise NotImplementedError("Only MZICrossingOutPhase supported for now.")
    p_splitter = mesh.p_splitter * np.ones([mesh.n_cr, 2])
    phi_out = 0*mesh.phi_out

    err = 0
    if method in ['direct', 'mod']:
        p = np.array([[2*j+j0, i+(2*j+j0)] for (i, j0) in enumerate(mesh.shifts) for j in range(mesh.lens[i])])
        p = p[np.lexsort(p.T)]; outfield = dict(p[np.where(np.roll(p[:, 1], 1) != p[:, 1])[0]][:, ::-1])  # out-fields
        X = mesh.X.flip(); assert ('T11' in X.tunable_indices);
        env = np.maximum.accumulate((np.array(mesh.shifts) + np.array(mesh.lens)*2 - 2)[::-1])[::-1]
        Psum = np.cumsum(np.abs(U[::-1]**2), axis=0)[::-1]  # Psum[i, j] = norm(U[i:, j])^2
        for pos_i in pos:
            E_in = 1.0; Tlist = []; ptr_last = None; l = pos_i[-1, 2]; #v = np.zeros([N], dtype=np.complex)
            w = np.zeros([N], dtype=np.complex)
            for (m, (i, j, ind, _)) in enumerate(pos_i[::-1]):  # Adjust MZIs *down* the diagonal one by one.
                ptr = mesh.inds[i] + j
                k = outfield[i+ind] if (i+ind in outfield) else ind  # Output index (or two) k:k+dk
                dk = (min(2, outfield[i+ind+2]-k) if (i+ind+2 in outfield) else 2) if (i < mesh.L-1) else 1
                # T11 = (U[k:k+dk, l] - Upost[k:k+dk, :].dot(v)).sum()/(E_in*Upost[k:k+dk, ind]).sum()
                U_kl = np.array(U[k:k+dk, l])
                if (method == 'mod'):
                    U_kl *= np.sqrt((np.linalg.norm(w[k:])**2 + np.abs(E_in)**2) / Psum[k, l])
                T11 = (U_kl - w[k:k+dk]).sum()/(E_in*Upost[k:k+dk, ind]).sum()
                ((theta, phi), d_err) = X.Tsolve(T11, 'T11', p_splitter[ptr]); err += d_err
                mesh.p_crossing[ptr, 0] = theta
                if m: mesh.p_crossing[ptr_last, 1] = phi  # Set theta for crossing & phi to *upper-left* of crossing.
                else: phi_out[ind] = phi
                T = X.T((theta, phi), p_splitter[ptr]); Tlist.append(T)
                # v[ind] += E_in * T[0, 0];
                w += E_in * T[0, 0] * Upost[:, ind]
                E_in *= T[1, 0]
                phi_out[ind+1] = np.angle(U[k, ind+1]/(Upost[k, ind]*T[0, 1]))
                ptr_last = ptr
            k = (ind+1) if (i == mesh.L-1 or ind > env[i+1]) else outfield[i+ind+2]
            dk = (min(2, outfield[i+ind+4]-k) if (i+ind+4 in outfield) else 2) if (i < mesh.L-1) else 1
            # Set final phase shift.
            # T11 = (U[k:k+dk,l] - Upost[k:k+dk,:].dot(v))/(E_in * Upost[k,ind+1])
            # mesh.p_crossing[ptr, 1] = phi = np.angle((U[k:k+dk,l] - Upost[k:k+dk,:].dot(v)).sum()/
            #                                          (E_in * Upost[k:k+dk,ind+1]).sum())
            T11 = (U[k:k+dk,l] - w[k:k+dk])/(E_in * Upost[k,ind+1])
            mesh.p_crossing[ptr,1] = phi = np.angle((U[k:k+dk,l] - w[k:k+dk]).sum()/ (E_in * Upost[k:k+dk,ind+1]).sum())
            Upost[:, ind+1] *= np.exp(1j*phi)
            for (ind, T) in zip(pos_i[:, 2], Tlist[::-1]):
                Upost[:, ind:ind+2] = Upost[:, ind:ind+2].dot(T)   # Multiply Upost by diagonal's T.

    elif method == 'ratio':
        X = mesh.X; assert ('T:1' in X.tunable_indices)
        for pos_i in pos:
            for (i, j, ind, _) in pos_i:  # Adjust MZIs *up* the diagonal one by one.
                ptr = mesh.inds[i] + j
                bc = Upost[:, ind:ind+2]  # Vectors [b, c]
                (T11, T21) =  bc.conj().T.dot(U[:, pos_i[-1][2]])   # Targets: (T11, T21) ~ (b* u_i, c* u_i)
                ((theta, phi), d_err) = X.Tsolve((T11, T21), 'T:1', p_splitter[ptr]); err += d_err
                mesh.p_crossing[ptr, :] = (theta, phi)  # Set theta for crossing & phi to *lower-right* of crossing.
                T = X.T((theta, phi), p_splitter[ptr])
                Upost[:, ind:ind+2] = Upost[:, ind:ind+2].dot(T)
            # Set input phases at the top of the diagonal.
            phi_out[ind:ind+2] = np.angle((Upost[:, ind:ind+2].conj() * U[:, ind:ind+2]).sum(0))

    mesh.phi_out[:] = phi_out
    if (warn and err):
        warnings.warn(
            "Mesh calibration: {:d}/{:d} matrix values could not be set.".format(err, len(mesh.p_crossing)))
