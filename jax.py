# meshes/jax.py
# Ryan Hamerly, 7/25/22
#
# JAX-differentiable functions for field propagation through the mesh.  This module exists separate from the rest of
# Meshes so that one can still import Meshes without JAX installed.
#
# History
#   07/25/22: Created this file.


from functools import partial
from jax import custom_vjp
import numpy as np

@partial(custom_vjp, nondiff_argnums=(0,))
def dot(mesh, p, x):
    r"""
    Differentiable function to compute the dot product mesh.dot(x, p_phase=p).
    :param mesh: StructuredMeshNetwork instance.
    :param p: The programmable mesh parameters (phase shifts), dim=(N**2) real for Reck / Clements / Butterfly nets.
    :param x: The input field, dim=(N, K) complex.
    :return:
    """
    return mesh.dot(np.array(x), p_phase=np.array(p))

def _dot_fwd(mesh, p, x):
    y = dot(mesh, p, x)
    return (y, (p, y))
def _dot_bwd(mesh, res, dJdy):
    # Quantities dJdy and dJdx are conjugated because JAX operates on dJ/dx, dJ/dy, while Meshes operates on
    # dJ/dx*, dJ/dy*.
    (p, y) = res
    (dJdp, dJdx) = mesh.dot_vjp(np.array(y), np.array(dJdy).conj(), p_phase=np.array(p))
    return (dJdp, dJdx.conj())

dot.defvjp(_dot_fwd, _dot_bwd)