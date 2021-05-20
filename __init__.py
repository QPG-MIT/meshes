# Meshes package.
# Ryan Hamerly, 5/19/21
#
# Implements the MeshNetwork class, which can be used for describing Reck and Clements meshes (or any other
# beamsplitter mesh where couplings occur between neighboring waveguides).  Clements decomposition is also
# implemented.
#
# History
#   12/12/18: First attempt at GPU code for mesh propagation.  Fast but not very extensible, soon abandoned.
#   11/16/19: Conceived SquareNet, wrote some scripts to do Reck / Clements / SquareNet decomposition (Note3/main.py).
#   06/19/20: Made Reck / Clements code object-oriented using the MeshNetwork class.
#   07/09/20: Made SquareNet subclass of MeshNetwork.  Custom crossings.  Converted from module to package.
#   07/11/20: Custom crossings for Reck / Clements.  Reck / Clements decomposition in presence of imperfections.
#   12/10/20: Added Ratio Method strategy for calibrating SquareNet and Reck.
#   12/15/20: Extended Ratio Method to triangular and Clements meshes.
#   12/19/20: Added MZICrossingSym and Direct Method for tuning triangular and Clements meshes.
#   01/05/21: Added new Clements tuning method (diagnoalization).
#   03/03/21: Extended "new" (matrix diagonalization) method to Reck and general mesh shapes.
#   03/06/21: Revamped SquareNet.  Added the triangular QR mesh.
#   03/22/21: Created ButterflyNetwork class and added the local error correction routine errcorr_local
#   03/29/21: Added CartesianCrossing class, crossing conversion utilities, tweaks to gradient function, Hessians.
#   04/03/21: First working CUDA code for mesh propagation.
#   04/06/21: Added forward- and reverse-mode differentiation to the CUDA code.
#   04/10/21: Added symmetric and real orthogonal crossings to the CUDA code.
#   04/13/21: Python interface to the CUDA code via MeshNetworkGPU class.
#   05/19/21: Added CUDA code for butterfly networks.

from .crossing import Crossing, MZICrossing, SymCrossing, CartesianCrossing
from .mesh import MeshNetwork, StructuredMeshNetwork
from .reck import ReckNetwork
from .clements import ClementsNetwork, SymClementsNetwork
from .square import SquareNetwork
from .qr import QRNetwork
from .butterfly import ButterflyNetwork
from .configure import errcorr_local

import scipy.stats


# Miscellaneous code.  No good place to put this right now...

# Haar random unitaries.

def haar_mat(N: int):
    r"""
    Returns an NxN Haar-random unitary matrix
    :param N: Matrix size.
    :return: NxN matrix.
    """
    return scipy.stats.unitary_group.rvs(N)