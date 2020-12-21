# Meshes package.
# Ryan Hamerly, 12/10/20
#
# Implements the MeshNetwork class, which can be used for describing Reck and Clements meshes (or any other
# beamsplitter mesh where couplings occur between neighboring waveguides).  Clements decomposition is also
# implemented.
#
# History
#   11/16/19: Conceived SquareNet, wrote some scripts to do Reck / Clements / SquareNet decomposition (Note3/main.py).
#   06/19/20: Made Reck / Clements code object-oriented using the MeshNetwork class.
#   07/09/20: Made SquareNet subclass of MeshNetwork.  Custom crossings.  Converted from module to package.
#   07/11/20: Custom crossings to Reck / Clements.  Reck / Clements decomposition in presence of imperfections.
#   12/10/20: Added Ratio Method strategy for calibrating SquareNet and Reck.
#   12/15/20: Extended Ratio Method to triangular and Clements meshes.
#   12/19/20: Added MZICrossingSym and Direct Method for tuning triangular and Clements meshes.

from .crossing import Crossing, MZICrossing
from .mesh import MeshNetwork, StructuredMeshNetwork
from .reck import ReckNetwork, reckdec
from .clements import ClementsNetwork, SymClementsNetwork, clemdec
from .square import SquareNetwork, SquareNetworkMZI, calibrateDiag, squaredec, squaremat

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