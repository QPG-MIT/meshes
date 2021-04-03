# meshes/gpu
# Ryan Hamerly, 4/3/21
#
# CUDA implementation of light propagation through a beamsplitter mesh.  The NumPy CPU versions are convenient
# for small meshes (N < 64) but do not scale well when the mesh size exceeds the CPU cache..  This code is 
# designed to achieve high local memory reuse and arithmetic intensity for meshes up to size N = 1024.
#
# History
#   12/12/18: First attempt at GPU code for beamsplitter meshes.  Not very extensible, soon abandoned.
#   04/02/21: Revived the idea.  New CUDA code for forward propagation (meshprop.cu, fwdprop.cu).
#   04/03/21: Added testing utility test.py.


# TODO -- fill in with real code.
raise NotImplementedError("Sorry!  This package is not implemented yet.")
