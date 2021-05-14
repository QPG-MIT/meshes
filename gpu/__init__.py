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
#   04/13/21: Added MeshNetworkGPU class (mesh.py) to interface with GPU code.


import numpy as np
import cupy as cp
from .mesh import MeshNetworkGPU
from .opt import MeshOptimizer

# Load the CUDA module.
mod = cp.RawModule(path=__path__[0]+"/meshprop.cubin")

# Load the optimal number of warps.
nwarps_nlist = []
nwarps_opt = dict()
for fname in ['fwdprop', 'fwddiff', 'backdiff']:
    for mode in ['mzi', 'sym', 'orth']:
        data = np.loadtxt(f"{__path__[0]}/benchmarks/{fname}_{mode}.txt", skiprows=1)
        data = data.reshape([2, data.shape[0]//2, data.shape[1]])
        for (data_i, shape) in zip(data, ['sq', 'fat']):
            nwarps_nlist = np.array(data_i[:, 0]).astype(int)
            for (N, warps) in data_i[:, [0, 3]]:
                nwarps_opt[(fname, mode, shape, int(N))] = int(warps)
