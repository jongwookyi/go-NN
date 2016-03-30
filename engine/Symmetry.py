import numpy as np
import random

def apply_symmetry_planes(planes, s):
    assert len(planes.shape) == 3
    if (s & 1) != 0: # flip x
        np.copyto(planes, planes[::-1,:,:])
    if (s & 2) != 0: # flip y
        np.copyto(planes, planes[:,::-1,:])
    if (s & 4) != 0: # swap x and y
        np.copyto(planes, np.transpose(planes[:,:,:], (1,0,2)))

def apply_symmetry_plane(plane, s):
    assert len(plane.shape) == 2
    if (s & 1) != 0: # flip x
        np.copyto(plane, plane[::-1,:])
    if (s & 2) != 0: # flip y
        np.copyto(plane, plane[:,::-1])
    if (s & 4) != 0: # swap x and y
        np.copyto(plane, np.transpose(plane[:,:], (1,0)))

def invert_symmetry_plane(plane, s):
    assert len(plane.shape) == 2
    # note reverse order of 4,2,1
    if (s & 4) != 0: # swap x and y
        np.copyto(plane, np.transpose(plane[:,:], (1,0)))
    if (s & 2) != 0: # flip y
        np.copyto(plane, plane[:,::-1])
    if (s & 1) != 0: # flip x
        np.copyto(plane, plane[::-1,:])

def apply_symmetry_vertex(vertex, N, s):
    assert vertex.size == 2
    if (s & 1) != 0: # flip x
        vertex[0] = N - vertex[0] - 1
    if (s & 2) != 0: # flip y
        vertex[1] = N - vertex[1] - 1
    if (s & 4) != 0: # swap x and y
        np.copyto(vertex, vertex[::-1])
    assert 0 <= vertex[0] < N
    assert 0 <= vertex[1] < N


