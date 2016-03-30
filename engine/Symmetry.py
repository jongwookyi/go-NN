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

def apply_random_symmetries(many_feature_planes, many_move_arrs):
    N = many_feature_planes.shape[1]
    for i in range(many_feature_planes.shape[0]):
        s = random.randint(0, 7)
        apply_symmetry_planes(many_feature_planes[i,:,:,:], s)
        apply_symmetry_vertex(many_move_arrs[i,:], N, s)

