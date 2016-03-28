#!/usr/bin/python
import numpy as np
import os
import math
import sys
import math

npz_dir = '/home/greg/coding/ML/go/NN/data/KGS/processed/stones_3lib_4hist_ko_Nf15-randomized-2'

N = 19
Nfeat = 15

sum_of_sample_feature_means = np.zeros((Nfeat,))
sum_of_sample_grand_means = 0.0

sum_of_sample_feature_mean_squares = np.zeros((Nfeat,))
sum_of_sample_grand_mean_squares = 0.0

Nsamples = 100
Nperfile = 128
big_matrix = np.empty((Nsamples*Nperfile*N*N, Nfeat))

sample_num = 0
for fn in os.listdir(npz_dir):
    filename = os.path.join(npz_dir, fn)
    npz = np.load(filename)
    features = npz['feature_planes'].astype(np.float64)
    npz.close()

    assert features.shape[3] == Nfeat

    big_matrix[sample_num*Nperfile*N*N:(sample_num+1)*Nperfile*N*N, :] = features.reshape(Nperfile*N*N, Nfeat)

    #features -= 0.154
    #features *= 2.77

    # using mean(ones) = 0.682 = 19^2 / 23^2, corresponding scaling factor is 1/sqrt(0.682(1-0.682)) = 2.148
    #features -= np.array([0.146, 0.148, 0.706, 0.682, 0.005, 0.018, 0.124, 0.004, 0.018, 0.126, 0.003, 0.003, 0.003, 0.003, 0])
    #features *= np.array([2.8292491, 2.8175156, 2.1945873, 2.148, 10, 7.5041088, 3.0369993, 10, 7.5756124, 3.0131227, 10, 10, 10, 10, 10])

    sum_of_sample_feature_means += features.mean(axis=(0,1,2))
    sum_of_sample_grand_means += features.mean()

    sum_of_sample_feature_mean_squares += np.square(features).mean(axis=(0,1,2))
    sum_of_sample_grand_mean_squares += np.mean(np.square(features))

    sample_num += 1
    if sample_num >= Nsamples: break

feature_means = sum_of_sample_feature_means / Nsamples
grand_mean = sum_of_sample_grand_means / Nsamples

feature_variances = sum_of_sample_feature_mean_squares / Nsamples - np.square(feature_means)
grand_variance = sum_of_sample_grand_mean_squares / Nsamples - grand_mean**2

feature_rescaling_factors = np.reciprocal(np.sqrt(feature_variances))

def print_arr(arr):
    sys.stdout.write("[")
    for i in xrange(arr.size):
        sys.stdout.write("%.7f" % arr[i])
        if i < arr.size - 1: sys.stdout.write(', ')
    sys.stdout.write("]\n")


print "feature_means =\n", feature_means
print_arr(feature_means)
print "feature_variances =\n", feature_variances
print_arr(feature_variances)
print "feature rescaling factors =\n", feature_rescaling_factors
print_arr(feature_rescaling_factors)

print
print "grand_mean =\n", grand_mean
print "variance =\n", grand_variance
print "overall rescaling factor =\n", 1/math.sqrt(grand_variance)

print
print "SVD"
print


U, diagS, V = np.linalg.svd(big_matrix, full_matrices=False)

scaledS = diagS / math.sqrt(big_matrix.shape[0])
print "rescaled diagS =\n", scaledS

print "V =\n", V




print
for f in xrange(Nfeat):
    print "stddev(?) of SVD vector", f, "is", scaledS[f]
    print "weighting on each feature is\n", V[f,:]


invS = np.diag(np.reciprocal(scaledS))

truncated_Nfeat = 12
truncated_invS = invS[:, :truncated_Nfeat]

print
print "truncated_invS =\n", truncated_invS

whitener = np.dot(V.T, truncated_invS)

print
print "whitener =\n", whitener
