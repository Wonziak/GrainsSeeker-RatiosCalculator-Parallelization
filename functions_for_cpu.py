import cv2
from numba import njit, prange
import numpy as np
import math


@njit(parallel=True)
def create_lists_of_xs_ys_edge_cpu(edge):
    xs = np.zeros(len(edge))
    ys = np.zeros(len(edge))
    for i in prange(len(edge)):
        xs[i] = edge[i][0][0]
        ys[i] = edge[i][0][1]
    return xs, ys


@njit(parallel=True)
def create_lists_of_xs_ys_domain_cpu(domain):
    xs = np.zeros(len(domain))
    ys = np.zeros(len(domain))
    for i in prange(len(domain)):
        xs[i] = domain[i][0]
        ys[i] = domain[i][1]
    return xs, ys


@njit(parallel=True)
def calculate_distance_sum_from_center_cpu(domain, center_of_mass_0, center_of_mass_1):
    output = np.zeros(len(domain))
    for i in prange(len(domain)):
        output[i] = math.pow((center_of_mass_0 - domain[i][0]), 2) + math.pow((
                center_of_mass_1 - domain[i][1]), 2)
    return output


@njit(parallel=True)
def calculate_distance_from_center_to_edge_cpu(edge, center_of_mass_0, center_of_mass_1):
    output = np.zeros(len(edge))
    for i in prange(len(edge)):
        output[i] = math.pow((center_of_mass_0 - edge[i][0][0]), 2) + math.pow((
                center_of_mass_1 - edge[i][0][1]), 2)
    return output


@njit(parallel=True)
def get_sum_of_minimal_distance_from_each_point_to_edge_cpu(domain, edge):
    sum_of_minimal_distances = 0
    for i in prange(len(domain)):
        list_of_distances = np.zeros(len(edge))
        for j in prange(len(edge)):
            list_of_distances[j] = math.sqrt(
                math.pow(edge[j][0][0] - domain[i][0], 2) + math.pow(edge[j][0][1] - domain[i][1],
                                                                     2))
        sum_of_minimal_distances += min(list_of_distances)
    return sum_of_minimal_distances


@njit(parallel=True)
def get_all_perpendicular_vectors_length_cpu(edge, max_distance_vector_x,
                                             max_distance_vector_y, max_vector_length):
    vector = np.zeros(2)
    all_lengths = np.zeros(len(edge))
    for i in prange(len(edge)):
        min_cosine = 1
        length = 0
        for j in prange(len(edge)):
            if edge[j][0][0] == edge[i][0][0] and edge[j][0][1] == edge[i][0][1]:
                continue
            vector[0] = int(edge[i][0][0] - edge[j][0][0])
            vector[1] = int(edge[i][0][1] - edge[j][0][1])
            vector_length = math.sqrt(math.pow(vector[0], 2) + math.pow(vector[1], 2))

            numerator = ((vector[0] * max_distance_vector_x) + (vector[1] * max_distance_vector_y))
            denominator = vector_length * max_vector_length
            cosa = abs(numerator / denominator)
            if cosa < min_cosine:
                min_cosine = cosa
                length = vector_length
        all_lengths[i] = length

    return np.amax(all_lengths)
