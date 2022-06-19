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
