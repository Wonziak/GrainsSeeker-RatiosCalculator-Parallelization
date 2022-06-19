import math

import numpy as np
from numba import cuda


@cuda.jit
def calculate_distance_sum_from_center_gpu(domain, output, center_of_mass_0, center_of_mass_1):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, len(domain), stride):
        output[i] = math.pow((center_of_mass_0 - domain[i][0]), 2) + math.pow((
                center_of_mass_1 - domain[i][1]), 2)


@cuda.jit
def calculate_distance_from_center_to_edge_gpu(edge, output, center_of_mass_0, center_of_mass_1):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, len(edge), stride):
        output[i] = math.pow((center_of_mass_0 - edge[i][0][0]), 2) + math.pow((
                center_of_mass_1 - edge[i][0][1]), 2)


@cuda.jit
def create_lists_of_xs_ys_edge_gpu(edge, xs, ys):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, len(edge), stride):
        xs[i] = edge[i][0][0]
        ys[i] = edge[i][0][1]


@cuda.jit
def create_lists_of_xs_ys_domain_gpu(domain, xs, ys):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, len(domain), stride):
        xs[i] = domain[i][0]
        ys[i] = domain[i][1]


@cuda.jit
def get_sum_of_minimal_distance_from_each_point_to_edge(edge, distances, x, y):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, len(edge), stride):
        if edge[i][0][0] == x and edge[i][0][1] == y:
            continue
        distances[i] = math.sqrt(
            math.pow(edge[i][0][0] - x, 2) + math.pow(edge[i][0][1] - y, 2))


@cuda.jit
def get_all_perpendicular_vectors_length(edge, vector, distances, x, y, max_distance_vector_x,
                                         max_distance_vector_y):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, len(edge), stride):
        if edge[i][0][0] == x and edge[i][0][1] == y:
            continue
        vector[0] = int(x - edge[i][0][0])
        vector[1] = int(y - edge[i][0][1])
        if ((vector[0] * max_distance_vector_x) + (
                vector[1] * max_distance_vector_y)) == 0:
            distances[i] = (math.sqrt(math.pow(vector[0], 2) + math.pow(vector[1], 2)))
