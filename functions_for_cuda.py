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
def get_all_perpendicular_vectors_length(edge, vector, distances, max_distance_vector_x,
                                         max_distance_vector_y, max_vector_length):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, len(edge), stride):
        min_cosine = 1
        length = 0
        for j in range(len(edge)):
            if edge[j][0][0] == edge[i][0][0] and edge[j][0][1] == edge[i][0][1]:
                continue
            vector[0] = int(edge[i][0][0] - edge[j][0][0])
            vector[1] = int(edge[i][0][1] - edge[j][0][1])
            vector_length = math.sqrt(math.pow(vector[0], 2) + math.pow(vector[1], 2))

            numerator = ((vector[0] * max_distance_vector_x) + (vector[1] * max_distance_vector_y))
            denominator = vector_length * max_vector_length
            cosa = abs(numerator/denominator)
            if cosa < min_cosine:
                min_cosine = cosa
                length = vector_length
        distances[i] = length

            # for i in range(start, len(edge), stride):
            #     if edge[i][0][0] == x and edge[i][0][1] == y:
            #         continue
            #     vector[0] = int(x - edge[i][0][0])
            #     vector[1] = int(y - edge[i][0][1])
            #     scalar_product = abs((vector[0] * max_distance_vector_x) + (
            #             vector[1] * max_distance_vector_y))
            #     if scalar_product < min_scalar:
            #         min_scalar = scalar_product
            #         vector_length = math.sqrt(math.pow(vector[0], 2) + math.pow(vector[1], 2))
            #     distances[i] = vector_length
