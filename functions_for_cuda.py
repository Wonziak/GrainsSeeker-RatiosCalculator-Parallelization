import math

from numba import cuda


@cuda.jit
def calculate_distance_sum_from_center_gpu(domain, output, center_of_mass_0, center_of_mass_1):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, len(domain), stride):
        output[i] = (center_of_mass_0 - domain[i][0]) ** 2 + (
                center_of_mass_1 - domain[i][1]) ** 2


@cuda.jit
def calculate_distance_from_center_to_edge_gpu(edge, output, center_of_mass_0, center_of_mass_1):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, len(edge), stride):
        output[i] = (center_of_mass_0 - edge[i][0][0]) ** 2 + (
                center_of_mass_1 - edge[i][0][1]) ** 2


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
def get_sum_of_minimal_distance_from_each_point_to_edge(edge, distances, domain_x, domain_y):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, len(edge), stride):
        if edge[i][0][0] == domain_x and edge[i][0][1] == domain_y:
            continue
        distances[i] = math.sqrt(
            math.pow(edge[i][0][0] - domain_x, 2) + math.pow(edge[i][0][1] - domain_y, 2))
