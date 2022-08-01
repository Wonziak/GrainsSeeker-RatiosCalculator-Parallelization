import math
from numba import cuda


@cuda.jit
def iterate_on_image_compare_color_cuda(image, color, layer):
    j, i = cuda.grid(2)
    m, n = image.shape[:2]

    if 0 <= i < n and 0 <= j < m:
        if image[j, i, 0] == color[2] and image[j, i, 1] == color[1] and image[j, i, 2] == color[0]:
            layer[j, i, 0] = 255


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
def get_sum_of_minimal_distance_from_each_point_to_edge(edge, domain, all_distances):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, len(domain), stride):
        for j in range(len(edge)):
            all_distances[i][j] = math.sqrt(
                math.pow(edge[j][0][0] - domain[i][0], 2) + math.pow(edge[j][0][1] - domain[i][1],
                                                                     2))


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
            cosa = abs(numerator / denominator)
            if cosa < min_cosine:
                min_cosine = cosa
                length = vector_length
        distances[i] = length


# OLD VERSION
# @cuda.jit
# def get_sum_of_minimal_distance_from_each_point_to_edge(edge, distances, x, y):
#     start = cuda.grid(1)
#     stride = cuda.gridsize(1)
#     for i in range(start, len(edge), stride):
#         if edge[i][0][0] == x and edge[i][0][1] == y:
#             continue
#         distances[i] = math.sqrt(
#             math.pow(edge[i][0][0] - x, 2) + math.pow(edge[i][0][1] - y, 2))


@cuda.jit
def color_black_borders_as_color_on_left(image, image_no_borders):
    j, i = cuda.grid(2)
    m, n = image.shape[:2]
    if 0 <= i < n and 0 <= j < m:
        if image[j, i, 0] == 0 and image[j, i, 1] == 0 and image[j, i, 2] == 0:
            image_no_borders[j, i, 0] = image[j - 1, i - 1, 0]
            image_no_borders[j, i, 1] = image[j - 1, i - 1, 1]
            image_no_borders[j, i, 2] = image[j - 1, i - 1, 2]
        else:
            image_no_borders[j, i, 0] = image[j, i, 0]
            image_no_borders[j, i, 1] = image[j, i, 1]
            image_no_borders[j, i, 2] = image[j, i, 2]


@cuda.jit
def iterate_on_image_and_assign_number(image, layer_of_numbers, color, number):
    j, i = cuda.grid(2)
    m, n = image.shape[:2]

    if 0 <= i < n and 0 <= j < m:
        if image[j, i, 0] == color[2] and image[j, i, 1] == color[1] and image[j, i, 2] == color[0]:
            layer_of_numbers[j, i] = number


@cuda.jit
def sum_neighbours_right(layer_of_numbers, layer_of_numbers_sum):
    j, i = cuda.grid(2)
    m, n = layer_of_numbers.shape[:2]

    if 0 <= i < n and 0 <= j < m - 1:
        layer_of_numbers_sum[j, i] = layer_of_numbers[j, i] + layer_of_numbers[j + 1, i]


@cuda.jit
def sum_neighbours_under(layer_of_numbers, layer_of_numbers_sum_under):
    j, i = cuda.grid(2)
    m, n = layer_of_numbers.shape[:2]

    if 0 <= i < n - 1 and 0 <= j < m:
        layer_of_numbers_sum_under[j, i] = layer_of_numbers[j, i] + layer_of_numbers[j, i + 1]


@cuda.jit
def angle_0(numbers, number, xs, ys, width, number_angle_zero_array, points_number):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    # for i in range(points_number):
    #     for j in range(width):
    #         number_angle_zero_array[i][j] = 0

    for i in range(start, points_number, stride):
        x = xs[i]
        y = ys[i]

        point_number = numbers[y, x]
        if point_number != number:
            continue
        for point_angle_0 in range(width - 1):
            point_to_check = x + point_angle_0 + 1
            if point_to_check >= width:
                point_to_check = point_to_check - width
            point_to_check_number = numbers[y, point_to_check]
            if point_number == point_to_check_number:
                number_angle_zero_array[i][point_angle_0 + 1] = 1 / points_number
            else:
                break


@cuda.jit
def angle_90(numbers, number, xs, ys, height, number_angle_90_array, points_number):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    # for i in range(points_number):
    #     for j in range(height):
    #         number_angle_90_array[i][j] = 0

    for i in range(start, points_number, stride):
        x = xs[i]
        y = ys[i]
        point_number = numbers[y, x]
        if point_number != number:
            continue
        for point_angle_90 in range(height - 1):
            point_to_check = y + point_angle_90 + 1
            if point_to_check >= height:
                point_to_check = point_to_check - height
            point_to_check_number = numbers[point_to_check, x]
            if point_number == point_to_check_number:
                number_angle_90_array[i][point_angle_90 + 1] = 1 / points_number
            else:
                break


@cuda.jit
def angle_45(numbers, number, xs, ys, width, height, number_angle_45_array, points_number):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    # for i in range(points_number):
    #     for j in range(height):
    #         number_angle_45_array[i][j] = 0

    for i in range(start, points_number, stride):
        x = xs[i]
        y = ys[i]
        point_number = numbers[y, x]
        if point_number != number:
            continue
        for point_angle_45 in range(height - 1):
            point_angle_45 = point_angle_45 + 1
            point_to_check_y = y - point_angle_45
            point_to_check_x = x + point_angle_45
            if point_to_check_x >= width:
                point_to_check_x = point_to_check_x - width
            if point_to_check_y < 0:
                point_to_check_y = point_to_check_y + height - 1

            point_to_check_number = numbers[point_to_check_y, point_to_check_x]
            if point_number == point_to_check_number:
                number_angle_45_array[i][point_angle_45] = 1 / points_number
            else:
                break
