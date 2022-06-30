import cv2
from numba import njit, prange
import numpy as np
import math


@njit(parallel=True)
def iterate_on_image_compare_color_cpu(image, width, height, layer, color):
    for i in prange(width):
        for j in prange(height):
            if image[j, i, 0] == color[2] and image[j, i, 1] == color[1] and image[j, i, 2] == \
                    color[0]:
                layer[j, i, 0] = 255


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


@njit(parallel=True)
def color_black_borders_as_color_on_left_cpu(image, image_no_borders, width, height):
    for i in prange(height):
        for j in prange(width):
            if image[j, i, 0] == 0 and image[j, i, 1] == 0 and image[j, i, 2] == 0:
                image_no_borders[j, i, 0] = image[j - 1, i - 1, 0]
                image_no_borders[j, i, 1] = image[j - 1, i - 1, 1]
                image_no_borders[j, i, 2] = image[j - 1, i - 1, 2]
            else:
                image_no_borders[j, i, 0] = image[j, i, 0]
                image_no_borders[j, i, 1] = image[j, i, 1]
                image_no_borders[j, i, 2] = image[j, i, 2]
    return image_no_borders


@njit(parallel=True)
def assign_color_number_cpu(image, colors_as_numbers, color, number):
    for i in prange(image.shape[1]):
        for j in prange(image.shape[0]):
            if image[j, i, 0] == color[2] and image[j, i, 1] == color[1] and image[j, i, 2] == \
                    color[0]:
                colors_as_numbers[j, i] = number


@njit(parallel=True)
def sum_neighbours_cpu(layer_of_numbers, layer_of_numbers_sum_under, layer_of_numbers_sum_right):
    for i in prange(layer_of_numbers.shape[1] - 1):
        for j in prange(layer_of_numbers.shape[0]):
            layer_of_numbers_sum_under[j, i] = layer_of_numbers[j, i] + layer_of_numbers[j, i + 1]
    for i in prange(layer_of_numbers.shape[1]):
        for j in prange(layer_of_numbers.shape[0] - 1):
            layer_of_numbers_sum_right[j, i] = layer_of_numbers[j, i] + layer_of_numbers[j + 1, i]
    return layer_of_numbers_sum_right, layer_of_numbers_sum_under


@njit(parallel=True)
def angle_0(numbers, number, xs, ys, width, number_angle_zero_array):
    for i in prange(50):
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
                number_angle_zero_array[point_angle_0 + 1] += 0.02
            else:
                break
    return number_angle_zero_array


@njit(parallel=True)
def angle_90(numbers, number, xs, ys, height, number_angle_90_array):
    for i in prange(50):
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
                number_angle_90_array[point_angle_90 + 1] += 0.02
            else:
                break
    return number_angle_90_array


@njit(parallel=True)
def angle_45(numbers, number, xs, ys, width, height, number_angle_45_array):
    for i in prange(50):
        x = xs[i]
        y = ys[i]
        point_number = numbers[y, x]
        if point_number != number:
            continue
        for point_angle_45 in range(height - 1):
            point_to_check_y = y - point_angle_45 + 1
            point_to_check_x = x + point_angle_45 + 1
            if point_to_check_y < 0:
                point_to_check_y = point_to_check_y + height - 1

            if point_to_check_x >= width:
                point_to_check_x = point_to_check_x - width

            point_to_check_number = numbers[point_to_check_y, point_to_check_x]
            if point_number == point_to_check_number:
                number_angle_45_array[point_angle_45 + 1] += 0.02
            else:
                break
    return number_angle_45_array
