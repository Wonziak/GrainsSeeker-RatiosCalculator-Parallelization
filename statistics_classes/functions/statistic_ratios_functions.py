from config.image_config import ImageConfig as ic
import numpy as np


def remove_borders(image_no_borders, iterations):
    for i in range(iterations):
        image_no_borders = color_black_borders_as_color_on_left(ic.image, image_no_borders)
    return image_no_borders


def color_black_borders_as_color_on_left(image, image_no_borders):
    for i in range(ic.width):
        for j in range(ic.height):
            if image[j, i, 0] == 0 and image[j, i, 1] == 0 and image[j, i, 2] == 0:
                image_no_borders[j, i, 0] = image[j - 1, i - 1, 0]
                image_no_borders[j, i, 1] = image[j - 1, i - 1, 1]
                image_no_borders[j, i, 2] = image[j - 1, i - 1, 2]
            else:
                image_no_borders[j, i, 0] = image[j, i, 0]
                image_no_borders[j, i, 1] = image[j, i, 1]
                image_no_borders[j, i, 2] = image[j, i, 2]
    return image_no_borders


def map_pixels_to_colors(image_no_borders):
    colors_as_numbers = np.zeros((ic.image.shape[0], ic.image.shape[1]))
    for phase in ic.colors_map.keys():
        if phase in ic.background:
            continue
        color = ic.colors_map[phase]
        number = ic.color_number[phase]
        assign_color_number(image_no_borders, colors_as_numbers, color, number)
    return colors_as_numbers


def assign_color_number(image, colors_as_numbers, color, number):
    for i in range(ic.width):
        for j in range(ic.height):
            if image[j, i, 0] == color[2] and image[j, i, 1] == color[1] and image[j, i, 2] == \
                    color[0]:
                colors_as_numbers[j, i] = number


def sum_neighbours(layer_of_numbers, layer_of_numbers_sum_under, layer_of_numbers_sum_right):
    for i in range(ic.width - 1):
        for j in range(ic.height):
            layer_of_numbers_sum_under[j, i] = layer_of_numbers[j, i] + layer_of_numbers[j, i + 1]
    for i in range(ic.width):
        for j in range(ic.height - 1):
            layer_of_numbers_sum_right[j, i] = layer_of_numbers[j, i] + layer_of_numbers[j + 1, i]
    return layer_of_numbers_sum_right, layer_of_numbers_sum_under
