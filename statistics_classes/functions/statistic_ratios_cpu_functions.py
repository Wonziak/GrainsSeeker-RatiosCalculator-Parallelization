from config.image_config import ImageConfig as ic
from devices_functions.functions_for_cpu import color_black_borders_as_color_on_left_cpu, \
    assign_color_number_cpu
import numpy as np


def remove_borders(image_no_borders, iterations):
    for i in range(iterations):
        image_no_borders = color_black_borders_as_color_on_left_cpu(ic.image,
                                                                    image_no_borders,
                                                                    ic.image.shape[0],
                                                                    ic.image.shape[1])
    return image_no_borders


def map_pixels_to_colors(image_no_borders):
    colors_as_numbers = np.zeros((ic.image.shape[0], ic.image.shape[1]))
    for phase in ic.colors_map.keys():
        if phase in ic.background:
            continue
        color = ic.colors_map[phase]
        number = ic.color_number[phase]
        assign_color_number_cpu(image_no_borders, colors_as_numbers, color, number)
    return colors_as_numbers
