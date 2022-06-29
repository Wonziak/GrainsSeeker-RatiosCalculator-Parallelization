from config.image_config import ImageConfig as ic
from devices_functions.functions_for_cuda import color_black_borders_as_color_on_left, \
    iterate_on_image_and_assign_number, sum_neighbours_right, sum_neighbours_under
from numba import cuda
import math
import numpy as np


def remove_borders(image_no_borders, iterations):
    for i in range(iterations):
        x_gpu = cuda.to_device(image_no_borders)
        out_gpu = cuda.device_array_like(image_no_borders)
        color_black_borders_as_color_on_left[blockspergrid, threadsperblock](x_gpu, out_gpu)
        cuda.synchronize()
        image_no_borders = out_gpu.copy_to_host()
    return image_no_borders


def map_pixels_to_colors(image_no_borders):
    colors_as_numbers = np.zeros((ic.image.shape[0], ic.image.shape[1]))
    x_gpu = cuda.to_device(image_no_borders)
    out_gpu = cuda.device_array_like(colors_as_numbers)
    for phase in ic.colors_map.keys():
        if phase in ic.background:
            continue
        color = ic.colors_map[phase]
        number = ic.color_number[phase]
        iterate_on_image_and_assign_number[blockspergrid, threadsperblock] \
            (x_gpu, out_gpu, color, number)
        cuda.synchronize()
    numbers = out_gpu.copy_to_host()
    return numbers


def sum_neighbours(numbers):
    x_gpu = cuda.to_device(numbers)
    out_gpu = cuda.device_array_like(numbers)
    sum_neighbours_under[blockspergrid, threadsperblock](x_gpu, out_gpu)
    cuda.synchronize()
    summed_under = out_gpu.copy_to_host()

    summed_right = right(numbers)
    return summed_right, summed_under


def right(numbers):
    x_gpu = cuda.to_device(numbers)
    out_gpu = cuda.device_array_like(numbers)
    sum_neighbours_right[blockspergrid, threadsperblock](x_gpu, out_gpu)
    cuda.synchronize()
    summed_right = out_gpu.copy_to_host()
    return summed_right


def set_block():
    global threadsperblock
    global blockspergrid_x
    global blockspergrid_y
    global blockspergrid
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(ic.image.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(ic.image.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
