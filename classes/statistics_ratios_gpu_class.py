from config.image_config import ImageConfig as ic
from functions_for_cuda import color_black_borders_as_color_on_left, \
    iterate_on_image_and_assign_number, sum_neighbours_right, sum_neighbours_under
import itertools
from numba import cuda
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time

statsRatiosToCalculateList = ['BorderNeighbour',
                              'Dispersion',
                              'OnePointProbability',
                              'Linealpath']
threadsperblock = 0
blockspergrid_x = 0
blockspergrid_y = 0
blockspergrid = 0


class StatisticsGPU:
    def __init__(self, grains, scale=1):
        self.imageArea = ic.height * ic.width
        self.borderNeighboursCountRatio = {}
        self.dispersionPhases = {}
        self.onePointProbability = {}
        self.linealPath = {}
        self.calculatedRatios = {}
        self.grains = grains
        self.scale = scale
        set_block()

    def blr(self):

        start_time = time.time()
        colors = list(ic.colors_map.keys())

        for pair in itertools.combinations(colors, 2):
            combination = pair[0] + "_" + pair[1]
            self.borderNeighboursCountRatio[combination] = ic.color_number[pair[0]] + \
                                                           ic.color_number[pair[1]]

        image_no_borders = ic.image
        image_no_borders = remove_borders(image_no_borders, 14)

        numbers = map_pixels_to_colors(image_no_borders)

        summed_right, summed_under = sum_neighbours(numbers)
        right_uniques, right_occurrences = np.unique(summed_right, return_counts=True)
        under_uniques, under_occurrences = np.unique(summed_under, return_counts=True)

        right_borders = {}
        under_borders = {}

        for i in range(len(right_uniques)):
            right_borders[int(right_uniques[i])] = right_occurrences[i]

        for i in range(len(under_uniques)):
            under_borders[int(under_uniques[i])] = under_occurrences[i]

        all_border_pixels = 0
        border_pixels = defaultdict(int)
        for key, value in self.borderNeighboursCountRatio.items():
            if value in right_borders.keys():
                all_border_pixels += right_borders[value]
                border_pixels[key] += right_borders[value]
            if value in under_borders.keys():
                all_border_pixels += under_borders[value]
                border_pixels[key] += under_borders[value]
        self.borderNeighboursCountRatio = {k: v / all_border_pixels for k, v in
                                           border_pixels.items()}
        print(border_pixels)
        print("Border length ratio on GPU time is: " + str(time.time() - start_time))


def remove_borders(image_no_borders, iterations):
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(ic.image.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(ic.image.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

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
