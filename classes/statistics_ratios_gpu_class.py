from config.image_config import ImageConfig as ic
import itertools
from numba import cuda
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

statsRatiosToCalculateList = ['BorderNeighbour',
                              'Dispersion',
                              'OnePointProbability',
                              'Linealpath']


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

    def blr(self):
        colors = list(ic.colors_map.keys())

        for pair in itertools.combinations(colors, 2):
            combination = pair[0]+"_" + pair[1]
            self.borderNeighboursCountRatio[combination] = ic.color_number[pair[0]] + \
                                                           ic.color_number[pair[1]]

        threadsperblock = (16, 16)
        blockspergrid_x = math.ceil(ic.image.shape[0] / threadsperblock[0])
        blockspergrid_y = math.ceil(ic.image.shape[1] / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        image_no_borders = ic.image
        for i in range(14):
            x_gpu = cuda.to_device(image_no_borders)
            out_gpu = cuda.device_array_like(image_no_borders)
            color_black_borders_as_color_on_left[blockspergrid, threadsperblock](x_gpu, out_gpu)
            cuda.synchronize()
            image_no_borders = out_gpu.copy_to_host()

        # cv2.imshow('no borders', image_no_borders)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

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

        x_gpu = cuda.to_device(numbers)
        out_gpu = cuda.device_array_like(numbers)
        sum_neighbours_right[blockspergrid, threadsperblock](x_gpu, out_gpu)
        cuda.synchronize()
        summed_right = out_gpu.copy_to_host()

        x_gpu = cuda.to_device(numbers)
        out_gpu = cuda.device_array_like(numbers)
        sum_neighbours_under[blockspergrid, threadsperblock](x_gpu, out_gpu)
        cuda.synchronize()
        summed_under = out_gpu.copy_to_host()
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

        self.borderNeighboursCountRatio = {k: v/all_border_pixels for k, v in border_pixels.items()}
        print(self.borderNeighboursCountRatio)
        print(all_border_pixels)
        print(border_pixels)


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

    if 0 <= i < n and 0 <= j < m:
        layer_of_numbers_sum_under[j, i] = layer_of_numbers[j, i] + layer_of_numbers[j, i+1]
