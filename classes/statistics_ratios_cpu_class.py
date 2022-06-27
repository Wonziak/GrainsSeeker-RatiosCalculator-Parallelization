from config.image_config import ImageConfig as ic
from functions_for_cpu import color_black_borders_as_color_on_left_cpu, sum_neighbours_cpu, \
    assign_color_number_cpu
import itertools
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict
import cv2

statsRatiosToCalculateList = ['BorderNeighbour',
                              'Dispersion',
                              'OnePointProbability',
                              'Linealpath']


class StatisticsCPU:
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
        start_time = time.time()
        colors = list(ic.colors_map.keys())

        for pair in itertools.combinations(colors, 2):
            combination = pair[0] + "_" + pair[1]
            self.borderNeighboursCountRatio[combination] = ic.color_number[pair[0]] + \
                                                           ic.color_number[pair[1]]

        image_no_borders = ic.image
        image_no_borders = remove_borders(image_no_borders, 20)

        numbers = map_pixels_to_colors(image_no_borders)

        layer_of_numbers_sum_under = np.zeros((ic.image.shape[0], ic.image.shape[1]))
        layer_of_numbers_sum_right = np.zeros((ic.image.shape[0], ic.image.shape[1]))
        summed_right, summed_under = sum_neighbours_cpu(numbers, layer_of_numbers_sum_under,
                                                        layer_of_numbers_sum_right)
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
        print(border_pixels)
        self.borderNeighboursCountRatio = {k: v / all_border_pixels for k, v in
                                           border_pixels.items()}

        print("Border length ratio on CPU time is: " + str(time.time() - start_time))


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
