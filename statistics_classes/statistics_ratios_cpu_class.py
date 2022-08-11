from devices_functions.functions_for_cpu import sum_neighbours_cpu, angle_0, angle_90, angle_45
from .functions.statistic_ratios_cpu_functions import *
import itertools
import numpy as np
import time
from collections import defaultdict
import math
# import matplotlib.pyplot as plt

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
        self.numbers = map_pixels_to_colors(ic.image)

    def blr(self):
        start_time = time.time()
        colors = list(ic.colors_map.keys())

        for pair in itertools.combinations(colors, 2):
            combination = pair[0] + "_" + pair[1]
            self.borderNeighboursCountRatio[combination] = ic.color_number[pair[0]] + \
                                                           ic.color_number[pair[1]]

        numbers = map_pixels_to_colors(ic.image)

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
        self.borderNeighboursCountRatio = {k: v / all_border_pixels for k, v in
                                           border_pixels.items()}

        print("Border length ratio on CPU time is: " + str(time.time() - start_time))

    def dispersion(self):
        start_time = time.time()
        area = self.imageArea * (math.pow(self.scale, 2))
        self.dispersionPhases = {k: (len(v) / area) * 100 for k, v in self.grains.items()}
        print("Dispersion time is: " + str(time.time() - start_time))

    def one_point_prob(self):
        start_time = time.time()
        uniques, occurrences = np.unique(self.numbers, return_counts=True)

        number_color_dict = {v: k for k, v in ic.color_number.items()}
        for key, value in number_color_dict.items():
            for unique in uniques:
                if unique == key:
                    index = np.where(uniques == unique)[0][0]
                    self.onePointProbability[number_color_dict[unique]] = \
                        occurrences[index] / self.imageArea
        print("One point probability on CPU time is: " + str(time.time() - start_time))

    def lineal_path(self, points_number):
        start_time = time.time()
        lineal_path = {}
        for phase in ic.colors_map.keys():
            lineal_path[phase] = {'angleZero': np.zeros((ic.width,), dtype=float),
                                  'angle90': np.zeros((ic.height,), dtype=float),
                                  'angle45': np.zeros((ic.height,), dtype=float)}

        rng = np.random.default_rng()
        x_coordinates = rng.choice(ic.width, points_number)
        y_coordinates = rng.choice(ic.height, points_number)

        x_coordinates = np.array(x_coordinates)
        y_coordinates = np.array(y_coordinates)
        for phase, number in ic.color_number.items():
            lineal_path[phase]['angleZero'] = angle_0(self.numbers, number, x_coordinates, y_coordinates, ic.width,
                                                      lineal_path[phase]['angleZero'], points_number)
            lineal_path[phase]['angle90'] = angle_90(self.numbers, number, x_coordinates, y_coordinates, ic.height,
                                                     lineal_path[phase]['angle90'], points_number)

            lineal_path[phase]['angle45'] = angle_45(self.numbers, number, x_coordinates, y_coordinates, ic.width, ic.height,
                                                     lineal_path[phase]['angle45'], points_number)

        for phase in ic.colors_map.keys():
            lineal_path[phase]['angleZero'] = np.delete(lineal_path[phase]['angleZero'], 0)
            lineal_path[phase]['angle45'] = np.delete(lineal_path[phase]['angle45'], 0)
            lineal_path[phase]['angle90'] = np.delete(lineal_path[phase]['angle90'], 0)
        print("lineal path on CPU time is: " + str(time.time() - start_time))
        # angles = ['angleZero', 'angle45', 'angle90']
        # x = range(1, ic.width)
        # y = range(1, ic.height)
        # for phase in ic.colors_map.keys():
        #     for angle in angles:
        #         if angle == 'angleZero':
        #             plt.plot(x, lineal_path[phase]['angleZero'])
        #         else:
        #             plt.plot(y, lineal_path[phase][angle])
        #         plt.xlabel('distance')
        #         plt.ylabel('probability')
        #         plt.title(phase + " " + angle + " CPU")
        #         plt.show()
        self.linealPath = lineal_path
