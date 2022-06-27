from config.image_config import ImageConfig as ic
from .functions.statistic_ratios_gpu_functions import *
import itertools
import numpy as np
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

    def dispersion(self):
        start_time = time.time()
        area = self.imageArea * (math.pow(self.scale, 2))
        self.dispersionPhases = {k: (len(v) / area) * 100 for k, v in self.grains.items()}
        print("Dispersion time is: " + str(time.time() - start_time))
