from .functions.statistic_ratios_functions import *
import itertools
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict
import math

statsRatiosToCalculateList = ['BorderNeighbour',
                              'Dispersion',
                              'OnePointProbability',
                              'Linealpath']


class Statistics:
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

        numbers = map_pixels_to_colors(ic.image)

        layer_of_numbers_sum_under = np.zeros((ic.image.shape[0], ic.image.shape[1]))
        layer_of_numbers_sum_right = np.zeros((ic.image.shape[0], ic.image.shape[1]))
        summed_right, summed_under = sum_neighbours(numbers, layer_of_numbers_sum_under,
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

        print("Border length ratio sequentially time is: " + str(time.time() - start_time))

    def dispersion(self):
        start_time = time.time()
        area = self.imageArea * (math.pow(self.scale, 2))
        self.dispersionPhases = {k: (len(v) / area) * 100 for k, v in self.grains.items()}
        print("Dispersion time is: " + str(time.time() - start_time))

    def one_point_prob(self):
        start_time = time.time()
        colors_dict = {v: k for k, v in ic.colors_map.items()}
        for phase in ic.colors_map.keys():
            self.onePointProbability[phase] = 0
        for i in range(ic.height):
            for j in range(ic.width):
                color = (ic.image[i, j, 2], ic.image[i, j, 1], ic.image[i, j, 0])
                if color in colors_dict.keys():
                    phasename = colors_dict[color]
                    self.onePointProbability[phasename] += 1
        for key, value in self.onePointProbability.items():
            self.onePointProbability[key] = value / self.imageArea
        print("One point probability sequentially time is: " + str(time.time() - start_time))

    def lineal_path(self, points_number):
        start_time = time.time()
        for phase in ic.colors_map.keys():
            self.linealPath[phase] = {'angleZero': np.zeros((ic.width,), dtype=float),
                                      'angle90': np.zeros((ic.height,), dtype=float),
                                      'angle45': np.zeros((ic.height,), dtype=float)}
        colorsDict = {v: k for k, v in ic.colors_map.items()}
        rng = np.random.default_rng()
        xCoordinates = rng.choice(ic.width, points_number)
        yCoordinates = rng.choice(ic.height, points_number)
        for point in range(points_number):
            x = xCoordinates[point]
            y = yCoordinates[point]
            xyColor = (ic.image[y, x, 2], ic.image[y, x, 1], ic.image[y, x, 0])
            if xyColor not in colorsDict.keys():
                continue
            for pointAngleZero in range(ic.width - 1):
                pointAngleZero = pointAngleZero + 1
                pointToCheck = x + pointAngleZero
                if pointToCheck >= ic.width:
                    pointToCheck = pointToCheck - ic.width
                pointToCheckColor = (ic.image[y, pointToCheck, 2], ic.image[y, pointToCheck, 1],
                                     ic.image[y, pointToCheck, 0])
                if xyColor[0] == pointToCheckColor[0] and xyColor[1] == pointToCheckColor[1] and xyColor[2] == \
                        pointToCheckColor[2]:
                    self.linealPath[colorsDict[xyColor]]['angleZero'][pointAngleZero] += 1/points_number
                else:
                    break
            for pointAngle90 in range(ic.height - 1):
                pointAngle90 = pointAngle90 + 1
                pointToCheck = y + pointAngle90
                if pointToCheck >= ic.height:
                    pointToCheck = pointToCheck - ic.height
                pointToCheckColor = (ic.image[pointToCheck, x, 2], ic.image[pointToCheck, x, 1],
                                     ic.image[pointToCheck, x, 0])
                if xyColor[0] == pointToCheckColor[0] and xyColor[1] == pointToCheckColor[1] and xyColor[2] == \
                        pointToCheckColor[2]:
                    self.linealPath[colorsDict[xyColor]]['angle90'][pointAngle90] += 1/points_number
                else:
                    break
            for pointAngle45 in range(ic.height - 1):
                pointAngle45 = pointAngle45 + 1
                pointToCheckY = y - pointAngle45
                pointToCheckX = x + pointAngle45
                if pointToCheckX >= ic.width:
                    pointToCheckX = pointToCheckX - ic.width
                if pointToCheckY < 0:
                    pointToCheckY = pointToCheckY + ic.height - 1
                pointToCheckColor = (
                    ic.image[pointToCheckY, pointToCheckX, 2], ic.image[pointToCheckY, pointToCheckX, 1],
                    ic.image[pointToCheckY, pointToCheckX, 0])
                if xyColor[0] == pointToCheckColor[0] and xyColor[1] == pointToCheckColor[1] and xyColor[2] == \
                        pointToCheckColor[2]:
                    self.linealPath[colorsDict[xyColor]]['angle45'][pointAngle45] += 1/points_number
                else:
                    break
        for phase in ic.colors_map.keys():
            self.linealPath[phase]['angleZero'] = np.delete(self.linealPath[phase]['angleZero'], 0)
            self.linealPath[phase]['angle45'] = np.delete(self.linealPath[phase]['angle45'], 0)
            self.linealPath[phase]['angle90'] = np.delete(self.linealPath[phase]['angle90'], 0)
        print("lineal path sequentially time is: " + str(time.time() - start_time))
        angles = ['angleZero', 'angle45', 'angle90']
        x = range(1, ic.width)
        y = range(1, ic.height)
        for phase in ic.colors_map.keys():
            for angle in angles:
                if angle == 'angleZero':
                    plt.plot(x, self.linealPath[phase]['angleZero'])
                else:
                    plt.plot(y, self.linealPath[phase][angle])
                plt.xlabel('distance')
                plt.ylabel('probability')
                plt.title(phase + " " + angle + " sequentially")
                plt.show()
