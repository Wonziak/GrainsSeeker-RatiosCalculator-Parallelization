from .functions.statistic_ratios_gpu_functions import *
import itertools
import numpy as np
from collections import defaultdict
import time
import matplotlib.pyplot as plt
from devices_functions.functions_for_cuda import angle_0, angle_90, angle_45

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

    def one_point_prob(self):
        start_time = time.time()
        numbers = map_pixels_to_colors(ic.image)
        uniques, occurrences = np.unique(numbers, return_counts=True)

        number_color_dict = {v: k for k, v in ic.color_number.items()}
        for key, value in number_color_dict.items():
            for unique in uniques:
                if unique == key:
                    index = np.where(uniques == unique)[0][0]
                    self.onePointProbability[number_color_dict[unique]] = \
                        occurrences[index] / self.imageArea
        print("One point probability on GPU time is: " + str(time.time() - start_time))

    def lineal_path(self):
        start_time = time.time()
        lineal_path = {}
        for phase in ic.colors_map.keys():
            lineal_path[phase] = {'angleZero': np.zeros(ic.width, dtype=float),
                                  'angle90': np.zeros(ic.height, dtype=float),
                                  'angle45': np.zeros(ic.height, dtype=float)}
        rng = np.random.default_rng()
        x_coordinates = rng.choice(ic.width, 50)
        y_coordinates = rng.choice(ic.height, 50)

        x_coordinates = np.array(x_coordinates)
        y_coordinates = np.array(y_coordinates)

        numbers = map_pixels_to_colors(ic.image)

        x_gpu = cuda.to_device(numbers)
        threads_per_block = 64
        blocks_per_grid = 96

        angle_zero_array = cuda.device_array_like(np.zeros((50, ic.width)))
        angle_90_array = cuda.device_array_like(np.zeros((50, ic.height)))
        angle_45_array = cuda.device_array_like(np.zeros((50, ic.height)))

        for phase, number in ic.color_number.items():

            arr0 = np.zeros(ic.width, dtype=float)
            angle_0[blocks_per_grid, threads_per_block](x_gpu, number, cuda.to_device(x_coordinates),
                                                        cuda.to_device(y_coordinates),
                                                        ic.width, angle_zero_array)
            angle_zero_arr = angle_zero_array.copy_to_host()
            cuda.synchronize()

            arr90 = np.zeros(ic.height, dtype=float)
            angle_90[blocks_per_grid, threads_per_block](x_gpu, number, cuda.to_device(x_coordinates),
                                                         cuda.to_device(y_coordinates),
                                                         ic.height, angle_90_array)
            angle_90_arr = angle_90_array.copy_to_host()
            cuda.synchronize()

            arr45 = np.zeros(ic.height, dtype=float)
            angle_45[blocks_per_grid, threads_per_block](x_gpu, number, cuda.to_device(x_coordinates),
                                                         cuda.to_device(y_coordinates), ic.width,
                                                         ic.height, angle_45_array)
            angle_45_arr = angle_45_array.copy_to_host()
            cuda.synchronize()

            for i in range(50):
                arr0 = np.add(arr0, angle_zero_arr[i])
                arr90 = np.add(arr90, angle_90_arr[i])
                arr45 = np.add(arr45, angle_45_arr[i])

            lineal_path[phase]['angleZero'] = arr0
            lineal_path[phase]['angle90'] = arr90
            lineal_path[phase]['angle45'] = arr45

        for phase in ic.colors_map.keys():
            lineal_path[phase]['angleZero'] = np.delete(lineal_path[phase]['angleZero'], 0)
            lineal_path[phase]['angle45'] = np.delete(lineal_path[phase]['angle45'], 0)
            lineal_path[phase]['angle90'] = np.delete(lineal_path[phase]['angle90'], 0)
        print("lineal path on GPU time is: " + str(time.time() - start_time))
        angles = ['angleZero', 'angle45', 'angle90']
        x = range(1, ic.width)
        y = range(1, ic.height)
        for phase in ic.colors_map.keys():
            for angle in angles:
                if angle == 'angleZero':
                    plt.plot(x, lineal_path[phase]['angleZero'])
                    plt.xlabel('distance')
                    plt.ylabel('probability')
                    plt.title(phase + " " + angle)
                    plt.show()
                else:
                    plt.plot(y, lineal_path[phase][angle])
                    plt.xlabel('distance')
                    plt.ylabel('probability sequentially')
                    plt.title(phase + " " + angle)
                    plt.show()
        self.linealPath = lineal_path
