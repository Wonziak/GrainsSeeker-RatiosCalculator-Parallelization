from .functions.statistic_ratios_gpu_functions import *
import itertools
import numpy as np
from collections import defaultdict
import time
import matplotlib.pyplot as plt

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

    # def lineal_path(self):
    #     lineal_path = {}
    #     for phase in ic.colors_map.keys():
    #         lineal_path[phase] = {'angleZero': np.zeros((ic.width,), dtype=float),
    #                               'angle90': np.zeros((ic.height,), dtype=float),
    #                               'angle45': np.zeros((ic.height,), dtype=float)}
    #     colors_dict = {v: k for k, v in ic.colors_map.items()}
    #     rng = np.random.default_rng()
    #     x_coordinates = rng.choice(ic.width, 50)
    #     y_coordinates = rng.choice(ic.height, 50)
    #     points = list(zip(x_coordinates, y_coordinates))
    #     numbers = map_pixels_to_colors(ic.image)
    #     threads_per_block = 64
    #     blocks_per_grid = 96
    #     x_gpu = cuda.to_device(numbers)
    #     for point in points:
    #         color = (ic.image[point[1], point[0], 2], ic.image[point[1], point[0], 1], ic.image[point[1], point[0], 0])
    #         if color in colors_dict.keys():
    #             number = ic.color_number[colors_dict[color]]
    #             angle_zero_array = cuda.device_array_like(lineal_path[colors_dict[color]]['angleZero'])
    #             angle_zero[blocks_per_grid, threads_per_block](x_gpu, point[1], point[0], number, angle_zero_array)
    #             cuda.synchronize()
    #             angle_zero_arr = angle_zero_array.copy_to_host()
    #
    #             lineal_path[colors_dict[color]]['angleZero'] = angle_zero_arr
    #
    #             angle_90_array = cuda.device_array_like(lineal_path[colors_dict[color]]['angle90'])
    #             angle_90[blocks_per_grid, threads_per_block](x_gpu, point[0], point[1], number, angle_90_array)
    #             angle_90_array = angle_90_array.copy_to_host()
    #             cuda.synchronize()
    #             lineal_path[colors_dict[color]]['angle90'] = angle_90_array
    #
    #             angle_45_array = cuda.device_array_like(lineal_path[colors_dict[color]]['angle45'])
    #             angle_45[blocks_per_grid, threads_per_block](x_gpu, point[0], point[1], number, ic.height, ic.width,
    #                                                          angle_45_array)
    #             angle_45_array = angle_45_array.copy_to_host()
    #             cuda.synchronize()
    #             lineal_path[colors_dict[color]]['angle45'] = angle_45_array
    #
    #     print(self.linealPath)
    #     for phase in ic.colors_map.keys():
    #         print(np.unique(self.linealPath[phase]['angleZero']))
    #         print(np.unique(self.linealPath[phase]['angle45']))
    #         print(np.unique(self.linealPath[phase]['angle90']))
    #
    #     for phase in ic.colors_map.keys():
    #         lineal_path[phase]['angleZero'] = np.delete(lineal_path[phase]['angleZero'], 0)
    #         lineal_path[phase]['angle45'] = np.delete(lineal_path[phase]['angle45'], 0)
    #         lineal_path[phase]['angle90'] = np.delete(lineal_path[phase]['angle90'], 0)
    #
    #     angles = ['angleZero', 'angle45', 'angle90']
    #     x = range(1, ic.width)
    #     y = range(1, ic.height)
    #     for phase in ic.colors_map.keys():
    #         for angle in angles:
    #             if angle == 'angleZero':
    #                 plt.plot(x, lineal_path[phase]['angleZero'])
    #                 plt.xlabel('distance')
    #                 plt.ylabel('probability')
    #                 plt.title(phase + " " + angle)
    #                 plt.show()
    #             else:
    #                 plt.plot(y, lineal_path[phase][angle])
    #             plt.xlabel('distance')
    #             plt.ylabel('probability')
    #             plt.title(phase + " " + angle)
    #             plt.show()
    #     self.linealPath = lineal_path

    # @cuda.jit
    # def angle_zero(numbers, x, y, number, phase_angle_zero_array):
    #     point_angle_zero = cuda.grid(1)
    #     if 0 <= point_angle_zero < len(phase_angle_zero_array) - 1:
    #         next_point_angle_zero = point_angle_zero + 1
    #         point_to_check = x + next_point_angle_zero
    #         if point_to_check >= len(phase_angle_zero_array):
    #             point_to_check = point_to_check - len(phase_angle_zero_array)
    #         point_to_check_number = numbers[y, point_to_check]
    #         if number == point_to_check_number:
    #             phase_angle_zero_array[point_angle_zero] += 1/ len(phase_angle_zero_array)
    #
    #
    # @cuda.jit
    # def angle_90(numbers, x, y, number, phase_angle_90_array):
    #     start = cuda.grid(1)
    #     stride = cuda.gridsize(1)
    #     for point_angle_90 in range(start, len(phase_angle_90_array) - 1, stride):
    #         next_point_angle_90 = point_angle_90 + 1
    #         point_to_check = y + next_point_angle_90
    #         if point_to_check >= len(phase_angle_90_array):
    #             point_to_check = point_to_check - len(phase_angle_90_array)
    #         point_to_check_number = numbers[point_to_check, x]
    #         if number == point_to_check_number:
    #             phase_angle_90_array[next_point_angle_90] += 0.02
    #
    #
    # @cuda.jit
    # def angle_45(numbers, x, y, number, height, width, phase_angle_45_array):
    #     start = cuda.grid(1)
    #     stride = cuda.gridsize(1)
    #     for point_angle_45 in range(start, len(phase_angle_45_array) - 1, stride):
    #         next_point_angle_45 = point_angle_45 + 1
    #         point_to_check_y = y - next_point_angle_45
    #         point_to_check_x = x + next_point_angle_45

    #         if point_to_check_x >= width:
    #             point_to_check_x = point_to_check_x - width

    #         if point_to_check_y < 0:
    #             point_to_check_y = point_to_check_y + height - 1
    #         point_to_check_number = numbers[point_to_check_y, point_to_check_x]
    #         if number == point_to_check_number:
    #             phase_angle_45_array[next_point_angle_45] += 0.02

    def lineal_path(self):
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
        for phase, number in ic.color_number.items():
            angle_zero_array = cuda.device_array_like(lineal_path[phase]['angleZero'])
            angle_0[blocks_per_grid, threads_per_block](x_gpu, number, x_coordinates, y_coordinates,
                                                        ic.width, angle_zero_array)
            cuda.synchronize()
            angle_zero_arr = angle_zero_array.copy_to_host()
            print(angle_zero_arr)
            # lineal_path[phase]['angleZero'] = angle_zero_arr
            # plt.plot(x_coordinates, lineal_path[phase]['angleZero'])
            # plt.xlabel('distance')
            # plt.ylabel('probability')
            # plt.title(phase)
            # plt.show()


@cuda.jit
def angle_0(numbers, number, xs, ys, width, number_angle_zero_array):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, 50, stride):
        x = xs[i]
        y = ys[i]
        point_number = numbers[y, x]
        if point_number != number:
            continue
        for point_angle_0 in range(width - 1):
            point_to_check = x + point_angle_0 + 1
            if point_to_check >= width:
                point_to_check = point_to_check - width
            point_to_check_number = numbers[y, point_to_check]
            if point_number == point_to_check_number:
                number_angle_zero_array[point_angle_0+1] += 0.02
            else:
                break