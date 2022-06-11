import numpy as np

from config.image_config import ImageConfig
import cv2
import math
from numba import njit, cuda
from classes.ratios_class import RatiosClass


class GrainGPUClass(RatiosClass):
    def __init__(self, edge, phase):
        self.edge = edge
        self.phase = phase
        self.domain = []
        self.perimeter = len(edge)  # obwód - długość
        self.area = 0
        self.__get_area()
        self.centerOfMass = []
        self.centerOfMassLocal = []
        self.distanceFromCenterPowerSum = 0
        self.distanceFromCenter = 0
        self.distanceFromEdgeToCenter = 0
        self.distanceFromEdgeToCenterSquared = 0
        self.minDistanceFromEgdeSum = 0
        self.minDistaceCenterEdge = 0
        self.maxDistaceCenterEdge = 0
        self.maxDistancePoints = 0
        self.maxDistanceVectorCoords = []
        self.VectorPerpendicularLength = 0
        self.LH = -10
        self.LW = -10
        super().__init__()

    def start_calculating(self):
        self.find_com()
        self.__calculate_com_distances_height_width()
        # self.calculateRatios()

    def __calculate_com_distances_height_width(self):
        self.__calculate_height_width()
        self.__calculate_distances_sum_from_each_point_to_center()
        self.__calculate_distances_from_edge_to_center()
        # self.__calculate_max_distance_in_grain()
        # self.__find_min_dist_dum()
        # self.__find_vector_perpendicular()

    def __get_area(self):  # powierzchnia to domain(współrzędne), area to ilosc punktow
        domain = []
        width, height = self.__get_rectangle_containing_grain()
        for i in range(width[0], width[1]):
            for j in range(height[0], height[1]):
                if cv2.pointPolygonTest(self.edge, (i, j), measureDist=False) >= 0:
                    domain.append([i, j])
        self.domain = domain
        self.area = len(self.domain)

    def __get_rectangle_containing_grain(self):
        list_of_xs = []
        list_of_ys = []

        for edge_point in self.edge:
            list_of_xs.append(edge_point[0][0])
            list_of_ys.append(edge_point[0][1])
        max_x = max(list_of_xs)
        min_x = min(list_of_xs)

        max_y = max(list_of_ys)
        min_y = min(list_of_ys)

        width = (min_x, max_x)
        height = (min_y, max_y)

        return width, height

    def find_com(self, offsetX=0, offsetY=0):  # srodek ciezkosci
        allx = 0
        ally = 0
        for i in range(self.area):
            allx += self.domain[i][0]  # suma wspołrzędnych x pola
            ally += self.domain[i][1]  # suma wspołrzędnych y pola
        meanX = int(allx / self.area)
        meanY = int(ally / self.area)
        self.centerOfMass.append(meanX)
        self.centerOfMass.append(meanY)
        self.centerOfMassLocal.append(meanX - offsetX * ImageConfig.widthOffset)
        self.centerOfMassLocal.append(meanY - offsetY * ImageConfig.heightOffset)

    def __calculate_height_width(self):  # wysokosc i szerokosc
        list_of_xs = []
        list_of_ys = []

        for edge_point in self.edge:
            list_of_xs.append(edge_point[0][0])
            list_of_ys.append(edge_point[0][1])
        max_x = max(list_of_xs)
        min_x = min(list_of_xs)

        max_y = max(list_of_ys)
        min_y = min(list_of_ys)

        x_dist = max_x - min_x
        y_dist = max_y - min_y

        self.LW = x_dist
        self.LH = y_dist

    def __calculate_distances_sum_from_each_point_to_center(
            self):  # suma odleglosci od srodka ciezkosci, jeden to kazda odleglosc podniesiona do kwadratu
        empty_array = np.zeros(len(self.domain))
        x_gpu = cuda.to_device(np.array(self.domain))
        threads_per_block = 64
        blocks_per_grid = 96
        out_gpu = cuda.device_array_like(empty_array)
        calculate_distance_sum_from_center_gpu[blocks_per_grid, threads_per_block](x_gpu, out_gpu,
                                                                                   self.centerOfMass[
                                                                                       0],
                                                                                   self.centerOfMass[
                                                                                       1])
        cuda.synchronize()
        empty_array = out_gpu.copy_to_host()
        distance_sum_power = sum(empty_array)
        self.distanceFromCenterPowerSum = distance_sum_power

    def __calculate_distances_from_edge_to_center(self):
        """
        This function finds sum of distances from edge to center of mass. One distance is not rooted.
        Additionally finds max and min distances.
        """
        empty_array = np.zeros(len(self.edge))
        x_gpu = cuda.to_device(np.array(self.edge))
        threads_per_block = 64
        blocks_per_grid = 96
        out_gpu = cuda.device_array_like(empty_array)
        calculate_distance_from_center_to_edge_gpu[blocks_per_grid, threads_per_block](x_gpu,
                                                                                       out_gpu,
                                                                                       self.centerOfMass[
                                                                                           0],
                                                                                       self.centerOfMass[
                                                                                           1])
        cuda.synchronize()
        empty_array = out_gpu.copy_to_host()
        list_of_distances = list(map(math.sqrt, empty_array))
        self.distanceFromEdgeToCenter = sum(empty_array)
        self.distanceFromEdgeToCenterSquared = sum(list_of_distances)
        self.maxDistaceCenterEdge = max(list_of_distances)
        self.minDistaceCenterEdge = min(list_of_distances)

    def __calculate_max_distance_in_grain(self):  # najwięsza odleglość miedzy punktami ziarna
        maxdist = -1
        coordinates = [0, 0, 0, 0]
        for edgePoint1 in self.edge:
            for edgePoint2 in self.edge:
                if edgePoint1[0][0] == edgePoint2[0][0] and edgePoint1[0][1] == edgePoint2[0][1]:
                    continue
                x = (edgePoint2[0][0] - edgePoint1[0][0]) ** 2 + (
                        edgePoint2[0][1] - edgePoint1[0][1]) ** 2
                dist = math.sqrt(x)
                if dist > maxdist:
                    coordinates[0] = edgePoint1[0][0]  # x1
                    coordinates[1] = edgePoint1[0][1]  # y1
                    coordinates[2] = edgePoint2[0][0]  # x2
                    coordinates[3] = edgePoint2[0][1]  # y2
                    maxdist = dist

        self.maxDistancePoints = maxdist
        self.maxDistanceVectorCoords = [coordinates[2] - coordinates[0],
                                        coordinates[3] - coordinates[1]]


@cuda.jit
def calculate_distance_sum_from_center_gpu(domain, output, center_of_mass_0, center_of_mass_1):
    i = cuda.grid(1)
    output[i] = (center_of_mass_0 - domain[i][0]) ** 2 + (
            center_of_mass_1 - domain[i][1]) ** 2


@cuda.jit
def calculate_distance_from_center_to_edge_gpu(edge, output, center_of_mass_0, center_of_mass_1):
    i = cuda.grid(1)
    output[i] = (center_of_mass_0 - edge[i][0][0]) ** 2 + (
            center_of_mass_1 - edge[i][0][1]) ** 2
