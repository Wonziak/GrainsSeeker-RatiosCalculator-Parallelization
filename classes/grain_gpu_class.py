import math

import cv2
import numpy as np
from numba import cuda

from classes.ratios_class import RatiosClass
from config.image_config import ImageConfig
from functions_for_cuda import calculate_distance_sum_from_center_gpu, \
    calculate_distance_from_center_to_edge_gpu, create_lists_of_xs_ys_edge_gpu, \
    create_lists_of_xs_ys_domain_gpu, get_sum_of_minimal_distance_from_each_point_to_edge


class GrainGPUClass(RatiosClass):
    def __init__(self, edge, phase):
        self.edge = edge
        self.phase = phase
        self.domain = []
        self.perimeter = len(edge)  # obwód - długość
        self.area = 0
        self.width_range = ()
        self.height_range = ()
        self.__get_rectangle_containing_grain()
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
        self.__calculate_distances_sum_from_each_point_to_center()
        self.__calculate_distances_from_edge_to_center()
        self.__calculate_max_distance_in_grain()
        self.__find_min_dist_dum()
        # self.__find_vector_perpendicular()

    def __get_area(self):  # powierzchnia to domain(współrzędne), area to ilosc punktow
        domain = []
        width = self.width_range
        height = self.height_range
        for i in range(width[0], width[1]):
            for j in range(height[0], height[1]):
                if cv2.pointPolygonTest(self.edge, (i, j), measureDist=False) >= 0:
                    domain.append([i, j])
        self.domain = domain
        self.area = len(self.domain)

    def __get_rectangle_containing_grain(self):
        x_gpu = cuda.to_device(self.edge)
        threads_per_block = 64
        blocks_per_grid = 96
        out_xs = cuda.device_array(shape=(len(self.edge)), dtype=np.int32)
        out_ys = cuda.device_array(shape=(len(self.edge)), dtype=np.int32)
        create_lists_of_xs_ys_edge_gpu[blocks_per_grid, threads_per_block](x_gpu, out_xs, out_ys)
        list_of_xs = out_xs.copy_to_host()
        list_of_ys = out_ys.copy_to_host()
        cuda.synchronize()
        max_x = int(max(list_of_xs))
        min_x = int(min(list_of_xs))

        max_y = int(max(list_of_ys))
        min_y = int(min(list_of_ys))

        width = (min_x, max_x)
        height = (min_y, max_y)
        self.width_range = width
        self.height_range = height
        x_dist = max_x - min_x
        y_dist = max_y - min_y

        self.LW = x_dist
        self.LH = y_dist

    def find_com(self, offsetX=0, offsetY=0):  # srodek ciezkosci
        empty_array = np.zeros(len(self.domain))
        x_gpu = cuda.to_device(np.array(self.domain))
        threads_per_block = 64
        blocks_per_grid = 96
        out_xs = cuda.device_array_like(empty_array)
        out_ys = cuda.device_array_like(empty_array)
        create_lists_of_xs_ys_domain_gpu[blocks_per_grid, threads_per_block](x_gpu, out_xs, out_ys)
        cuda.synchronize()
        meanX = out_xs.copy_to_host().mean()
        meanY = out_ys.copy_to_host().mean()

        self.centerOfMass.append(meanX)
        self.centerOfMass.append(meanY)
        self.centerOfMassLocal.append(meanX - offsetX * ImageConfig.widthOffset)
        self.centerOfMassLocal.append(meanY - offsetY * ImageConfig.heightOffset)

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
        convex_hull = cv2.convexHull(self.edge)
        for edgePoint1 in convex_hull:
            for edgePoint2 in convex_hull:
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

    def __find_min_dist_dum(self):  # suma minimalnych odleglosc od krawedzi
        distances = np.zeros(len(self.domain))
        x_gpu = cuda.to_device(np.array(self.edge))
        threads_per_block = 64
        blocks_per_grid = 96
        out_gpu = cuda.device_array_like(distances)
        for areaPoint in self.domain:
            get_sum_of_minimal_distance_from_each_point_to_edge[blocks_per_grid, threads_per_block] \
                (x_gpu, out_gpu, areaPoint[0], areaPoint[1])
            cuda.synchronize()
            self.minDistanceFromEgdeSum += min(out_gpu.copy_to_host())
