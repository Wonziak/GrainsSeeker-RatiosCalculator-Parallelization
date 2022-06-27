import math

import cv2
import numpy as np

from grain_classes.ratios_class import RatiosClass
from config.image_config import ImageConfig
from devices_functions.functions_for_cpu import create_lists_of_xs_ys_edge_cpu, create_lists_of_xs_ys_domain_cpu, \
    calculate_distance_sum_from_center_cpu, calculate_distance_from_center_to_edge_cpu, \
    get_sum_of_minimal_distance_from_each_point_to_edge_cpu, \
    get_all_perpendicular_vectors_length_cpu


class GrainCPUClass(RatiosClass):
    def __init__(self, edge, phase):
        self.edge = edge
        self.phase = phase
        self.domain = []
        self.perimeter = len(edge)  # obwód - długość
        self.area = 0
        self.width_range = ()
        self.height_range = ()
        self.LH = -10
        self.LW = -10
        self.centerOfMass = []
        self.centerOfMassLocal = []
        self.distanceFromCenterPowerSum = 0
        self.distanceFromEdgeToCenter = 0
        self.distanceFromEdgeToCenterSquared = 0
        self.minDistanceFromEdgeSum = 0
        self.minDistaceCenterEdge = 0
        self.maxDistaceCenterEdge = 0
        self.maxDistancePoints = 0
        self.maxDistanceVectorCoords = []
        self.VectorPerpendicularLength = 0
        self.__get_rectangle_containing_grain()
        self.__get_area()
        super().__init__()

    def start_calculating(self):
        self.find_com()
        self.__calculate_com_distances_height_width()
        self.calculateRatios()

    def __calculate_com_distances_height_width(self):
        self.__calculate_distances_sum_from_each_point_to_center()
        self.__calculate_distances_from_edge_to_center()
        self.__calculate_max_distance_in_grain()
        self.__find_min_dist_sum()
        self.__find_vector_perpendicular()

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

        list_of_xs, list_of_ys = create_lists_of_xs_ys_edge_cpu(self.edge)

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
        list_of_xs, list_of_ys = create_lists_of_xs_ys_domain_cpu(np.array(self.domain))
        meanX = int(list_of_xs.mean())
        meanY = int(list_of_ys.mean())

        self.centerOfMass.append(meanX)
        self.centerOfMass.append(meanY)
        self.centerOfMassLocal.append(meanX - offsetX * ImageConfig.widthOffset)
        self.centerOfMassLocal.append(meanY - offsetY * ImageConfig.heightOffset)

    def __calculate_distances_sum_from_each_point_to_center(
            self):  # suma odleglosci od srodka ciezkosci, jeden to kazda odleglosc podniesiona do kwadratu

        distances = calculate_distance_sum_from_center_cpu(np.array(self.domain),
                                                           self.centerOfMass[0],
                                                           self.centerOfMass[1])
        distance_sum_power = sum(distances)
        self.distanceFromCenterPowerSum = distance_sum_power

    def __calculate_distances_from_edge_to_center(self):
        """
        This function finds sum of distances from edge to center of mass. One distance is not rooted.
        Additionally finds max and min distances.
        """
        distances = calculate_distance_from_center_to_edge_cpu(self.edge, self.centerOfMass[0],
                                                               self.centerOfMass[1])
        list_of_distances = list(map(math.sqrt, distances))
        self.distanceFromEdgeToCenter = sum(list_of_distances)
        self.distanceFromEdgeToCenterSquared = sum(distances)
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
                x = math.pow((edgePoint2[0][0] - edgePoint1[0][0]), 2) + math.pow((
                        edgePoint2[0][1] - edgePoint1[0][1]), 2)
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

    def __find_min_dist_sum(self):  # suma minimalnych odleglosc od krawedzi
        sum_of_distances = \
            get_sum_of_minimal_distance_from_each_point_to_edge_cpu(np.array(self.domain),
                                                                    self.edge)
        self.minDistanceFromEdgeSum = sum_of_distances

    def __find_vector_perpendicular(self):
        self.VectorPerpendicularLength = \
            get_all_perpendicular_vectors_length_cpu(self.edge,
                                                     self.maxDistanceVectorCoords[0],
                                                     self.maxDistanceVectorCoords[1],
                                                     self.maxDistancePoints)
