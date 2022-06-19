from config.image_config import ImageConfig
import cv2
import math
from classes.ratios_class import RatiosClass
import numpy as np
from numba import njit, cuda


class GrainClass(RatiosClass):
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
        self.minDistanceFromEdgeSum = 0
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
        self.__calculate_distances_sum_from_center()
        self.__calculate_distances_from_edge_to_center()
        self.__calculate_max_min_from_center()
        # self.__calculate_max_distance_in_grain()
        self.__calculate_max_distance_in_grain_convexhull()
        self.__find_min_dist_sum()
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
        list_of_xs = []
        list_of_ys = []

        for edge_point in self.edge:
            list_of_xs.append(edge_point[0][0])
            list_of_ys.append(edge_point[0][1])
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

    def __calculate_distances_sum_from_center(
            self):  # suma odleglosci od srodka ciezkosci, jeden to kazda odleglosc podniesiona do kwadratu
        distanceSumPower = 0
        for p in self.domain:
            distpower = math.pow((self.centerOfMass[0] - p[0]), 2) + math.pow(
                (self.centerOfMass[1] - p[1]), 2)
            distanceSumPower += distpower
        self.distanceFromCenterPowerSum = distanceSumPower

    def __calculate_distances_from_edge_to_center(self):
        distanceSumPower = 0
        distanceSum = 0
        for p in self.edge:
            distpower = math.pow((self.centerOfMass[0] - p[0][0]), 2) + math.pow((
                    self.centerOfMass[1] - p[0][1]), 2)
            dist = math.sqrt(distpower)
            distanceSumPower += distpower
            distanceSum += dist
        self.distanceFromEdgeToCenter = distanceSum
        self.distanceFromEdgeToCenterSquared = distanceSumPower

    def __find_min_dist_sum(self):  # suma minimalnych odleglosc od krawedzi
        mindist = float('inf')
        for areaPoint in self.domain:
            for edgePoint in self.edge:
                if areaPoint[0] == edgePoint[0][0] and areaPoint[1] == edgePoint[0][1]:
                    continue
                x = math.pow((edgePoint[0][0] - areaPoint[0]), 2) + math.pow(
                    (edgePoint[0][1] - areaPoint[1]), 2)
                dist = math.sqrt(x)
                if dist < mindist:
                    mindist = dist
            self.minDistanceFromEdgeSum += mindist
            mindist = float('inf')

    def __calculate_max_min_from_center(
            self):  # najwieszka i najmniejsza odleglosc miedzy srodkiem i krawedzia
        maxdist = -1
        mindist = float('inf')
        for edgePoint in self.edge:
            x = math.pow((self.centerOfMass[0] - edgePoint[0][0]), 2) + math.pow((
                    self.centerOfMass[1] - edgePoint[0][1]), 2)
            dist = math.sqrt(x)
            if dist > maxdist:
                maxdist = dist
            if dist < mindist:
                mindist = dist
        self.maxDistaceCenterEdge = maxdist
        self.minDistaceCenterEdge = mindist

    def __calculate_max_distance_in_grain(self):  # najwięsza odleglość miedzy punktami ziarna
        maxdist = -1
        coordinates = [0, 0, 0, 0]
        for edgePoint1 in self.edge:
            for edgePoint2 in self.edge:
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

    def __calculate_max_distance_in_grain_convexhull(
            self):  # najwięsza odleglość miedzy punktami ziarna
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

    def __find_vector_perpendicular(self):
        dst = []
        for edgePoint1 in self.edge:
            for edgePoint2 in self.edge:
                if edgePoint1[0][0] == edgePoint2[0][0] and edgePoint1[0][1] == edgePoint2[0][1]:
                    continue
                vec = [edgePoint2[0][0] - edgePoint1[0][0], edgePoint2[0][1] - edgePoint1[0][1]]
                if ((vec[0] * self.maxDistanceVectorCoords[0]) + (
                        vec[0] * self.maxDistanceVectorCoords[1])) == 0:
                    dst.append(math.sqrt(vec[0] ** 2 + vec[1] ** 2))
        self.VectorPerpendicularLength = max(dst)
