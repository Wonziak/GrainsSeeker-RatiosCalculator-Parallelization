from config.image_config import ImageConfig
import cv2
import math
from classes.ratios_class import RatiosClass


class GrainClass(RatiosClass):
    def __init__(self, edge):
        self.edge = edge
        self.domain = []
        self.perimeter = 0  # obwód - długość
        self.area = 0
        # self.__get_area()
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
        self.LH = 0
        self.LW = 0

        # self.domain = []
        # self.perimeter = len(edge)  # obwód - długość
        # self.area = 0
        # self.__getArea()
        # self.centerOfMass = []
        # self.centerOfMassLocal = []
        # self.distanceFromCenterPowerSum = 0
        # self.distanceFromCenter = 0
        # self.distanceFromEdgeToCenter = 0
        # self.distanceFromEdgeToCenterSquared = 0
        # self.minDistanceFromEgdeSum = 0
        # self.minDistaceCenterEdge = 0
        # self.maxDistaceCenterEdge = 0
        # self.maxDistancePoints = 0
        # self.maxDistanceVectorCoords = []
        # self.VectorPerpendicularLength = 0
        # self.LH = 0
        # self.LW = 0
        # super().__init__()

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
        most_left = tuple(self.edge[self.edge[:, :, 0].argmin()][0])[0]
        most_right = tuple(self.edge[self.edge[:, :, 0].argmax()][0])[0]
        width = (most_left, most_right)

        most_up = tuple(self.edge[self.edge[:, :, 1].argmin()][0])[1]
        most_down = tuple(self.edge[self.edge[:, :, 1].argmax()][0])[1]
        height = (most_up, most_down)

        return width, height
