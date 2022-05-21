from config.image_config import ImageConfig
import cv2
import math
from classes.ratios_class import RatiosClass


class GrainClass(RatiosClass):
    def __init__(self, edge):
        self.edge = edge
        self.domain = []
        self.perimeter = 0 # obwód - długość
        self.area = 0
        #self.__get_area()
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
        for i in range(ImageConfig.width):
            for j in range(ImageConfig.height):
                if cv2.pointPolygonTest(self.edge, (i, j), measureDist=False) >= 0:
                    domain.append([i, j])
        self.domain = domain
        self.area = len(self.domain)
