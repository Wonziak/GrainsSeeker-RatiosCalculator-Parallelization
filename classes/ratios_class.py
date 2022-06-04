import math
import config.ratios_config as rc
import numpy as np


class RatiosClass:
    def __init__(self):
        self.Malinowska = 0
        self.Mz = 0
        self.Blair_Bliss = 0
        self.Danielsson = 0
        self.Haralick = 0
        self.RLS = 0
        self.RF = 0
        self.RC1 = 0
        self.RC2 = 0
        self.RCom = 0
        self.Lp1 = 0
        self.Lp2 = 0
        self.Lp3 = 0
        self.MeanCurvature = 0
        self.calculatedRatiosDict = {}

    def __malinowska(self):
        self.Malinowska = self.perimeter / (2 * math.sqrt(math.pi * self.area)) - 1

    def __blair_bliss(self):
        self.Blair_Bliss = self.area / math.sqrt(2 * math.pi * self.distanceFromCenterPowerSum)

    def __danielsson(self):
        self.Danielsson = (self.area ** 3) / self.minDistanceFromEgdeSum

    def __haralick(self):
        self.Haralick = math.sqrt(
            (self.distanceFromEdgeToCenter ** 2) / (
                    abs(self.area * self.distanceFromEdgeToCenterSquared - 1)))

    def __mz(self):
        self.Mz = (2 * math.sqrt(math.pi * self.area)) / self.perimeter

    def __rls(self):
        self.RLS = self.perimeter / self.area

    def __rf(self):
        self.RF = self.LH / self.LW

    def __rc1(self):
        self.RC1 = math.sqrt(4 * self.area / math.pi)

    def __rc2(self):
        self.RC2 = self.perimeter / math.pi

    def __rcom(self):
        self.RCom = self.perimeter ** 2 / self.area

    def __lp1(self):
        self.Lp1 = self.minDistaceCenterEdge / self.maxDistaceCenterEdge

    def __lp2(self):
        self.Lp2 = self.maxDistancePoints / self.perimeter

    def __lp3(self):
        self.Lp3 = self.maxDistancePoints / self.VectorPerpendicularLength

    def __meanCurvature(self):
        edge = []
        for i in range(len(self.edge)):
            edge.append(Point(self.edge[i][0][0], self.edge[i][0][1]))
        edge.insert(0, edge[len(edge) - 1])
        edge.append(edge[1])
        i = 1
        while i < len(edge) - 1:
            x1 = edge[i - 1].x
            x2 = edge[i].x
            x3 = edge[i + 1].x
            if (x1 == x2 and x2 != x3) or (x2 == x3 and x1 != x2):
                del edge[i]
                i = i - 1
                continue
            i += 1
        del edge[0]
        del edge[len(edge) - 1]

        curvature = []
        ptCount = 5
        ptCount2 = int(ptCount / 2)
        ypts = np.zeros(ptCount, np.uint8)
        xpts = np.zeros(ptCount, np.uint8)

        for i in range(len(edge)):
            k = 0
            for j in range(i - ptCount2, i + ptCount2 + 1):
                xpts[k] = edge[j % len(edge)].x
                ypts[k] = edge[j % len(edge)].y
                k += 1
            sortedX = np.sort(xpts)
            if sortedX[0] == sortedX[len(sortedX) - 1]:
                for j in range(ptCount):
                    ypts[j] = xpts[j] * math.sin(math.pi / 2) + ypts[j] * math.cos(math.pi / 2)
            first = self.__calculateDerviative(ypts)
            second = self.__calculateDerviative(ypts, derivative=2)
            if first == 0:
                continue
            c = abs(second / math.pow(1 + first * first, 3.0 / 2.0))
            curvature.append(c)
        mean = np.mean(curvature)
        self.MeanCurvature = mean

    def __calculateDerviative(self, ys, derivative=1):
        if derivative == 1:
            result = (-ys[4] + 8 * ys[3] - 8 * ys[1] + ys[0]) / 12 + (1 / 30) * ys[2]
            return result
        if derivative == 2:
            result = ys[3] - 2 * ys[2] + ys[1]
            return result

    def calculateRatios(self):
        rc.to_lower_case()
        if 'malinowska' in rc.ratiosToCalculateList:
            self.__malinowska()
            self.calculatedRatiosDict['malinowska'] = self.Malinowska
        if 'blair bliss' in rc.ratiosToCalculateList:
            self.__blair_bliss()
            self.calculatedRatiosDict['blair bliss'] = self.Blair_Bliss
        if 'danielsson' in rc.ratiosToCalculateList:
            self.__danielsson()
            self.calculatedRatiosDict['danielsson'] = self.Danielsson
        if 'haralick' in rc.ratiosToCalculateList:
            self.__haralick()
            self.calculatedRatiosDict['haralick'] = self.Haralick
        if 'mz' in rc.ratiosToCalculateList:
            self.__mz()
            self.calculatedRatiosDict['mz'] = self.Mz
        if 'rls' in rc.ratiosToCalculateList:
            self.__rls()
            self.calculatedRatiosDict['rls'] = self.RLS
        if 'rf' in rc.ratiosToCalculateList:
            self.__rf()
            self.calculatedRatiosDict['rf'] = self.RF
        if 'rc1' in rc.ratiosToCalculateList:
            self.__rc1()
            self.calculatedRatiosDict['rc1'] = self.RC1
        if 'rc2' in rc.ratiosToCalculateList:
            self.__rc2()
            self.calculatedRatiosDict['rc2'] = self.RC2
        if 'rcom' in rc.ratiosToCalculateList:
            self.__rcom()
            self.calculatedRatiosDict['rcom'] = self.RCom
        if 'lp1' in rc.ratiosToCalculateList:
            self.__lp1()
            self.calculatedRatiosDict['lp1'] = self.Lp1
        if 'lp2' in rc.ratiosToCalculateList:
            self.__lp2()
            self.calculatedRatiosDict['lp2'] = self.Lp2
        if 'lp3' in rc.ratiosToCalculateList:
            self.__lp3()
            self.calculatedRatiosDict['lp3'] = self.Lp3
        if 'curvature' in rc.ratiosToCalculateList:
            self.__meanCurvature()
            self.calculatedRatiosDict['mean_curvature'] = self.MeanCurvature


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
