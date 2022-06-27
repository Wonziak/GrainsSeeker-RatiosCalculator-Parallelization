import math
import config.ratios_config as rc


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
        self.Danielsson = (math.pow(self.area, 3)) / self.minDistanceFromEdgeSum

    def __haralick(self):
        self.Haralick = math.sqrt(
            (math.pow(self.distanceFromEdgeToCenter, 2)) / (
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
        self.RCom = math.pow(self.perimeter, 2) / self.area

    def __lp1(self):
        self.Lp1 = self.minDistaceCenterEdge / self.maxDistaceCenterEdge

    def __lp2(self):
        self.Lp2 = self.maxDistancePoints / self.perimeter

    def __lp3(self):
        self.Lp3 = self.maxDistancePoints / self.VectorPerpendicularLength

    def calculateRatios(self):
        rc.to_lower_case()
        self.calculatedRatiosDict["Phase"] = self.phase
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
