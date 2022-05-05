import multiprocessing

import cv2
import time
from multiprocessing import Pool
from Config.ImageConfig import ImageConfig


def find_contours(phase_layers=dict):
    phase_contours = {}
    start_time = time.time()
    for phase, layer in phase_layers.items():
        contours, _ = cv2.findContours(layer, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        phase_contours[phase] = contours
    print("findContours sequentially time is: " + str(time.time() - start_time))
    # for phase, contours in phase_contours.items():
    #     print("{phase}: {contours_count} contours found".format(phase=phase, contours_count=len(contours)))


def find_contours_threading(phase_layers=dict):
    manager = multiprocessing.Manager()
    phase_contours = manager.dict()
    arguments = []
    for phase, layer in phase_layers.items():
        args = (phase_contours, phase, layer)
        arguments.append(args)
    start_time = time.time()
    with Pool(ImageConfig.colorsNumber) as pool:
        pool.starmap(parallel_find_contours, arguments)
    print("findContours multiprocessing time is: " + str(time.time() - start_time))
    # for phase, contours in phase_contours.items():
    #     print("{phase}: {contours_count} contours found".format(phase=phase, contours_count=len(contours)))


def parallel_find_contours(phase_contours: dict, phase, layer):
    contours, _ = cv2.findContours(layer, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    phase_contours[phase] = contours
    return phase_contours
