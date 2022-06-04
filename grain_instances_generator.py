from classes.grain_class import GrainClass
from collections import defaultdict
import time
from multiprocessing import Pool, Manager
from config.image_config import ImageConfig


def generate_grains_instances_sequentially(contours: dict):
    phase_grains_dict = defaultdict(list)
    start_time = time.time()
    for phase in contours:
        for grain_contours in contours[phase]:
            grain = GrainClass(grain_contours, phase)
            if grain.area <= grain.perimeter:
                del grain
                continue
            grain.start_calculating()
            phase_grains_dict[phase].append(grain)
    print("Grains instances generator sequentially time is: " + str(time.time() - start_time))
    for phase in phase_grains_dict:
        print(phase)
        print(len(phase_grains_dict[phase]))


def generate_grains_instances_threading(contours: dict):
    manager = Manager()
    phase_grains_dict = {}
    arguments = []
    for phase, grains_contours in contours.items():
        phase_grains_dict[phase] = manager.list()
        args = (phase_grains_dict[phase], grains_contours, phase)
        arguments.append(args)

    start_time = time.time()
    with Pool(ImageConfig.colorsNumber) as pool:
        pool.starmap(parallel_instances_generator, arguments)

    print("Grains instances generator multiprocessing time is: " + str(time.time() - start_time))
    for phase in phase_grains_dict:
        print(phase)
        print(len(phase_grains_dict[phase]))


def parallel_instances_generator(phase_grains_list: list, phase_grains_contours: list, phase: str):
    for grain_contours in phase_grains_contours:
        grain = GrainClass(grain_contours, phase)
        if grain.area <= grain.perimeter:
            del grain
            continue
        grain.start_calculating()
        phase_grains_list.append(grain)
    return phase_grains_list
