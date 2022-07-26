from grain_classes.grain_class import GrainClass
from grain_classes.grain_cpu_class import GrainCPUClass
from collections import defaultdict
import time
from multiprocessing import Pool, Manager
from config.image_config import ImageConfig
from numba import njit, prange


def generate_grains_instances_sequentially(contours: dict):
    phase_grains_dict = defaultdict(list)
    start_time = time.time()
    for phase in contours:
        for grain_contours in contours[phase]:
            grain = GrainClass(grain_contours, phase)
            if grain.area <= grain.perimeter:
                continue
            grain.start_calculating()
            phase_grains_dict[phase].append(grain)
    print("Grains instances generator sequentially time is: " + str(time.time() - start_time))
    # for phase in phase_grains_dict:
    #     print(phase)
    #     print(len(phase_grains_dict[phase]))
    save_to_file("./Results/sequentially.txt", phase_grains_dict)
    return phase_grains_dict


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

    print("Grains instances generator multithreading time is: " + str(time.time() - start_time))
    # for phase in phase_grains_dict:
    #     print(phase)
    #     print(len(phase_grains_dict[phase]))
    save_to_file("./Results/threading.txt", phase_grains_dict)
    return phase_grains_dict


def generate_grains_instances_sequentially_with_parallel_calculations_cpu(contours: dict):
    phase_grains_dict = defaultdict(list)
    start_time = time.time()
    for phase in contours:
        for grain_contours in contours[phase]:
            grain = GrainCPUClass(grain_contours, phase)
            if grain.area <= grain.perimeter:
                continue
            grain.start_calculating()
            phase_grains_dict[phase].append(grain)
    print(
        "Grains instances generator sequentially with parallel calculations on CPU time is: " + str(
            time.time() - start_time))
    # for phase in phase_grains_dict:
    #     print(phase)
    #     print(len(phase_grains_dict[phase]))
    save_to_file("./Results/parallel_numba.txt", phase_grains_dict)
    return phase_grains_dict


# This functions do not work because of type limitations of numba
# def generate_grain_instances_threads_per_grain(contours: dict):
#     start_time = time.time()
#     phase_grains_dict = {}
#     for phase in contours:
#         grains_list = [GrainClass(contours[phase][0], 'q')]
#         phase_grains_dict[phase] = __parallel_thread_per_grain(phase, contours[phase], grains_list)
#     print("Grains instances generator thread per grain time is: " + str(time.time() - start_time))
#
#
# @njit(parallel=True)
# def __parallel_thread_per_grain(phase: str, grain_contours: list, grains_list):
#     for i in prange(len(grain_contours)):
#         grain = GrainClass(grain_contours[i], phase)
#         if grain.area <= grain.perimeter:
#             continue
#         grain.start_calculating()
#         grains_list.append(grain)
#     return grains_list[1:]


def generate_grains_instances_threading_with_numba_cpu(contours: dict):
    manager = Manager()
    phase_grains_dict = {}
    arguments = []
    for phase, grains_contours in contours.items():
        phase_grains_dict[phase] = manager.list()
        args = (phase_grains_dict[phase], grains_contours, phase)
        arguments.append(args)

    start_time = time.time()
    with Pool(ImageConfig.colorsNumber) as pool:
        pool.starmap(parallel_instances_generator_with_numba, arguments)

    print("Grains instances generator multithreading with numba cpu time is: " + str(
        time.time() - start_time))
    save_to_file("./Results/threading_numba.txt", phase_grains_dict)
    # for phase in phase_grains_dict:
    #     print(phase)
    #     print(len(phase_grains_dict[phase]))
    return phase_grains_dict


def parallel_instances_generator(phase_grains_list: list, phase_grains_contours: list, phase: str):
    for grain_contours in phase_grains_contours:
        grain = GrainClass(grain_contours, phase)
        if grain.area <= grain.perimeter:
            continue
        grain.start_calculating()
        phase_grains_list.append(grain)
    return phase_grains_list


def parallel_instances_generator_with_numba(phase_grains_list: list, phase_grains_contours: list,
                                            phase: str):
    for grain_contours in phase_grains_contours:
        grain = GrainCPUClass(grain_contours, phase)
        if grain.area <= grain.perimeter:
            continue
        grain.start_calculating()
        phase_grains_list.append(grain)
    return phase_grains_list


def save_to_file(file_name, phase_grains_dict):
    pass
    # with open(file_name, "w") as file:
    #     for key in phase_grains_dict.keys():
    #         for grain in phase_grains_dict[key]:
    #             file.write(str(grain.area) + "\n")
    #             file.write(str(grain.width_range) + "\n")
    #             file.write(str(grain.height_range) + "\n")
    #             file.write(str(grain.centerOfMass) + "\n")
    #             file.write(str(grain.centerOfMassLocal) + "\n")
    #             file.write(str(grain.distanceFromCenterPowerSum) + "\n")
    #             file.write(str(grain.distanceFromEdgeToCenter) + "\n")
    #             file.write(str(grain.distanceFromEdgeToCenterSquared) + "\n")
    #             file.write(str(grain.minDistanceFromEdgeSum) + "\n")  # check
    #             file.write(str(grain.minDistaceCenterEdge) + "\n")
    #             file.write(str(grain.maxDistaceCenterEdge) + "\n")
    #             file.write(str(grain.maxDistancePoints) + "\n")
    #             file.write(str(grain.maxDistanceVectorCoords) + "\n")
    #             file.write(str(grain.LH) + "\n")
    #             file.write(str(grain.LW) + "\n")
    #             file.write((str(grain.VectorPerpendicularLength)) + "\n" + "\n")
