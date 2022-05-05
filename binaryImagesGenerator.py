import cv2
import numpy as np
from Config.ImageConfig import ImageConfig
from numba import njit, cuda, prange
import math
import time


def generate_binary_images(method="SEQ"):
    colors = ImageConfig.colors_map
    image = ImageConfig.image
    if method.__eq__("SEQ"):
        return __find_layers_sequentially(image, colors)
    if method.__eq__("GPU"):
        return __find_layers_gpu(image, colors)
    if method.__eq__("CPU"):
        return __find_layers_cpu_parallel(image, colors)
    raise ValueError("Method has to be: 'SEQ','GPU', 'CPU'")


def __find_layers_sequentially(image, colors):
    phase_layers = {}
    start_time = time.time()
    for phase in colors.keys():
        if phase in ImageConfig.background:
            continue
        color = colors[phase]
        layer = np.zeros((ImageConfig.height, ImageConfig.width, 1), np.uint8)
        for i in range(ImageConfig.width):
            for j in range(ImageConfig.height):
                if image[j, i, 0] == color[2] and image[j, i, 1] == color[1] and image[j, i, 2] == color[0]:
                    layer[j, i, 0] = 255
        phase_layers[phase] = layer
    print("Sequentially time: " + str(time.time() - start_time))
    return phase_layers


def __find_layers_gpu(image, colors):
    phase_layers = {}
    empty_array = np.zeros((ImageConfig.height, ImageConfig.width, 1), np.uint8)
    x_gpu = cuda.to_device(image)

    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(x_gpu.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(x_gpu.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    start_time = time.time()
    for phase in colors.keys():
        if phase in ImageConfig.background:
            continue
        color = colors[phase]
        out_gpu = cuda.device_array_like(empty_array)
        __iterate_on_image_compare_color_cuda[blockspergrid, threadsperblock](x_gpu, color, out_gpu)
        cuda.synchronize()
        layer = out_gpu.copy_to_host()
        phase_layers[phase] = layer
    print("GPU parallel time: " + str(time.time()-start_time))
    return phase_layers


@cuda.jit
def __iterate_on_image_compare_color_cuda(image, color, layer):
    j, i = cuda.grid(2)
    m, n = image.shape[:2]

    if 1 <= i < n - 1 and 1 <= j < m - 1:
        if image[j, i, 0] == color[2] and image[j, i, 1] == color[1] and image[j, i, 2] == color[0]:
            layer[j, i, 0] = 255


def __find_layers_cpu_parallel(image, colors):
    phase_layers = {}
    start_time = time.time()
    for phase in colors.keys():
        if phase in ImageConfig.background:
            continue
        color = colors[phase]
        layer = np.zeros((ImageConfig.height, ImageConfig.width, 1), np.uint8)
        __iterate_on_image_compare_color_cpu(image, ImageConfig.width, ImageConfig.height, layer, color)
        phase_layers[phase] = layer
    print("CPU parallel time: " + str(time.time() - start_time))
    return phase_layers


@njit(parallel=True)
def __iterate_on_image_compare_color_cpu(image, width, height, layer, color):
    for i in prange(width):
        for j in prange(height):
            if image[j, i, 0] == color[2] and image[j, i, 1] == color[1] and image[j, i, 2] == color[0]:
                layer[j, i, 0] = 255


def show_layers(phase_layers=dict):
    for layer in phase_layers.values():
        cv2.imshow('binary', layer)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
