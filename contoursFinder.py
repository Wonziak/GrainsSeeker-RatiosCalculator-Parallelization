import cv2
import numpy as np
import ImageConfig
from numba import njit, cuda, prange
from timeit import default_timer as timer
import math

ImageConfig.image = cv2.imread('RealImages/2400x2400.png')
ImageConfig.height, ImageConfig.width = ImageConfig.image.shape[:2]
image = ImageConfig.image


def find_layers_gpu(colors):  # problem równoleglości zadań: jeden obraz, kilka kolorów do wyszukania
    phaselayers = []
    layer = np.zeros((ImageConfig.height, ImageConfig.width, 1), np.uint8)
    x_gpu = cuda.to_device(image)

    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(x_gpu.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(x_gpu.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    for color in colors.values():
        out_gpu = cuda.device_array_like(layer)
        iterate_on_image_compare_color_cuda[blockspergrid, threadsperblock](x_gpu, color, out_gpu)
        cuda.synchronize()
        array = out_gpu.copy_to_host()
        phaselayers.append(array)
    return phaselayers


def find_layers_sequentially(colors):  # problem równoleglości zadań: jeden obraz, kilka kolorów do wyszukania
    phaselayers = np.zeros((ImageConfig.height, ImageConfig.width, len(ImageConfig.colors_map)), np.uint8)
    index = 0
    for color in colors.values():
        for i in range(ImageConfig.width):
            for j in range(ImageConfig.height):
                if ImageConfig.image[j, i, 0] == color[2] and ImageConfig.image[j, i, 1] == color[1] and \
                        ImageConfig.image[
                            j, i, 2] == color[0]:  # BGR nie RGB
                    phaselayers[j, i, index] = 255
        index = index + 1
    return phaselayers


@cuda.jit
def iterate_on_image_compare_color_cuda(image, color, layer):
    j, i = cuda.grid(2)
    m, n = image.shape[:2]

    if 1 <= i < n - 1 and 1 <= j < m - 1:
        if image[j, i, 0] == color[2] and image[j, i, 1] == color[1] and \
                image[j, i, 2] == color[0]:  # BGR nie RGB
            layer[j, i, 0] = 255


@njit(parallel=True)
def iterate_on_image_compare_color_cpu(image, layer, color):
    for i in prange(ImageConfig.width):
        for j in prange(ImageConfig.height):
            if image[j, i, 0] == color[2] and image[j, i, 1] == color[1] and image[j, i, 2] == color[0]:  # BGR nie RGB
                layer[j, i, 0] = 255


def find_layers_cpu_parallel(colors):  # problem równoleglości zadań: jeden obraz, kilka kolorów do wyszukania
    phaselayers = []
    for color in colors.values():
        layer = np.zeros((ImageConfig.height, ImageConfig.width, 1), np.uint8)
        iterate_on_image_compare_color_cpu(image, layer, color)
        phaselayers.append(layer)
    return phaselayers


start = timer()
layers = find_layers_gpu(ImageConfig.colors_map)
print("Parallel 1 GPU " + str(timer() - start))

start = timer()
layers = find_layers_gpu(ImageConfig.colors_map)
print("Parallel 2 GPU " + str(timer() - start))

start = timer()
layers = find_layers_sequentially(ImageConfig.colors_map)
print("CPU Seq 1 " + str(timer() - start))

start = timer()
layers = find_layers_sequentially(ImageConfig.colors_map)
print("CPU Seq 2 " + str(timer() - start))


start = timer()
layers = find_layers_cpu_parallel(ImageConfig.colors_map)
print("CPU Parallel 1 " + str(timer() - start))

start = timer()
layers = find_layers_cpu_parallel(ImageConfig.colors_map)
print("CPU Parallel 2 " + str(timer() - start))


# cv2.imshow('binary', layers)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

for i in range(3):
    cv2.imshow('binary', layers[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# contours, hierarchy = cv2.findContours(layer, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
