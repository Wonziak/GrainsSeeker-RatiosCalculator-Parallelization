import cv2


def find_contours(phase_layers=dict):
    for phase, layer in phase_layers.items():
        print(phase)
        print(layer)
