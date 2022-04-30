import cv2
from binaryImagesGenerator import generate_binary_images, show_layers
from ImageConfig import ImageConfig

if __name__ == '__main__':
    image_path = 'RealImages/fragmenty_kontury/DP800-200x200_kontury.png'
    image_config = ImageConfig.generate_image_info(image_path=image_path)
    layers = generate_binary_images(method="GPU")
    show_layers(layers)
    # ratiosToCalculateList = ['Malinowska',
    #                          'Blair Bliss',
    #                          'Danielsson',
    #                          'Haralick',
    #                          'Mz',
    #                          'RLS',
    #                          'RF',
    #                          'RC1',
    #                          'RC2',
    #                          'RCOM',
    #                          'LP1',
    #                          'LP2',
    #                          'LP3',
    #                          'curavture']
    # statsRatiosToCalculateList = ['BorderNeighbour',
    #                               'OnePointProbability',
    #
    #                               'Linealpath']
    colors = {
        'ferrite': (0, 255, 0),
        'bainite': (0, 0, 255),
        'martensite': (255, 0, 0),
    }


    # x, y = Rc().calculate_ratios(image=image, background='bainite')
    # print(x, y)
