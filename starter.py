from binaryImagesGenerator import generate_binary_images, show_layers
from Config.ImageConfig import ImageConfig
from Config.DevicesInfo import devices_info
from countoursFinder import find_contours, find_contours_threading


if __name__ == '__main__':
    devices_info()
    image_path = 'RealImages/9600x9600.png'
    # image_path = 'RealImages/fragmenty_kontury/DP800-200x200_kontury.png'
    # color_map = ImageConfig.colors_map = {
    #             'ferrite': (29, 143, 255),
    #             'bainite': (172, 255, 46),
    #             'martensite': (255, 0, 0)
    # }
    image_config = ImageConfig.generate_image_info(image_path=image_path)
    phase_layers = generate_binary_images(method="GPU")
    phase_layers = generate_binary_images(method="GPU")
    phase_layers = generate_binary_images(method="CPU")
    phase_layers = generate_binary_images(method="CPU")
    # phase_layers = generate_binary_images(method="SEQ")

    find_contours(phase_layers)
    find_contours_threading(phase_layers)
    # show_layers(phase_layers)
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
