from binaryImagesGenerator import generate_binary_images, show_layers
from Config.ImageConfig import ImageConfig
from countoursFinder import find_contours


if __name__ == '__main__':
    image_path = 'RealImages/9600x9600.png'
    image_config = ImageConfig.generate_image_info(image_path=image_path)
    phase_layers = generate_binary_images(method="GPU")
    phase_layers = generate_binary_images(method="GPU")
    phase_layers = generate_binary_images(method="CPU")
    phase_layers = generate_binary_images(method="CPU")
    phase_layers = generate_binary_images(method="SEQ")
    phase_layers = generate_binary_images(method="SEQ")

    # find_contours(phase_layers)
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
