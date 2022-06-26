import cv2


class ImageConfig:
    image = []
    imageCopy = []
    width = 0
    height = 0
    colorsNumber = 0
    colors_map = {}
    heightOffset = 0
    widthOffset = 0
    background = []
    numbers = [1, 3, 5, 11, 19, 21, 23, 27]
    color_number = {}

    @staticmethod
    def generate_image_info(image_path, colors_map=None, background=None):
        ImageConfig.image = cv2.imread(image_path)
        ImageConfig.imageCopy = ImageConfig.image
        ImageConfig.height, ImageConfig.width = ImageConfig.image.shape[:2]

        if colors_map:
            ImageConfig.colors_map = colors_map
        else:
            ImageConfig.colors_map = {
                'ferrite': (29, 143, 255),
                'bainite': (172, 255, 46),
                'martensite': (255, 0, 0)
            }
        if background:
            ImageConfig.background = background
        ImageConfig.colorsNumber = len(ImageConfig.colors_map)
        ImageConfig.heightOffset = 0
        ImageConfig.widthOffset = 0
        iterator_numbers = iter(ImageConfig.numbers)
        for key in ImageConfig.colors_map.keys():
            ImageConfig.color_number[key] = next(iterator_numbers)
