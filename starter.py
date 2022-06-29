from binary_images_generator import generate_binary_images
from config.image_config import ImageConfig
from config.devices_info import devices_info
from contours_finder import find_contours
from statistics_classes.statistics_ratios_gpu_class import StatisticsGPU
from statistics_classes.statistics_ratios_cpu_class import StatisticsCPU
from statistics_classes.statistics_ratios_class import Statistics
from grain_instances_generator import generate_grains_instances_sequentially_gpu

if __name__ == '__main__':
    devices_info()
    image_path = 'RealImages/DP800-200x200.png'
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

    # contours = find_contours(phase_layers)
    # find_contours_threading(phase_layers)

    # phase_grains_dict = generate_grains_instances_sequentially_gpu(contours)
    # phase_grains_dict = generate_grains_instances_threading(contours)
    # phase_grains_dict = generate_grains_instances_sequentially(contours)
    # phase_grains_dict = generate_grains_instances_threading_with_gpu(contours)
    # phase_grains_dict = generate_grains_instances_sequentially_with_parallel_calculations_cpu(contours)
    # phase_grains_dict = generate_grains_instances_threading_with_numba_cpu(contours)
    # phase_grains_dict = generate_grain_instances_threads_per_grain(contours)
    # _ = create_series_from_ratios(phase_grains_dict)
    # save_results_to_excel_file(phase_grains_dict, "seq_gpu.xlsx")
    colors = {
        'ferrite': (0, 255, 0),
        'bainite': (0, 0, 255),
        'martensite': (255, 0, 0),
    }
    stats = Statistics(grains=[], scale=1)
    stats.lineal_path()
    # stats.one_point_prob()
    statsGPU = StatisticsGPU(grains=[])
    statsGPU.lineal_path()
    # statsGPU.one_point_prob()
    # statsGPU.blr()
    # statsCPU = StatisticsCPU(grains=[])
    # statsCPU.one_point_prob()
    # statsCPU.blr()
    # x, y = Rc().calculate_ratios(image=image, background='bainite')
    # print(x, y)
