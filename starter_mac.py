from binary_images_generator import generate_binary_images
from config.image_config import ImageConfig
from contours_finder import find_contours, find_contours_threading
from statistics_classes.statistics_ratios_cpu_class import StatisticsCPU
from statistics_classes.statistics_ratios_class import Statistics
from grain_instances_generator import generate_grains_instances_threading, \
    generate_grains_instances_sequentially, \
    generate_grains_instances_sequentially_with_parallel_calculations_cpu, \
    generate_grains_instances_threading_with_numba_cpu

if __name__ == '__main__':
    for image in ['200x200', '400x400', '800x800', '1600x1600', '3200x3200']:
        print(f"\nCalculating for image {image}:\n")
        image_path = f'RealImages/{image}.png'
        image_config = ImageConfig.generate_image_info(image_path=image_path)
        phase_layers = generate_binary_images(method="CPU")
        phase_layers = generate_binary_images(method="CPU")
        phase_layers = generate_binary_images(method="SEQ")

        contours = find_contours(phase_layers)
        find_contours_threading(phase_layers)

        if image == '200x200':
            phase_grains_dict = generate_grains_instances_threading(contours)
            phase_grains_dict = generate_grains_instances_sequentially(contours)

        phase_grains_dict = generate_grains_instances_sequentially_with_parallel_calculations_cpu(contours)
        phase_grains_dict = generate_grains_instances_threading_with_numba_cpu(contours)

        # phase_grains_dict = generate_grain_instances_threads_per_grain(contours)
        # _ = create_series_from_ratios(phase_grains_dict)
        # save_results_to_excel_file(phase_grains_dict, "seq_gpu.xlsx")

        stats = Statistics(grains=phase_grains_dict, scale=1)
        stats.lineal_path(20000)
        stats.one_point_prob()
        stats.dispersion()

        statsCPU = StatisticsCPU(grains=phase_grains_dict)
        statsCPU.lineal_path(20000)
        statsCPU.lineal_path(20000)
        statsCPU.one_point_prob()
        statsCPU.one_point_prob()
        statsCPU.blr()
        statsCPU.blr()
        statsCPU.dispersion()
        print(f"\n Calculating for image {image} ended.\n")
    exit(0)
