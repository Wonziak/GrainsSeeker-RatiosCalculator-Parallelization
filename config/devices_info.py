from numba import cuda
import subprocess


def devices_info(cpu=True, gpu=True):
    if gpu:
        cc_cores_per_SM_dict = {
            (2, 0): 32,
            (2, 1): 48,
            (3, 0): 192,
            (3, 5): 192,
            (3, 7): 192,
            (5, 0): 128,
            (5, 2): 128,
            (6, 0): 64,
            (6, 1): 128,
            (7, 0): 64,
            (7, 5): 64,
            (8, 0): 64,
            (8, 6): 128
        }
        # the above dictionary should result in a value of "None" if a cc match
        # is not found.  The dictionary needs to be extended as new devices become
        # available, and currently does not account for all Jetson devices
        device = cuda.get_current_device()
        my_sms = getattr(device, 'MULTIPROCESSOR_COUNT')
        my_cc = device.compute_capability
        cores_per_sm = cc_cores_per_SM_dict.get(my_cc)
        total_cores = cores_per_sm * my_sms
        print("GPU total CUDA cores: ", total_cores)
    if cpu:
        print("CPU model: ", sep=" ")
        print(subprocess.check_output("wmic cpu get name, numberofcores, maxclockspeed", shell=True).strip().decode()
              [91:])


def return_sm_and_threads_of_gpu():
    cc_cores_per_SM_dict = {
        (2, 0): 32,
        (2, 1): 48,
        (3, 0): 192,
        (3, 5): 192,
        (3, 7): 192,
        (5, 0): 128,
        (5, 2): 128,
        (6, 0): 64,
        (6, 1): 128,
        (7, 0): 64,
        (7, 5): 64,
        (8, 0): 64,
        (8, 6): 128
    }
    device = cuda.get_current_device()
    my_sms = getattr(device, 'MULTIPROCESSOR_COUNT')
    my_cc = device.compute_capability
    cores_per_sm = cc_cores_per_SM_dict.get(my_cc)
    SMx2 = my_sms*2
    threads = int(cores_per_sm / 2)
    return SMx2, threads
