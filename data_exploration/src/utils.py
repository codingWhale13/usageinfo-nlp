import multiprocessing


def apply_to_files_with_multithreading(map_function, num_processes: int, files: list):
    pool = multiprocessing.Pool(processes=num_processes)
    return pool.map(map_function, files)
