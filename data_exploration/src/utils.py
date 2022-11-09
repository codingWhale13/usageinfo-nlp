import multiprocessing
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
import dask.dataframe as dd
import logging


def apply_to_files_with_multithreading(map_function, num_processes: int, files: list):
    pool = multiprocessing.Pool(processes=num_processes)
    return pool.map(map_function, files)


def get_slurm_client(
    walltime="02:00:00",
    cores=128,
    processes=32,
    memory="256GB",
    account="demelo-student",
    queue="magic",
    nodes: int = 1,
    extra_args=["--output=/dev/null", "--exclude nvram-[01-06]"],
    logging=False,
):
    if logging:
        logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)

    cluster = SLURMCluster(
        walltime=walltime,
        cores=cores,
        processes=processes,
        memory=memory,
        account=account,
        shebang="#!/usr/bin/env bash",
        queue=queue,
        job_extra_directives=extra_args,
        local_directory="~/.tmp/dask-worker-space",
        # worker_extra_args=["--lifetime", "55m", "--lifetime-stagger", "4m"]
    )

    cluster.scale(jobs=nodes)

    return Client(cluster)
