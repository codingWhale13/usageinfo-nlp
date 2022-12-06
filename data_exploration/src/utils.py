import multiprocessing
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
import dask.dataframe as dd
import logging


def get_slurm_client(
    walltime="02:00:00",
    cores=128,
    processes=8,
    memory="512GB",
    account="demelo-student",
    queue="magic",
    nodes: int = 3,
    slurm_output_file="/dev/null",
    extra_args=[],
    suppress_output=True,
):
    if not suppress_output:
        logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)

    cluster = SLURMCluster(
        walltime=walltime,
        cores=cores,
        processes=processes,
        memory=memory,
        account=account,
        shebang="#!/usr/bin/env bash",
        queue=queue,
        job_extra_directives=[
            "--exclude nvram-[01-06]",
            f"--output={slurm_output_file}",
        ]
        + extra_args,
        local_directory="/tmp/$USER",
        # worker_extra_args=["--lifetime", "55m", "--lifetime-stagger", "4m"]
    )

    cluster.scale(jobs=nodes)

    return Client(cluster)
