import os
import random
import argparse
import glob
import time
import paramiko
from scp import SCPClient
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional


def is_true(value: Optional[str]):
    return isinstance(value, str) and value.lower() == "true"


def get_args():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-u",
        "--user",
        type=str,
        default=os.getenv("USER_NAME"),
        help="Which user connects to the server",
    )
    parser.add_argument(
        "-lp",
        "--local_path",
        type=str,
        default=os.getenv("LOCAL_PATH"),
        help="Where should the files be stored on the local machine",
    )
    parser.add_argument(
        "-k",
        "--key_filename",
        type=str,
        default=os.getenv("SSH_FILE_NAME"),
        help="Which key is used to connect to the server",
    )
    parser.add_argument(
        "-f",
        "--files",
        type=str,
        default=os.getenv("FILES_TO_BE_UPLOADED"),
        help="Which files should be shared",
    )
    parser.add_argument(
        "-ce",
        "--conda_environment",
        type=str,
        default=os.getenv("CONDA_ENVIRONMENT"),
        help="Which conda environment should be used",
    )
    parser.add_argument(
        "-uc",
        "--update_conda",
        default=os.getenv("UPDATE_CONDA"),
        help="Should the conda enviroment be updated",
    )
    parser.add_argument(
        "-cp",
        "--cluster_path",
        default=os.getenv("CLUSTER_PATH"),
        help="Where to copy the scripts on the cluster",
    )
    parser.add_argument(
        "-rm",
        "--remove_folder",
        default=os.getenv("REMOVE_FOLDER"),
        help="Boolean flag to decide, if job folder should be deleted after the job is done",
    )
    args = parser.parse_args()
    return args


def open_ssh_connection():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        "summon.delab.i.hpi.de",
        username=args["user"],
        key_filename=args["key_filename"],
    )
    print("Connection established")
    return ssh


def copy_files_to_server():
    print(f"Remote path: {remote_path}")
    if args["files"] == "all":
        data_files = glob.glob("*.py")
        data_files.extend(glob.glob("*.slurm"))
        if is_true(args["update_conda"]):
            data_files.append("../requirements.txt")
        print(f"Going to copy the files: {data_files}")
    else:
        data_files = args["files"].split(" ")

    __, __, stderr = ssh.exec_command(f"mkdir -p {remote_path}/sbatch_out")
    check_for_errors(stderr)

    for file in data_files:
        if file != "job_starter.py":
            scp.put(file, remote_path)
            print(f"Copied file: {file}")

    print("Files on server")


def update_conda():
    __, stdout, stderr = ssh.exec_command(
        f"source /hpi/fs00/home/{args['user']}/miniconda3/etc/profile.d/conda.sh;conda install -n {args['conda_environment']} --file requirements.txt"
    )
    check_for_errors(stderr)
    print(stdout.read().decode("utf-8"))


def start_job():
    print("Starting job")
    __, stdout, stderr = ssh.exec_command(
        f"source /hpi/fs00/home/{args['user']}/miniconda3/etc/profile.d/conda.sh;conda activate {args['conda_environment']};cd {remote_path};sbatch job.slurm"
    )
    check_for_errors(stderr)
    job_id = stdout.read().decode("utf-8").split(" ")[-1].strip()
    print(f"Job started with id: {job_id}")
    return job_id


def display_output():
    print("Displaying output")
    while True:
        __, stdout0, stderr = ssh.exec_command(
            f"sacct -j {job_id} -X -o State |  sed -n 3p"
        )
        check_for_errors(stderr)
        state = stdout0.read().decode("utf-8").strip()
        if state != "PENDING" and state != "RUNNING":
            break
        time.sleep(5)

    # Need to give slurm time to write the output file
    time.sleep(3)
    print("Job finished")

    __, stdout, stderr1 = ssh.exec_command(f"cat {remote_path}/sbatch_out/slurm.out")
    out_string = stdout.read().decode("utf-8")
    check_for_errors(stderr1)

    __, stdout, stderr2 = ssh.exec_command(f"cat {remote_path}/sbatch_out/slurm.err")
    check_for_errors(stderr2)
    err_string = stdout.read().decode("utf-8")

    print("Output: " + out_string)
    print("Error: " + err_string)

    local_path = f"{args['local_path']}/job_starter/sbatch_out_{job_id}_{random_job_id}"
    Path(local_path).mkdir(parents=True, exist_ok=True)
    scp.get(f"{remote_path}/sbatch_out", local_path, recursive=True)


def check_for_errors(stderr):
    error = stderr.read().decode("utf-8")
    if error:
        print("Error\n")
        print(error)
        terminate_connection()
        exit(1)


def terminate_connection():
    if is_true(args["remove_folder"]):
        __, __, stderr = ssh.exec_command(f"rm -rf {remote_path}")
        check_for_errors(stderr)

    scp.close()
    ssh.close()
    print("Connection terminated")


if __name__ == "__main__":
    random_job_id = random.randint(0, 1000000)
    print(f"Random job id: {random_job_id}")
    args = vars(get_args())
    if args["cluster_path"] == "None":
        remote_path = f"/hpi/fs00/home/{args['user']}/job_starter/job_{random_job_id}"
    else:
        remote_path = f"/hpi/fs00/home/{args['user']}/job_starter/{args['cluster_path']}/job_{random_job_id}"

    ssh = open_ssh_connection()
    scp = SCPClient(ssh.get_transport())
    copy_files_to_server()

    if is_true(args["update_conda"]):
        update_conda()
    job_id = start_job()
    display_output()
    terminate_connection()
