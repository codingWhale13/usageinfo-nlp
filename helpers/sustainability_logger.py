import subprocess
import threading
import time
from typing import Optional

import codecarbon
import wandb
import json

MEASUREMENT_INTERVAL = 5  # in seconds
experiment_running = False


def log_power_consumption(power_samples):
    command = ["nvidia-smi", "-q", "-d", "POWER"]
    while experiment_running:
        error_occured = False
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            if result.returncode != 0:
                error_occured = True
        except Exception as e:
            error_occured = True

        if error_occured:
            power_samples = [-1]
            print("Unable to collect power usage samples using nvidia-smi")
            break

        # extract power consumption (in Watt) from the command output
        for output_line in result.stdout.splitlines():
            if "Power Draw" in output_line:
                power_sample = float(output_line.split(": ")[1].split()[0])
                break
        power_samples.append(power_sample)

        time.sleep(MEASUREMENT_INTERVAL)


def compute_kWh(power_samples):
    """power_samples is a list of power measurements (in Watt) at time points"""
    if len(power_samples) == 0 or sum(power_samples) == 0:
        return 0  # avoid division by zero
    average_power = sum(power_samples) / len(power_samples) / 1000  # in kWh
    duration = MEASUREMENT_INTERVAL * len(power_samples)  # in seconds
    return average_power * duration / 3600


class SustainabilityLogger:
    """This class tracks emissions and power consumption using CodeCarbon nvidia-smi. Add log_file to log data in json file"""

    def __init__(
        self, description: Optional[str] = None, log_file: Optional[str] = None
    ):
        
        if log_file is not None and not log_file.endswith(".json"):
            log_file += ".json"

        self.description = description
        self.log_file = log_file
        self.power_measurements = []

    def __enter__(self):
        global experiment_running
        experiment_running = True

        # start library trackers
        self.codecarbon_tracker = codecarbon.EmissionsTracker(log_level="critical")
        self.codecarbon_tracker.start()

        # start logging kWh with nvidia-smi
        self.power_thread = threading.Thread(
            target=log_power_consumption, args=(self.power_measurements,)
        )
        self.power_thread.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        global experiment_running
        experiment_running = False

        prefix = "sustainability"
        if self.description is not None:
            prefix += f" ({self.description})"

        self.codecarbon_tracker.stop()

        #https://github.com/mlco2/codecarbon/blob/master/codecarbon/output.py#L27
        codecarbon_results_ordered_dict = self.codecarbon_tracker._prepare_emissions_data().values
        codecarbon_results_dict = dict(codecarbon_results_ordered_dict)

        results = {
            f"{prefix}/NVIDIA power_consumption(kWh)": compute_kWh(
                self.power_measurements
            ),
        } | codecarbon_results_dict

        self.power_thread.join()

        if wandb.run is not None:  # log to wandb if wandb.init() has been called
            wandb.log(results)

        if self.log_file is not None:
            with open(self.log_file, "w") as f:
               json.dump(results, f, indent=4)
