import codecarbon
from codecarbon.output import EmissionsData
import pandas as pd
from typing import Union
from statistics import mean, variance


def compute_mean_emission(emissions: list[dict]):
    def accumalate(key, list):
        return [x[key] for x in list]

    data = {
        key: accumalate(key, emissions)
        for key in [
            "duration",
            "emissions",
            "cpu_energy",
            "gpu_energy",
            "ram_energy",
            "energy_consumed",
        ]
    }
    return {f"mean_{key}": mean(values) for key, values in data.items()} | {
        f"variance_{key}": variance(values) for key, values in data.items()
    }


class SustainabilityTracker:
    """This class tracks emissions and power consumption using CodeCarbon"""

    def __init__(self):
        self.codecarbon_tracker = codecarbon.EmissionsTracker()
        self.current_tracking_data = {}
        self.results = {}

    def initalize(self):
        self.codecarbon_tracker.start()

    def _tracking_key(tracking_name: str, iteration: Union[None, int] = None):
        return (tracking_name, iteration)

    def start(self, tracking_name: str, iteration: Union[None, int] = None) -> None:
        if tracking_name in self.current_tracking_data:
            raise ValueError(f"{tracking_name} already being tracked")
        self.current_tracking_data[
            self._tracking_key(tracking_name, iteration)
        ] = self.codecarbon_tracker._prepare_emissions_data()

    def stop(
        self, tracking_name: str, iteration: Union[None, int] = None
    ) -> EmissionsData:
        tracking_key = self._tracking_key(tracking_name, iteration)
        if tracking_key not in self.current_tracking_data:
            raise ValueError(
                f"{tracking_name} is not currently being tracked. Please call start first"
            )
        if tracking_key in self.results:
            raise ValueError(
                f"{tracking_name} already in results. Please don't reuse any names"
            )
        delta_emissions = (
            self.codecarbon_tracker._prepare_emissions_data().compute_delta_emission(
                self.current_tracking_data[tracking_key]
            )
        )
        del self.current_tracking_data[tracking_key]
        tracking_results = vars(delta_emissions) | {
            "tracking_name": tracking_name,
            "iteration": iteration,
        }
        self.results[tracking_key] = tracking_results
        return tracking_results

    def aggregated_results(self) -> list[dict]:
        aggregated_results = {}
        results = list(self.results.values())
        for (tracking_name, _), emission_data in self.results.items():
            if tracking_name in aggregated_results:
                aggregated_results[tracking_name].append(emission_data)
            else:
                aggregated_results[tracking_name] = [emission_data]

        for key, values in aggregated_results.items():
            if len(values) > 0:
                results.append(compute_mean_emission(values) | {"tracking_name": key})

        return results

    def results_to_dataframe(self) -> pd.DataFrame:
        data = self.aggregated_results()
        return pd.DataFrame.from_records(data)

    def terminate(self) -> dict:
        self.codecarbon_tracker.stop()
        return self.aggregated_results()
