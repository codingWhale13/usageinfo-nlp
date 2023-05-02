import codecarbon
import eco2ai
import wandb


def extract_from_eco2ai_data(data, key):
    return float(data[key][0])


class SustainabilityLogger:
    """This class tracks emissions and power consumption using CodeCarbon and Eco2AI."""

    def __init__(
        self,
        project_name: str = "RealLifeUsageOptions",
        experiment_description: str = "default experiment",
    ):
        self.project_name = project_name
        self.experiment_description = experiment_description

    def __enter__(self):
        # start trackers
        self.eco2ai_tracker = eco2ai.Tracker(
            project_name=self.project_name,
            experiment_description=self.experiment_description,
            ignore_warnings=True,
        )
        self.eco2ai_tracker.start()

        self.codecarbon_tracker = codecarbon.EmissionsTracker(log_level="critical")
        self.codecarbon_tracker.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # log Eco2AI data
        eco2ai_attributes = self.eco2ai_tracker._construct_attributes_dict()
        eco2ai_results = {
            f"{metric} ({self.experiment_description}, Eco2AI)": extract_from_eco2ai_data(
                eco2ai_attributes, metric
            )
            for metric in ["power_consumption(kWh)", "CO2_emissions(kg)"]
        }
        wandb.log(eco2ai_results)
        self.eco2ai_tracker.stop()

        # log CodeCarbon data
        emissions = self.codecarbon_tracker.stop()
        wandb.log(
            {
                f"CO2 emissions(kg) ({self.experiment_description}, CodeCarbon)": emissions
            }
        )
