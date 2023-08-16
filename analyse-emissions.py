#%%
from training.utils import get_model_dir_file_path
import pandas as pd
#subset_clustered_entropy_b32_run_4

emissions_total = []
for i in ["1", "2", "3"]:
    artifact = f"random_baseline_b32_run_{i}-active_learning_dir"

    emissions = pd.read_csv(get_model_dir_file_path(artifact, "emissions.csv"))
    emissions = emissions[:-3]
    #print(emissions)

    emissions = emissions[emissions["tracking_name"] != "active_learning_iteration"]
    #print(emissions.columns)
    emissions["duration"] = emissions["duration"] / 60
    emissions_total.append(emissions)

df = pd.concat(emissions_total)
r = df.groupby("tracking_name")[["duration", "energy_consumed"]].agg(["sum", "mean"])
r[("duration", "sum")] = r[("duration", "sum")] / 3
r[("energy_consumed", "sum")] = r[("energy_consumed", "sum")] / 3
r
#%%
emissions[["cpu_model", "cpu_count", "ram_total_size"]]
#%%
import seaborn as sns
import matplotlib.pyplot as plt
data = emissions
#fig, ax1 = plt.subplots()
sns.lineplot(data, x="iteration", y="duration", hue="tracking_name")
plt.savefig("active_learning_iteration_duration_greedy_entropy_b32_run_1.png")
#ax1.tick_params(axis="y", labelcolor="b")

# Create a second y-axis (right side) for the computation duration
#ax2 = ax1.twinx()
#
