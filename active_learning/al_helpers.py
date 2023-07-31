from active_learning.module import (
    AbstractActiveDataModule,
    ActiveDataModule,
    ActiveLearningDataModule,
    NullActiveDataModule,
)
from copy import deepcopy
from active_learning.metrics.entropy import (
    EntropyActiveLearningMetric,
    RandomEntropySampleActiveLearningMetric,
)
from active_learning.metrics.least_confidence import LeastConfidenceActiveLearningMetric
from active_learning.metrics.random_baseline import RandomAquisitionFunction

from active_learning.sampler import GreedySampler, GreedyOptimalSubsetSampler, GreedyOptimalClusterSubsetSampler

from typing import Union


def load_sampler(sampler_config):
    sampler_name = sampler_config["name"]
    sampler_parameters = sampler_config.get("parameters", {})

    sampler_class = globals()[sampler_name]
    return sampler_class(**sampler_parameters)


def load_metric(metric_config):
    metric_name = metric_config["name"]
    metric_parameters = metric_config.get("parameters", {})
    metric_class = globals()[metric_name]
    return metric_class(**metric_parameters)


def load_active_data_module(
    name: str, parameters: dict = {}
) -> AbstractActiveDataModule:
    parameters = deepcopy(parameters)
    additional_parameters = {}

    if "sampler" in parameters:
        additional_parameters["sampler"] = load_sampler(parameters["sampler"])
        del parameters["sampler"]
    if "metric" in parameters:
        additional_parameters["metric"] = load_metric(parameters["metric"])
        del parameters["metric"]
    additional_parameters = additional_parameters | parameters

    print("Loading with:", additional_parameters)

    if name == "ActiveDataModule":
        return ActiveDataModule(**additional_parameters)
    elif name == "NullActiveDataModule":
        return NullActiveDataModule()
    elif name == "ActiveLearningDataModule":
        return ActiveLearningDataModule(**additional_parameters)
    else:
        raise ValueError(f"Unknown active data module {name}")
