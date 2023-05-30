from training.generator import Generator, GENERATION_CONFIGS
from training.utils import get_dataset_path
from helpers.review_set import ReviewSet
from helpers.label_selection import DatasetSelectionStrategy, LabelIDSelectionStrategy
import time

ARTIFACT = "splendid-forest-441"
DATASET_NAME = "blinding-light-99"
GOLD_DATASET = "/hpi/fs00/share/fg-demelo/bsc2022-usageinfo/data_labeled/golden_dataset/golden_v3.json"
golden_label_id = "bp-golden_v3"
MAX_REVIEWS = 10

USE_GOLDEN = True

if USE_GOLDEN:
    test_dataset_sample = ReviewSet.from_files(GOLD_DATASET)
    label_selection_strategy = LabelIDSelectionStrategy(golden_label_id)
else:
    label_selection_strategy = DatasetSelectionStrategy((DATASET_NAME, "test"))

    test_dataset = ReviewSet.from_files(
        get_dataset_path(DATASET_NAME)
    ).filter_with_label_Pstrategy(
        label_selection_strategy,
        inplace=False,
    )

    test_dataset_sample, _ = test_dataset.split(
        MAX_REVIEWS / len(test_dataset), seed=42069
    )

print(f"Using test dataset with length {len(test_dataset_sample)}")

scores = None
for config_name in GENERATION_CONFIGS:
    generator = Generator(ARTIFACT, config_name, checkpoint=6)
    label_id = f"{config_name}-{ARTIFACT}"
    start_time = time.time()
    generator.generate_label(test_dataset_sample, label_id)
    end_time = time.time()
    reports = test_dataset_sample.get_scores_report(
        LabelIDSelectionStrategy(label_id), label_selection_strategy
    )
    print("Saving stats for generation config:", config_name)
    for i, plot in enumerate(reports):
        plot.figure.savefig(f"{config_name}-report-{i}.png", bbox_inches="tight")
