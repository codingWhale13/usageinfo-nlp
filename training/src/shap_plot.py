import matplotlib.pyplot as plt
import argparse
from utils import get_model_config_from_model_name
import shap

def arg_parse():
    parser = argparse.ArgumentParser(
        description="Predict usage options on given dataset using a trained model."
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input text to generate shap values for",
    )
    parser.add_argument(
        "prediction",
        type=str,
        help="Prediction of the model",
    )
    parser.add_argument(
        "artifact_name",
        type=str,
        help="Name of model artifact to use (wandb run name)",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=int,
        default=None,
        help="Checkpoint to use for prediction (default is last)",
    )
    return parser.parse_args(), parser.format_help()

def generate_shap(model, tokenizer, input, prediction):
    assert len(prediction.split()) == 1, "Prediction must be a single word"
    explainer = shap.Explainer(model, tokenizer)
    shap_values = explainer([input], [prediction])
    plt.figure(constrained_layout=True, figsize=(15, 5))

    #shap.plots.bar(shap_values[0, :, prediction], max_display=20)
    shap.plots.waterfall(shap_values[0, :, prediction], max_display=20)
    plt.savefig('shap.png')

def main():
    args, _ = arg_parse()
    artifact_name = args.artifact_name
    checkpoint = args.checkpoint
    input=args.input
    prediction=args.prediction
    (
        model,
        tokenizer,
        __,
    ) = get_model_config_from_model_name(artifact_name, checkpoint)

    generate_shap(model, tokenizer, input, prediction)

if __name__ == "__main__":
    main()