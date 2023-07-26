import utils
import torch
import argparse

from helpers.review_set import ReviewSet


BATCH_SIZE = 32


def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble Generator")
    parser.add_argument(
        "base_file",
        type=str,
        help="Filepath of the base reviewset that you want to use",
    )
    parser.add_argument(
        "artifact_names",
        type=str,
        help="List of artifacts names to ensemble, seperated by commas",
    )
    parser.add_argument(
        "checkpoints",
        type=str,
        help="List of checkpoints to ensemble, seperated by commas",
    )
    parser.add_argument(
        "last_part_of_label_id",
        type=str,
        nargs="?",
        default=None,
        metavar="last_part_of_label_id",
        help="Last part (aka the unique identifier) of the label_id that the annotation should be (optionally) saved under",
    )
    parser.add_argument(
        "--prompt_id",
        "-p",
        type=str,
        default="original",
        help="Prompt id to use for all annotation. Default is original",
    )
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    models, tokenizer, max_length = initialize_models(
        args.checkpoints, args.artifact_names
    )
    dataloader, reviews = initialize_dataloader(
        args.base_file, tokenizer, max_length, args.prompt_id
    )
    label_metadata = initialize_metadata(
        args.artifact_names, args.checkpoints, args.prompt_id
    )
    label_id = f"model_ensemble_{args.artifact_names}_{args.last_part_of_label_id}"

    for batch in dataloader:
        inputs, predictions = generate_usage_options(batch, models, tokenizer)
        predictions = [
            format_usage_options(usage_options) for usage_options in predictions
        ]
        save_label(
            reviews,
            batch["review_id"],
            inputs,
            predictions,
            label_id,
            label_metadata,
        )


def initialize_models(checkpoints, artifact_names):
    models = []
    checkpoints = checkpoints.split(",")
    artifact_names = artifact_names.split(",")
    for i in range(len(checkpoints)):
        model_artifact = {"name": artifact_names[i], "checkpoint": checkpoints[i]}
        model, tokenizer, max_length, _ = utils.initialize_model_tuple(model_artifact)
        models.append(model)
    return models, tokenizer, max_length


def initialize_dataloader(base_file, tokenizer, max_length, prompt_id):
    reviews = ReviewSet.from_files(base_file)

    dataloader = reviews.get_dataloader(
        batch_size=BATCH_SIZE,
        num_workers=0,
        tokenizer=tokenizer,
        model_max_length=max_length,
        for_training=False,
        prompt_id=prompt_id,
    )
    return dataloader, reviews


def initialize_metadata(artifact_names, checkpoints, prompt_id):
    label_metadata = {
        "generator": {
            "artifact_names": artifact_names,
            "checkpoints": checkpoints,
            "prompt_id": prompt_id,
            "model_name": "ensemble",
        }
    }
    return label_metadata


def generate_usage_options(batch, models, tokenizer):
    MAX_LENGTH = 128
    decoder_input_ids = torch.zeros(
        [BATCH_SIZE, MAX_LENGTH + 1], dtype=torch.torch.int32
    )
    input_ids = batch["input"]["input_ids"]
    attention_mask = batch["input"]["attention_mask"]
    running = torch.ones([BATCH_SIZE], dtype=torch.bool)

    encoder_outputs = []
    for model in models:
        encoder_outputs.append(
            model.get_encoder()(input_ids=input_ids, attention_mask=attention_mask)
        )

    for i in range(MAX_LENGTH):
        current_logits = []
        for j, model in enumerate(models):
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    encoder_outputs=encoder_outputs[j],
                    return_dict=True,
                    decoder_input_ids=decoder_input_ids,
                )
            current_logits.append(outputs.logits[:, i, :])
        logits = sum(current_logits) / len(current_logits)
        log_probs = torch.nn.functional.softmax(logits, dim=-1)
        # To prevent the model from predicting tokens after the end token, we mask out the prediction, if it has finished
        prediction = torch.argmax(log_probs, dim=-1) * running
        decoder_input_ids[:, i + 1] = prediction
        for idx, pred in enumerate(prediction):
            if pred == 1:
                running[idx] = False
        if sum(running) == 0:
            break

    output = tokenizer.batch_decode(decoder_input_ids, skip_special_tokens=True)
    input_ids = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

    return input_ids, output


def format_usage_options(text_completion: str):
    return [
        usage_option.strip()
        for usage_option in text_completion.split(", ")
        if usage_option.strip()
    ]


def save_label(reviews, review_ids, inputs, predictions, label_id, label_metadata):
    for review_id, model_input, usage_options in zip(review_ids, inputs, predictions):
        reviews[review_id].add_label(
            label_id=label_id,
            usage_options=usage_options,
            metadata=label_metadata,
        )
        print(f"Review {review_id}")
        print(f"Model input:\n{model_input}")
        print(f"Usage options:\n\t{usage_options}", end="\n\n")


if __name__ == "__main__":
    main()
