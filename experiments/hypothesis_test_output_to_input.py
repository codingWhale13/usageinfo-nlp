#!/usr/bin/env python3
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers.utils import logging
import torch.nn.functional as f
import torch

logging.set_verbosity_error()
num_token_options = 4

enc_dec_tokenizer = AutoTokenizer.from_pretrained("t5-small", model_max_length=512)
enc_dec_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

dec_only_tokenizer = AutoTokenizer.from_pretrained("gpt2")
dec_only_model = AutoModelForCausalLM.from_pretrained("gpt2")

model_types = {
    "enc-dec": {"tokenizer": enc_dec_tokenizer, "model": enc_dec_model},
    "dec-only": {"tokenizer": dec_only_tokenizer, "model": dec_only_model},
}


def format_token_pos_output(tokens_with_probs: list, token_number: int):
    print(
        f"\n\t\tDistribution of tokens with probabilities for token position {token_number}:"
    )

    for i in range(num_token_options):
        print(
            f"\t\t\t Token: {tokens_with_probs[token_number]['token'][i]} \t\t",
            f"Prob: {tokens_with_probs[token_number]['prob'][i]}",
        )


def eval_model(model_input: str, model_type: str):
    model = model_types[model_type]["model"]
    tokenizer = model_types[model_type]["tokenizer"]

    encoded_input = tokenizer(model_input, return_tensors="pt")
    num_input_tokens = len(encoded_input["input_ids"][0])
    output = model.generate(
        **encoded_input,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=10,
    )

    # For decoder only models, the input tokens are also included in the output, so we need to remove them
    if model_type == "dec-only":
        output["sequences"] = output["sequences"][:, num_input_tokens:]

    probs = [f.softmax(scores, dim=1) for scores in output["scores"]]
    best_probs = [
        torch.argsort(probs[token_number][0, :], dim=-1, descending=True)[
            :num_token_options
        ]
        for token_number, _ in enumerate(output["sequences"][0][1:])
    ]
    tokens_with_probs = [
        {
            "token": [tokenizer.decode(token) for token in best_probs[token_number]],
            "prob": list(
                probs[token_number][0, best_probs[token_number]].cpu().numpy()
            ),
        }
        for token_number, _ in enumerate(output["sequences"][0][1:])
    ]

    tokens = [token_pos["token"][0] for token_pos in tokens_with_probs]

    return tokens, tokens_with_probs


def run_experiment(input_1: str, input_2: str, model_type: str):
    print(f"\n\n\nExperiment for model type {model_type}:")
    output_1, token_probs_1 = eval_model(input_1, model_type)
    output_2, token_probs_2 = eval_model(input_2, model_type)

    print(f"\n\tInput 1: {input_1}")
    print(f"\tOutput: {' | '.join(output_1)}")
    format_token_pos_output(token_probs_1, 0)
    format_token_pos_output(token_probs_1, 1)

    print(f"\n\tInput 2: {input_2}")
    print(f"\tOutput: {' | '.join(output_2)}")
    format_token_pos_output(token_probs_2, 0)
    format_token_pos_output(token_probs_2, 1)


enc_dec_input_1 = "A,B,C,D,"
enc_dec_input_2 = "A,B,C,D,A"
run_experiment(enc_dec_input_1, enc_dec_input_2, "enc-dec")

dec_only_input_1 = "A,B,C,D,"
dec_only_input_2 = "A,B,C,D,E"
run_experiment(dec_only_input_1, dec_only_input_2, "dec-only")
