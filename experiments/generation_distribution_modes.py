#!/usr/bin/env python3
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers.utils import logging
import torch.nn.functional as f
import torch
from training import utils

logging.set_verbosity_error()
num_token_options = 4
model = "t5-small"
artifact = {"name": "bright-salad-245", "checkpoint": None}

model, tokenizer, max_length = utils.get_model_config(model, artifact)


def format_token_pos_output(tokens_with_probs: list, token_number: int):
    print(
        f"\n\t\tDistribution of tokens with probabilities for token position {token_number}:"
    )

    for i in range(num_token_options):
        print(
            f"\t\t\t Token: {tokens_with_probs[token_number]['token'][i]} \t\t",
            f"Prob: {tokens_with_probs[token_number]['prob'][i]}",
        )


def eval_model(model, tokenizer, model_input: str, decoder_start: str = None):
    encoded_input = tokenizer(model_input, return_tensors="pt")
    forced_decoder_ids = None
    if decoder_start:
        forced_decoder_ids = list(
            enumerate(
                tokenizer(decoder_start, return_tensors="pt")["input_ids"][0].tolist()[
                    :-1
                ],
                start=1,
            )
        )

    output = model.generate(
        **encoded_input,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=10,
        forced_decoder_ids=forced_decoder_ids,
    )

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


def run_experiment(
    model,
    tokenizer,
    product_title: str,
    review_body: str,
    experiment_num: int,
    decoder_start: str = None,
):
    model_input = f"Product title: {product_title} \nReview body: {review_body}\n"
    model.eval()

    print(f"\n\n\nExperiment Number {experiment_num}:")
    output, token_probs = eval_model(
        model, tokenizer, model_input, decoder_start=decoder_start
    )

    print(f"\n\tInput 1: {model_input}")
    print(f"\tOutput: {' | '.join(output)}")
    for i in range(len(output)):
        format_token_pos_output(token_probs, i)


product_title_1 = "American Flag Crewneck Men's Fleece Pullover Sweatshirt"
review_body_1 = "I ordered a 5xl and it's a little snug.  I usually buy my flag shirts at Dollar General, but since this was a lightning deal I decided to pull the trigger.  See what I did there?  I love pulling triggers.<br /><br />My only gripe with this shirt is that it's not a confederate flag, but as is, this will work for almost any occasion. Weddings, funerals, work functions, I'll be wearing this to all of them."
run_experiment(model, tokenizer, product_title_1, review_body_1, 1)
run_experiment(
    model, tokenizer, product_title_1, review_body_1, 1, decoder_start="wedding"
)
run_experiment(
    model, tokenizer, product_title_1, review_body_1, 1, decoder_start="work"
)

product_title_2 = "Beach Umbrella Carry Bag (Umbrella Not Included) Large 53L x 12W in."
review_body_2 = "I love this bag, I use it every time I go to the beach.  I have two umbrellas and they both fit perfectly in this roomy bag!  The material is very sturdy too."
run_experiment(model, tokenizer, product_title_2, review_body_2, 2)
run_experiment(
    model, tokenizer, product_title_2, review_body_2, 2, decoder_start="travel"
)
run_experiment(
    model, tokenizer, product_title_2, review_body_2, 2, decoder_start="beach"
)

product_title_3 = "Kids Dream Girls Organza Floral Special Occasion Dress"
review_body_3 = "bought for my youngest daughter as a flower girl dress for my oldest daughters wedding....gorgeous we couldn't be happier"
run_experiment(model, tokenizer, product_title_3, review_body_3, 3)
run_experiment(
    model, tokenizer, product_title_3, review_body_3, 3, decoder_start="flower"
)
run_experiment(
    model, tokenizer, product_title_3, review_body_3, 3, decoder_start="wedding"
)
