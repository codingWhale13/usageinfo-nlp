# original code: https://colab.research.google.com/drive/1Hrz6Q7GFdH5OVg3Dis86f5OqiGdkDfRP?usp=sharing
# pip install git+https://github.com/keirp/automatic_prompt_engineer

import json

import openai
from automatic_prompt_engineer import ape

openai.api_key = ""
JSON_PATH = "prompt_labelled_v3.json"

dataset_in = []
dataset_out = []

with open(JSON_PATH) as file:
    data = json.load(file)
    for i, review in enumerate(data["reviews"]):
        inp = review["review_body"]
        usage_options = review["label"]["customUsageOptions"]
        outp = ", ".join(usage_options) if len(usage_options) > 0 else "no use case"
        dataset_in.append(inp)
        dataset_out.append(outp)
        if i > 10:
            break


dataset = [dataset_in, dataset_out]
print(len(dataset[0]))

eval_template = """Instruction: [PROMPT]
Input: [INPUT]
Output: [OUTPUT]"""

result, demo_fn = ape.simple_ape(
    dataset=dataset,
    eval_template=eval_template,
)

print(result)
