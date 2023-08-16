from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model)


model_input = tokenizer(
    [
        "You are a data labeller. I will give you a product review: 'I, like many others, are having \"sticker shock\" over the price of cartridge razors. I decide to go back to what my Dad and Grandfather did years ago, use a double-edge razor. This razor is a work of art. It's heavy, yet well balanced. I was a bit apprehensive when I made the first pull, but was amazed by how easy it shaved (Using the included blade. I ordered some BIC blades which are supposed to be sharper). The finish on the razor is superb. The shave was the best I've had in years. Looking forward to many years of service.' What are the uses cases described in this product review? Give me a short list and say 'No use cases' if there are no use cases."
    ],
    return_tensors="pt",
)
print(model_input)
model = AutoModelForCausalLM.from_pretrained(
    "tiiuae/falcon-7b-instruct",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
input_ids = model_input["input_ids"].to("cuda")
attention_mask = model_input["attention_mask"].to("cuda")
output = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    return_dict_in_generate=True,
    max_new_tokens=50
    # generation_config={"do_sample": False},
)
print(output)
print(tokenizer.batch_decode(output["sequences"]))
