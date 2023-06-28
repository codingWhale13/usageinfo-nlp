from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import time
import torch._dynamo as torchdynamo
import torch
from kernl.model_optimization import optimize_model

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# default cache size needs to be increased to store the many graphs with generative models
torchdynamo.config.cache_size_limit = 512

model_name = "google/flan-t5-base"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model = model.eval().cuda()

tokenizer = AutoTokenizer.from_pretrained(model_name)

input_ids = tokenizer(
    [
        "translate English to French: The house in the woods is wonderful, can we buy it ?",
        "translate French to German: The world is new in the woods is wonderful, can we buy it ?",
    ]
    * 128,
    return_tensors="pt",
    pad_to_multiple_of=8,
    padding=True,
).to("cuda")

with torch.inference_mode(), torch.autocast(cache_enabled=True, device_type="cuda"):
    for _ in range(10):
        output = model.generate(
            inputs=input_ids["input_ids"],
            min_length=22,
            max_length=22,
        )
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = model.generate(
        inputs=input_ids["input_ids"],
        min_length=22,
        max_length=22,
    )
    torch.cuda.synchronize()
    latency_baseline = time.perf_counter() - start
    print(latency_baseline)

optimize_model(model.encoder)
optimize_model(model.decoder)

# warmup (IRL, encoder and decoder should be warmed each on their own)
with torch.inference_mode(), torch.autocast(cache_enabled=True, device_type="cuda"):
    start = time.perf_counter()
    model.generate(inputs=input_ids["input_ids"], min_length=22, max_length=22)
    print(time.perf_counter() - start)


with torch.inference_mode(), torch.autocast(cache_enabled=True, device_type="cuda"):
    for _ in range(10):
        model.generate(
            inputs=input_ids["input_ids"],
            min_length=22,
            max_length=22,
        )
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = model.generate(
        inputs=input_ids["input_ids"],
        min_length=22,
        max_length=22,
    )
    torch.cuda.synchronize()
    latency_optimized = time.perf_counter() - start
    print(latency_optimized)
    print(f"{latency_baseline/latency_optimized:.1f}x speedup")
    print(
        tokenizer.decode(
            output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
    )
