import torch
import time
from statistics import mean

TOTAL_ITERATIONS = 10000
sort_times = []
for _ in range(TOTAL_ITERATIONS):
    a = torch.rand([32000])
    start_time = time.perf_counter_ns()
    sorted, indices = torch.sort(a, descending=True, dim=-1)
    sort_times.append(time.perf_counter_ns() - start_time)


print(sum(sort_times) / 10 ** (-9), mean(sort_times))
top_k = []
for _ in range(TOTAL_ITERATIONS):
    a = torch.rand([32000])
    start_time = time.perf_counter_ns()
    sorted, indices = torch.topk(a, k=1000, dim=-1)
    sorted, indices = torch.sort(sorted, descending=True, dim=-1)
    top_k.append(time.perf_counter_ns() - start_time)

print(sum(top_k) / 10 ** (-9), mean(top_k))

print(mean(sort_times) / mean(top_k))
