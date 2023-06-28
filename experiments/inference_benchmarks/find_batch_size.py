"""
WIP!!
"""

import time
import torch


def find_optimal_batch_size(model, sequence_length: int):
    START_BATCH_SIZE = 8
    optimal_sequence_length = sequence_length + (1 if sequence_length % 2 != 0 else 0)

    def test_batch_size(model, sequence_length: int, batch_size: int) -> bool:
        print(
            f"Testing sequence length: {sequence_length} with batch_size: {batch_size}"
        )
        torch.cuda.empty_cache()
        input_ids = torch.full((batch_size, sequence_length), 5).to("cuda")

        try:
            with torch.no_grad():
                output = model(input_ids=input_ids)
            return True
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(
                    f"| WARNING: ran out of memory, batch of size: {batch_size} with sequence_length: {sequence_length}"
                )
                time.sleep(5)
                torch.cuda.empty_cache()
                return False
            else:
                raise e

    def find_optimal_batch_size(lower_batch_size, upper_batch_size):
        batch_size = int((upper_batch_size - lower_batch_size) / 2 + lower_batch_size)
        if batch_size % 8 != 0:
            return lower_batch_size
        elif test_batch_size(model, optimal_sequence_length, batch_size):
            return find_optimal_batch_size(batch_size, upper_batch_size)
        else:
            return find_optimal_batch_size(lower_batch_size, batch_size)

    batch_size = START_BATCH_SIZE
    while test_batch_size(model, optimal_sequence_length, batch_size):
        batch_size = batch_size * 2

    print("Failed at batch size:", batch_size)

    return find_optimal_batch_size(int(batch_size / 2), batch_size)


torch.cuda.empty_cache()
find_optimal_batch_size(model.get_encoder(), 512)
