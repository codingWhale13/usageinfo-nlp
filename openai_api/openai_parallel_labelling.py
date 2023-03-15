import os
import argparse
import json
from helpers.review_set import ReviewSet
from openai_api.openai_labelling import generate_label, chat_models, model_name_mapping
import asyncio


def parse_args():
    def restricted_temperature(x):
        try:
            x = float(x)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{x} not a floating-point literal")

        if x < 0.0 or x > 2.0:
            raise argparse.ArgumentTypeError(f"{x} not in range [0.0, 2.0]")
        return x

    arg_parser = argparse.ArgumentParser(
        description="Let an OpenAI model label the reviews in a json file."
    )
    arg_parser.add_argument("--file", "-f", required=True, help="json file path")
    arg_parser.add_argument(
        "--prompt", "-p", default="nils_v2", help="Prompt to use for openai labelling"
    )
    arg_parser.add_argument(
        "--id", default="", help="Unique identifier for the new label"
    )
    arg_parser.add_argument(
        "--save",
        "-s",
        choices=range(1, 101),
        type=int,
        default=5,
        help="Make an intermediate save every n percent of processed reviews",
        metavar="[0-100]",
    )
    arg_parser.add_argument(
        "--model", "-m", default="text-davinci-003", help="OpenAI model to use"
    )
    arg_parser.add_argument(
        "--temperature",
        "-t",
        type=restricted_temperature,
        default=1,
        help="Set the temperature for the sampling",
        metavar="[0.0-2.0]",
    )
    arg_parser.add_argument(
        "--logprobs",
        "-l",
        choices=range(1, 6),
        type=int,
        default=5,
        help="Define how many logprobs you want to save",
        metavar="[1-5]",
    )

    return arg_parser.parse_args(), arg_parser.format_help()


class Worker:
    def __init__(self, queue: asyncio.Queue, n=10):
        self.n = n
        self.queue = queue
        self.semaphore = asyncio.Semaphore(self.n)

    async def run(self):
        tasks = []
        while True:
            try:
                func = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            tasks.append(asyncio.ensure_future(self.do_work(func)))
        await asyncio.gather(*tasks)

    async def do_work(self, func):
        async with self.semaphore:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, func)


async def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    args, help_text = parse_args()

    review_file_name = os.path.abspath(args.file)
    prompt_file_name = f"{script_dir}/prompts.json"

    prompt_id = args.prompt
    with open(prompt_file_name, "r") as prompt_file:
        prompts_json = json.load(prompt_file)
        prompt_type = "chat" if args.model in chat_models else "text"
        if prompt_id not in prompts_json[prompt_type].keys():
            exit(
                f"Prompt {prompt_id} of type {prompt_type} not found in {prompt_file_name}."
            )
        prompt = prompts_json[prompt_type][prompt_id]["prompt"]

    label_id = "{model_name}-{prompt_id}{hyphen}{id}".format(
        model_name=model_name_mapping[args.model],
        prompt_id=prompt_id,
        id=args.id,
        hyphen="-" if args.id != "" else "",
    )

    review_set = ReviewSet.from_files(review_file_name)
    count = 0

    if label_id in review_set.get_all_label_ids():
        if input(f"Label {label_id} already exists. Overwrite? (y/N): ").lower() != "y":
            exit("Aborted.")

    def generate_label_worker_item(review_id: str, review: dict):
        def task():
            nonlocal count
            usageOptions, metadata = generate_label(
                review,
                prompt,
                model=args.model,
                temperature=args.temperature,
                logprobs=args.logprobs,
                prompt_id=prompt_id,
            )
            review_set.add_label(review_id, label_id, usageOptions, metadata)

            if count % intermediate_save_size == 0 and count != 0:
                review_set.save()
                print(
                    f"{count}/{num_reviews} reviews processed. Intermediate results saved. 🚀"
                )

            count += 1

        return task

    num_reviews = len(review_set)
    intermediate_save_size = max(1, int(num_reviews * (args.save / 100)))

    labelling_queue = asyncio.Queue()

    for review_id, review in review_set.reviews.items():
        labelling_queue.put_nowait(generate_label_worker_item(review_id, review))

    worker = Worker(labelling_queue, n=20)
    print(
        f"Setup queue with {worker.queue.qsize()} reviews. Starting {worker.n} workers in parallel 🚀"
    )

    await worker.run()

    print("All reviews processed. 🌪️🌪️🌪️")
    review_set.save()


if __name__ == "__main__":
    asyncio.run(main())
