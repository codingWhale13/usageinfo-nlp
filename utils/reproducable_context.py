import argparse
import random
import yaml
from datetime import datetime
import os
import hashlib
import uuid


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


PRINT_COLOR = "\033[94m"
COLOR_END_CHARACTER = "\033[0m"


class ReproducableContext:
    def __init__(
        self, run_name, verbose=True, export_context=True, append_timestamp=True
    ) -> None:
        self.run_name = run_name
        self.export_context = export_context
        self.append_timestamp = append_timestamp
        self.verbose = verbose
        self.output_files = []
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.id = str(uuid.uuid4())[:8]
        self.additional_info = {}

    def create_argument_parser(self, *args, **kwargs) -> None:
        self.arg_parser = argparse.ArgumentParser(*args, **kwargs)
        self.add_argument("--seed", help="Random seed to use")

    def add_argument(self, *args, **kwargs):
        return self.arg_parser.add_argument(*args, **kwargs)

    def __update_args(self, key, value):
        namespace_dict = vars(self.args)
        namespace_dict[key] = value

    def __set_seed(self):
        if self.args.seed is None:
            self.__update_args("seed", random.randrange(1000_000_000))
        elif type(self.args.seed) != int:
            self.__update_args("seed", int(self.args.seed))
        random.seed(self.args.seed)
        self.print(f"Using seed: {self.args.seed}")

    def parse_args(self):
        self.args = self.arg_parser.parse_args()
        self.__set_seed()
        return self.args

    def format_help(self, *args, **kwargs):
        return self.arg_parser.format_help(*args, **kwargs)

    def output_dir(self):
        output_dir = f"{self.run_name}"
        if self.append_timestamp:
            output_dir += f"-{self.timestamp}"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        return output_dir

    def set_additional_info(self, key, object) -> dict:
        self.additional_info[key] = object
        return self.additional_info

    def output_file(self, file_name):
        output_file = f"{self.output_dir()}/{file_name}"
        if output_file not in self.output_files:
            self.output_files.append(output_file)
        return output_file

    def export(self):
        if not hasattr(self, "args"):
            self.args = self.arg_parser.parse_args()
        data = {
            "id": self.id,
            "args": vars(self.args),
            "info": self.additional_info,
            "files": {"input": [], "output": []},
            "cwd": os.getcwd(),
            "timestamp": self.timestamp,
        }

        for arg, arg_value in data["args"].items():
            if isinstance(arg_value, str) and os.path.isfile(arg_value):
                data["files"]["input"].append(
                    {"arg": arg, "name": arg_value, "md5": md5(arg_value)}
                )

        for output_file in self.output_files:
            if isinstance(output_file, str) and os.path.isfile(output_file):
                data["files"]["output"].append(
                    {"name": output_file, "md5": md5(output_file)}
                )

        output_dir = self.output_dir()
        output_file = f"{output_dir}/context-{self.id}.yml"
        with open(output_file, "w") as outfile:
            yaml.dump(data, outfile)

        self.print(f"Saving context information to: {output_file}")

    def print(self, message):
        if self.verbose:
            print(f"{PRINT_COLOR}CONTEXT | {message}{COLOR_END_CHARACTER}")

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if exception_type is None and self.export_context:
            self.export()
