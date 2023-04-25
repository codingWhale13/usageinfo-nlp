# %%
from helpers.aws.ec2 import EC2Launcher

# %%
import argparse


def parse_args():
    arg_parser = argparse.ArgumentParser(
        description="Launch and manage your ec2 instance"
    )
    arg_parser.add_argument("--stop", "-s", action="store_true", help="Stop instance")

    arg_parser.add_argument(
        "--terminate", "-t", action="store_true", help="Terminate instance"
    )
    return arg_parser.parse_args()


args = parse_args()

launcher = EC2Launcher()

if args.terminate:
    launcher.terminate_instance()
elif args.stop:
    launcher.stop_instance()
else:
    launcher.run()
