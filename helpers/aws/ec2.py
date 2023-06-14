# %%
import boto3
from botocore.exceptions import ClientError
import yaml
import os.path
import time


class EC2InstanceNotFound(Exception):
    pass


class LogLevel:
    DEBUG = 1
    INFO = 2
    WARNING = 3
    ERROR = 4
    FORCE_LOGGING = 5


"""
Possible instance states: 'pending'|'running'|'shutting-down'|'terminated'|'stopping'|'stopped'
custom additional state: 'not-found'
"""


class EC2State:
    PENDING = "pending"
    RUNNING = "running"
    SHUTTING_DOWN = "shutting-down"
    TERMINATED = "terminated"
    STOPPING = "stopping"
    STOPPED = "stopped"
    NOT_FOUND = "not-found"


IMAGE_IDS = {"anaconda3": "ami-0a925c3db0ea64491", "ubuntu": "ami-04e601abe3e1a910f", "ubuntu-vpn-miniconda": "ami-0d2a6730eb086e476", "ubuntu-vpn-miniconda-2": "ami-05878ba6e31e2efc9"}
DEFAULT_IMAGE = "ubuntu-vpn-miniconda-2"

EC2_CONFIG_PATH = f"{os.path.dirname(__file__)}/ec2_config.yml"


class EC2Launcher:
    def log(self, message, level=LogLevel.INFO):
        if level >= self.config["log_level"]:
            print(message)

    def __init__(self) -> None:
        self.ec2 = boto3.client("ec2", region_name="eu-central-1")
        self.username = self.__username()
        self.config = self.load_config()
        self.generate_key_pair()

    def load_config(self, config_path=None) -> dict:
        if config_path is None:
            config_path = EC2_CONFIG_PATH
        if os.path.isfile(config_path):
            print("Loaded config")
            with open(config_path, "r") as file:
                return yaml.safe_load(file)
        else:
            print("Creating new config")
            return self.__default_config()

    def __default_config(self) -> dict:
        return {
            "InstanceId": None,
            "InstanceType": "t2.micro",
            "log_level": LogLevel.INFO,
        }

    def save_config(self):
        with open(EC2_CONFIG_PATH, "w") as file:
            yaml.dump(self.config, file)

    def instance_state(self) -> str:
        try:
            return self.describe_instance_status()["InstanceState"]["Name"]
        except EC2InstanceNotFound:
            return EC2State.NOT_FOUND

    def validate_instance_type(self):
        state = self.instance_state()
        if state != EC2State.NOT_FOUND and state != EC2State.TERMINATED:
            instance_type = self.instance_type()
            if instance_type != self.config["InstanceType"]:
                self.log(f"Found instance with old type: {instance_type}")
                self.terminate_instance()

    def run(self):
        self.validate_instance_type()

        state = self.instance_state()
        if state == EC2State.NOT_FOUND or state == EC2State.TERMINATED:
            self.launch_instance()
        elif state == EC2State.STOPPED:
            self.start_instance()
        elif state == EC2State.RUNNING:
            self.log(f"Instance {self.config['InstanceId']} already running")
        else:
            raise ValueError(f"Instance in unhandled state: {state}")

        self.wait_until_instance_is_reachable()
        self.log(
            f"Connect to instance with the following command: \n\n{self.ssh_connection_string()}",
            LogLevel.FORCE_LOGGING,
        )

    def start_instance(self):
        _ = self.ec2.start_instances(InstanceIds=[self.config["InstanceId"]])
        self.log(f"Starting instance {self.config['InstanceId']}")

    def __username(self) -> str:
        username = None
        try:
            iam = boto3.client("iam")
            username = iam.get_user()["User"]["UserName"]
        except ClientError as e:
            # The username is actually specified in the Access Denied message...
            username = e.response["Error"]["Message"].split(" ")[-1]
        return username

    def key_pair_name(self):
        return f"ec2_ssh_key_pair_{self.__username()}"

    def ssh_username(self):
        return "ubuntu" #"ec2-user" is the username for the anaconda image

    def generate_key_pair(self):
        info = self.ec2.describe_key_pairs()
        is_a_valid_key_pair_available = True
        if len(info["KeyPairs"]) == 0:
            is_a_valid_key_pair_available = False
        else:
            is_key_pair_attached_to_ec2 = self.key_pair_name() in [
                key_pair["KeyName"] for key_pair in info["KeyPairs"]
            ]
            if not os.path.isfile(self.identity_file_path()):
                is_a_valid_key_pair_available = False
                if is_key_pair_attached_to_ec2:
                    self.log(
                        f"Old key pair still attached to ec2. Deleting old key: {self.key_pair_name()}"
                    )
                    self.ec2.delete_key_pair(KeyName=self.key_pair_name())
            elif is_key_pair_attached_to_ec2 is False:
                is_a_valid_key_pair_available = False

        if is_a_valid_key_pair_available is False:
            self.log("No keypairs available. Generating a new key pair")
            response = self.ec2.create_key_pair(KeyName=self.key_pair_name())
            with open(self.identity_file_path(), "w") as file:
                file.write(response["KeyMaterial"])
            # ec2 ssh keys must have this type of permission (only read for current user)
            os.chmod(self.identity_file_path(), 0o400)
            self.log(f"Written new keypair to {self.key_pair_name()}")
        else:
            self.log(f"Found existing key: {self.key_pair_name()}")

    def describe_instance(self):
        return self.ec2.describe_instances(InstanceIds=[self.config["InstanceId"]])[
            "Reservations"
        ][0]["Instances"][0]

    def public_ip_address(self):
        return self.describe_instance()["PublicIpAddress"]

    def describe_instance_status(self):
        if self.config["InstanceId"] is None:
            raise EC2InstanceNotFound()
        response = self.ec2.describe_instance_status(
            InstanceIds=[self.config["InstanceId"]], IncludeAllInstances=True
        )
        self.log(response, LogLevel.DEBUG)
        if len(response["InstanceStatuses"]) == 0:
            raise EC2InstanceNotFound()
        return response["InstanceStatuses"][0]

    def launch_instance(self):
        instance = self.ec2.run_instances(
            InstanceType=self.config["InstanceType"],  # "t2.micro",
            ImageId=IMAGE_IDS[DEFAULT_IMAGE],
            MaxCount=1,
            MinCount=1,
            DryRun=False,
            KeyName=self.key_pair_name(),
        )

        self.config["InstanceId"] = instance["Instances"][0]["InstanceId"]
        self.save_config()
        self.log(
            f"Instance {self.config['InstanceId']} launched. Waiting until it is ready for a ssh connection. This might take 1-3 minutes..."
        )
        return instance

    def stop_instance(self):
        self.log(
            f"Stopping instance {self.config['InstanceId']}. Warning: We still being charged for storing the file system"
        )
        response = self.ec2.stop_instances(InstanceIds=[self.config["InstanceId"]])
        self.log(response, LogLevel.DEBUG)
        waiter = self.ec2.get_waiter("instance_stopped")
        self.log("Waiting until instance is stopped")
        waiter.wait(InstanceIds=[self.config["InstanceId"]], WaiterConfig={"Delay": 5})
        self.log(f"Instance successfully stopped")

    def terminate_instance(self):
        self.log(f"Terminating instance {self.config['InstanceId']}")
        response = self.ec2.terminate_instances(InstanceIds=[self.config["InstanceId"]])
        self.log(response, LogLevel.DEBUG)
        waiter = self.ec2.get_waiter("instance_terminated")
        waiter.wait(InstanceIds=[self.config["InstanceId"]], WaiterConfig={"Delay": 3})
        self.log(f"Successfully terminated instance {self.config['InstanceId']}")

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if exception_type is None:
            self.stop_instance()

    def is_instance_existing(self) -> bool:
        return False

    def is_instance_stopped(self) -> bool:
        return False

    def identity_file_path(self) -> str:
        return f"{os.path.dirname(__file__)}/{self.key_pair_name()}"

    def is_instance_reachable(self):
        try:
            status = self.describe_instance_status()
        except EC2InstanceNotFound:
            return False

        if (
            status["InstanceStatus"]["Status"] == "ok"
            and status["SystemStatus"]["Status"] == "ok"
        ):
            return True
        else:
            return False

    def host(self) -> str:
        return self.public_ip_address()

    def instance_type(self) -> str:
        return self.describe_instance()["InstanceType"]

    def ssh_connection_string(self) -> str:
        return f"ssh -i {self.identity_file_path()} {self.ssh_username()}@{self.host()}"

    def wait_until_instance_is_reachable(self):
        MAX_TIMEOUT = 300
        total_wait_time = 0
        RETRY_PAUSE = 0.8
        while self.is_instance_reachable() is False:
            time.sleep(RETRY_PAUSE)
            total_wait_time += RETRY_PAUSE
            if total_wait_time > MAX_TIMEOUT:
                raise Exception("Timeout. Instance still not reachable")
        self.log(
            f"Instance is up and now reachable via ssh.\nInstance startup time: {total_wait_time} seconds"
        )
