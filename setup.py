import os
import sys
import subprocess
from pathlib import Path
from abc import abstractmethod

BASE_DIR = os.path.dirname(__file__)
cwd = os.getcwd()

class PackageManger:
    @abstractmethod
    def get_installed_packages():
        pass
    
    @abstractmethod
    def install_requirements():
        pass

class Pip(PackageManger):
    def __init__(self, cwd=os.getcwd()) -> None:
        super().__init__()
        self.cwd = cwd
        print("Using pip to install dependencies")
    def get_installed_packages(self) -> set:
        # process output with an API in the subprocess module:
        reqs = subprocess.check_output([sys.executable, '-m', 'pip','freeze'], cwd=self.cwd)
        return set([r.decode().split('==')[0] for r in reqs.split()])
    def install_all_requirements(self, ):
        self.install_requirements()
        self.install_local_requirements()

    def install_requirements(self, requirements_file='requirements.txt'):
        # implement pip as a subprocess:
        if os.path.isfile(requirements_file):
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_file], cwd=self.cwd)
        else:
            print(requirements_file, "not found.")

    def install_local_requirements(self, requirements_file='local_requirements.txt'):
        with open(Path(self.cwd, requirements_file), 'r') as file:
            required_packages = file.readlines()
        
        print(required_packages)
        for package in required_packages:
            installer = Pip()
            installer.install_requirements(Path(BASE_DIR, package, 'requirements.txt'))


    
print("Setting PYTHONPATH to allow imports from the project root")
os.system(f'export PYTHONPATH="${{PYTHONPATH}}:{BASE_DIR}"')

package_manager = None
if os.path.isfile(Path(cwd, 'requirements.txt')):
    package_manager = Pip()

if package_manager is None:
    print("No dependencies found")
else:
    already_installed_packages = package_manager.get_installed_packages()
    package_manager.install_all_requirements()
    all_packages = package_manager.get_installed_packages()
    
    new_packages = list(all_packages.difference(already_installed_packages))
    if len(new_packages) > 0:
        print("Installed the following new packages:", list(new_packages))