import os
import wget


def fetch_files_from_txt(url_file: str, target_dir: str):
    with open(url_file, "r") as urls:
        target_dir = os.path.expanduser(target_dir)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            
        file_paths = []
        for url in [url.strip() for url in urls.readlines()]:
            file_name = os.path.split(url)[1]
            file_path = os.path.join(target_dir, file_name)
            file_paths.add(file_path)
            
            if os.path.exists(file_path):
                print(f"Skipping {file_name}... file already exists")
                continue
            
            print(f"Downloading {url}...")
            wget.download(url, target_dir)
            print()
            
        return file_paths
