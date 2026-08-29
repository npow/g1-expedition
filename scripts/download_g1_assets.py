import urllib.request
import json
import os
from concurrent.futures import ThreadPoolExecutor

BASE_API = "https://api.github.com/repos/google-deepmind/mujoco_menagerie/contents/unitree_g1"
RAW_BASE = "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/unitree_g1"
LOCAL_DIR = "/home/npow/code/himalaya/assets/unitree_g1"

def fetch_dir(api_url, local_path):
    os.makedirs(local_path, exist_ok=True)
    req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        items = json.loads(resp.read().decode("utf-8"))
    
    files_to_download = []
    for item in items:
        if item["type"] == "file":
            files_to_download.append((item["download_url"], os.path.join(local_path, item["name"])))
        elif item["type"] == "dir":
            sub_files = fetch_dir(item["url"], os.path.join(local_path, item["name"]))
            files_to_download.extend(sub_files)
    return files_to_download

def download_file(url_path_pair):
    url, target_path = url_path_pair
    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        return
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp, open(target_path, "wb") as f:
            f.write(resp.read())
        print(f"Downloaded: {os.path.basename(target_path)} ({os.path.getsize(target_path)} bytes)")
    except Exception as e:
        print(f"Failed {url}: {e}")

def main():
    print("Listing files from MuJoCo Menagerie unitree_g1...")
    files = fetch_dir(BASE_API, LOCAL_DIR)
    print(f"Found {len(files)} files to download. Starting multi-threaded download...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        list(executor.map(download_file, files))
    print("All Unitree G1 assets downloaded successfully!")

if __name__ == "__main__":
    main()
