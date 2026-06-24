import os
import shutil
import argparse

from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--method", "-m", required=True, type=str, help="Method name")
parser.add_argument("--dataset", "-d", required=True, type=str, help="Dataset name")
args = parser.parse_args()

method = args.method
dataset = args.dataset

root = os.path.join("outputs", method, dataset)
collected_videos_dir = os.path.join("videos", method, dataset)

if not os.path.exists(collected_videos_dir):
    os.makedirs(collected_videos_dir)

for scene in tqdm(os.listdir(root), desc="Collecting videos"):
    if os.path.isdir(os.path.join(root, scene)):
        video_path = os.path.join(root, scene, "video", "render_traj.mp4")
        assert os.path.exists(video_path), f"Video not found: {video_path}"
        
        new_video_path = os.path.join(collected_videos_dir, f"{scene}.mp4")
        shutil.copy(video_path, new_video_path)

print(f"Collected videos for method '{method}' and dataset '{dataset}' in '{collected_videos_dir}'")