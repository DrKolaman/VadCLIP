import os
import cv2
import torch
import clip
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model
model, preprocess = clip.load("ViT-B/32", device=device)

# Input paths
video_base_dir = "C:/Users/kolaman/Downloads/UCF_Crimes/Videos"  # Base directory containing video folders
csv_file = "C:/Users/kolaman/PycharmProjects/vision_seminar/20_המרצה_VadCLIP/VadCLIP/list/ucf_CLIP_rgbtest_bad.csv"   # Path to the testing CSV file
output_base_dir = "UCFClipFeatures_new"  # Base directory for output .npy files

# Read the CSV file
df = pd.read_csv(csv_file)

# Ensure output directory exists
os.makedirs(output_base_dir, exist_ok=True)

def extract_clip_features(video_path, num_frames=5):
    """
    Extract CLIP features from a video by sampling num_frames frames.
    Returns a numpy array of features.
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    # Get total frames and calculate frame indices to sample
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames or num_frames==0:
        print(f"Warning: Video {video_path} has fewer than {num_frames} frames")
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    # Extract frames
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB and to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frames.append(frame_pil)
        else:
            print(f"Warning: Could not read frame {idx} from {video_path}")

    cap.release()

    if not frames:
        print(f"Error: No frames extracted from {video_path}")
        return None

    # Preprocess and extract CLIP features
    features = []
    for frame in frames:
        image = preprocess(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
        features.append(image_features.cpu().numpy())

    # Stack features (shape: [num_frames, feature_dim])
    features = np.concatenate(features, axis=0)
    return features

def get_video_path(video_name, label):
    """
    Construct the video file path based on the video name and label.
    """
    if label == "Normal":
        # Normal videos are in Testing_Normal_Videos_Anomaly or z_Normal_Videos_event
        for folder in ["Testing_Normal_Videos_Anomaly", "z_Normal_Videos_event"]:
            video_path = os.path.join(video_base_dir, folder, f"{video_name}.mp4")
            if os.path.exists(video_path):
                return video_path
    else:
        # Crime videos are in their respective category folders (e.g., Assault, Arson)
        video_path = os.path.join(video_base_dir, label, f"{video_name}.mp4")
        if os.path.exists(video_path):
            return video_path
    return None

# Process each entry in the CSV
for index, row in df.iterrows():
    output_path = row["path"]  # e.g., /UCFClipFeatures/Abuse/Abuse028_x264__5.npy
    label = row["label"]       # e.g., Abuse, Normal, etc.

    # Extract video name from output path
    video_name = Path(output_path).stem.replace("__5", "")  # e.g., Abuse028_x264

    # Get video path
    video_path = get_video_path(video_name, label)
    if not video_path:
        print(f"Error: Video file for {video_name} (label: {label}) not found")
        continue

    # Extract CLIP features
    features = extract_clip_features(video_path, num_frames=0)
    if features is None:
        continue

    # Create output directory for the label
    output_label_dir = os.path.join(output_base_dir, label)
    os.makedirs(output_label_dir, exist_ok=True)

    # Save features as .npy file
    output_file = os.path.join(output_base_dir, output_path.lstrip("/"))
    np.save(output_file, features)
    print(f"Saved features to {output_file}")

print("Feature extraction completed.")