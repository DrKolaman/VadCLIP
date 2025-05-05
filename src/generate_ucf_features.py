import os
import cv2
import torch
import clip
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
    filename="feature_extraction.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

# Load CLIP model
try:
    model, preprocess = clip.load("ViT-B/32", device=device)
    logging.info("CLIP model loaded successfully")
except Exception as e:
    logging.error(f"Failed to load CLIP model: {e}")
    raise

# Input paths
video_base_dir = "C:/Users/kolaman/Downloads/UCF_Crimes/Videos"
csv_file = "C:/Users/kolaman/PycharmProjects/vision_seminar/20_המרצה_VadCLIP/VadCLIP/list/ucf_CLIP_rgbtest_bad.csv"
output_base_dir = "UCFClipFeatures_new"

# Number of threads
num_threads = 16

# Read the CSV file
try:
    df = pd.read_csv(csv_file)
    logging.info(f"CSV file loaded with {len(df)} entries")
except Exception as e:
    logging.error(f"Failed to read CSV file: {e}")
    raise

# Ensure output directory exists
os.makedirs(output_base_dir, exist_ok=True)

def extract_clip_features(video_path):
    """
    Extract CLIP features from all frames in a video.
    Returns a numpy array of features or None if extraction fails.
    """
    try:
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Could not open video: {video_path}")
            return None

        # Get total frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            logging.error(f"Video {video_path} has no frames")
            cap.release()
            return None

        # Initialize features list
        features = []

        # Process each frame with a progress bar
        for frame_idx in tqdm(range(total_frames), desc=f"Processing frames in {Path(video_path).name}", leave=False):
            ret, frame = cap.read()
            if not ret:
                logging.warning(f"Could not read frame {frame_idx} from {video_path}")
                continue

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image for CLIP preprocessing
            frame_pil = Image.fromarray(frame_rgb)

            # Preprocess and extract CLIP features
            try:
                image = preprocess(frame_pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_features = model.encode_image(image)
                features.append(image_features.cpu().numpy())
            except Exception as e:
                logging.error(f"Failed to process frame {frame_idx} in {video_path}: {e}")
                continue

        cap.release()

        if not features:
            logging.error(f"No features extracted from {video_path}")
            return None

        # Stack features (shape: [num_extracted_frames, feature_dim])
        features = np.concatenate(features, axis=0)
        if features.size == 0 or features.shape[0] == 0:
            logging.error(f"Empty feature array for {video_path}")
            return None

        logging.info(f"Extracted features for {video_path}: shape {features.shape}")
        return features
    except Exception as e:
        logging.error(f"Error processing {video_path}: {e}")
        cap.release() if 'cap' in locals() else None
        return None

def get_video_path(video_name, label):
    """
    Construct the video file path based on the video name and label.
    """
    try:
        if label == "Normal":
            for folder in ["Testing_Normal_Videos_Anomaly", "z_Normal_Videos_event"]:
                video_path = os.path.join(video_base_dir, folder, f"{video_name}.mp4")
                if os.path.exists(video_path):
                    return video_path
        else:
            video_path = os.path.join(video_base_dir, label, f"{video_name}.mp4")
            if os.path.exists(video_path):
                return video_path
        logging.error(f"Video file not found for {video_name} (label: {label})")
        return None
    except Exception as e:
        logging.error(f"Error finding video path for {video_name}: {e}")
        return None

def process_video(row):
    """
    Process a single video: extract features and save to .npy file.
    Designed to be run in a thread.
    """
    try:
        output_path = row["path"]
        label = row["label"]
        video_name = Path(output_path).stem.replace("__5", "")

        # Get video path
        video_path = get_video_path(video_name, label)
        if not video_path:
            return f"Failed: {video_name} (video not found)"

        # Check if output file already exists and is non-empty
        output_file = os.path.join(output_base_dir, output_path.lstrip("/"))
        if os.path.exists(output_file):
            try:
                existing_features = np.load(output_file)
                if existing_features.size > 10:
                    logging.info(f"Skipping {video_name}: non-empty .npy file exists")
                    return f"Skipped: {output_file} (already exists)"
            except:
                logging.warning(f"Existing .npy file {output_file} is corrupted, will overwrite")

        # Extract CLIP features
        features = extract_clip_features(video_path)
        if features is None or features.size == 0:
            return f"Failed: {video_name} (no features extracted)"

        # Create output directory for the label (thread-safe)
        output_label_dir = os.path.join(output_base_dir, label)
        os.makedirs(output_label_dir, exist_ok=True)

        # Save features as .npy file
        np.save(output_file, features)
        logging.info(f"Saved features to {output_file}: shape {features.shape}")
        return f"Saved: {output_file}"
    except Exception as e:
        logging.error(f"Error processing {video_name}: {e}")
        return f"Failed: {video_name}"

# Process videos in parallel using ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = [executor.submit(process_video, row) for index, row in df.iterrows()]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing videos"):
        result = future.result()
        print(result)

logging.info("Feature extraction completed.")
print("Feature extraction completed. Check feature_extraction.log for details.")