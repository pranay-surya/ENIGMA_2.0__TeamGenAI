# prepare_aszed_training.py
import os
import numpy as np
from tqdm import tqdm
import glob
from ml_predictor import extract_features
from eeg_processor import process_aszed_subject_folder

DATA_ROOT = "./data/ASZED-153"   # adjust after unzip
OUTPUT_DIR = "./data/processed_aszed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    # Recursive search for subject-like folders (handles nested structure)
    potential_folders = []
    for root, dirs, _ in os.walk(DATA_ROOT):
        if any(re.search(r'sub|pat|con|sz|hc|ctrl', d.lower()) for d in dirs):
            potential_folders.extend([os.path.join(root, d) for d in dirs])
        # or look for folders containing .edf
        edfs = glob.glob(os.path.join(root, "*.edf"))
        if edfs and len(glob.glob(os.path.join(root, "*"))) < 20:  # likely subject/session folder
            potential_folders.append(root)

    subject_folders = list(set(potential_folders))  # dedup
    print(f"Found {len(subject_folders)} potential subject/session folders")

    X_list, y_list = [], []
    skipped = 0

    for folder in tqdm(subject_folders):
        epochs, label = process_aszed_subject_folder(folder)
        if epochs is None or label == -1:
            skipped += 1
            continue

        try:
            features, _ = extract_features(epochs)
            X_list.append(features)
            y_list.append(label)
        except Exception as e:
            print(f"Feature extraction failed in {folder}: {e}")
            skipped += 1

    if not X_list:
        print("No valid subjects processed.")
        return

    X = np.array(X_list)
    y = np.array(y_list)

    print(f"Processed {len(X)} subjects | SZ: {sum(y)} | Controls: {len(y)-sum(y)} | Skipped: {skipped}")

    np.save(os.path.join(OUTPUT_DIR, "X_aszed.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y_aszed.npy"), y)
    print(f"Saved to {OUTPUT_DIR}")