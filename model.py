import os
import glob
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import simpson

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

import joblib

mne.set_log_level("WARNING")

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
DATA_ROOT  = r"C:\Users\Dell\Desktop\EEG_detection\data\ASZED"
OUTPUT_DIR = r"./data/processed_aszed"
MODEL_PATH = r"./data/processed_aszed/eeg_model.pkl"   # <-- saved model path

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# CHANNEL STANDARDIZATION
# ─────────────────────────────────────────────
TARGET_CHANNELS = [
    "Fp1","Fp2","F7","F3","Fz","F4","F8",
    "T3","C3","Cz","C4","T4",
    "T5","P3","Pz","P4","T6","O1","O2"
]

def standardize_channels(raw):
    raw.rename_channels(lambda x: x.strip())
    raw.pick_types(eeg=True)

    missing = [ch for ch in TARGET_CHANNELS if ch not in raw.ch_names]
    if missing:
        info = mne.create_info(missing, raw.info["sfreq"], ch_types="eeg")
        zeros = np.zeros((len(missing), raw.n_times))
        raw_missing = mne.io.RawArray(zeros, info)
        raw.add_channels([raw_missing], force_update_info=True)

    raw.pick_channels(TARGET_CHANNELS)
    raw.set_montage("standard_1020", on_missing="ignore")
    return raw

# ─────────────────────────────────────────────
# LOAD EEG  (FIX: IIR filters for short signals)
# ─────────────────────────────────────────────
def load_eeg(filepath):
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    raw = standardize_channels(raw)
    raw.resample(250)
    raw.notch_filter(50, method="iir")   # FIX: was causing filter-length warnings
    raw.filter(0.5, 45, method="iir")    # FIX: same
    raw.set_eeg_reference("average")
    return raw

# ─────────────────────────────────────────────
# EPOCHS
# ─────────────────────────────────────────────
def create_epochs(raw, duration=1.0, overlap=0.5):
    events = mne.make_fixed_length_events(raw, duration=duration, overlap=overlap)
    epochs = mne.Epochs(
        raw, events, tmin=0, tmax=duration,
        baseline=None, preload=True, verbose=False
    )
    return epochs

# ─────────────────────────────────────────────
# FEATURE EXTRACTION (PSD bands)
# ─────────────────────────────────────────────
BANDS = {
    "delta": (0.5, 4),
    "theta": (4,   8),
    "alpha": (8,  13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}

def extract_features(epochs):
    psd  = epochs.compute_psd(method="welch", fmin=0.5, fmax=45)
    psds = psd.get_data()
    freqs = psd.freqs

    n_epochs, n_channels, _ = psds.shape
    features = []

    for ep in range(n_epochs):
        ep_feat = []
        for ch in range(n_channels):
            signal      = psds[ep, ch]
            total_power = simpson(signal, x=freqs)

            for fmin, fmax in BANDS.values():
                mask       = (freqs >= fmin) & (freqs <= fmax)
                band_power = simpson(signal[mask], x=freqs[mask])
                rel_power  = band_power / (total_power + 1e-10)
                ep_feat.extend([band_power, rel_power])

        features.append(ep_feat)

    return np.array(features).mean(axis=0)

# ─────────────────────────────────────────────
# FIND EDF FILES
# ─────────────────────────────────────────────
def find_edf_files(root):
    files = []
    for path, _, f in os.walk(root):
        for file in f:
            if file.lower().endswith(".edf"):
                files.append(os.path.join(path, file))
    return files

edf_files = find_edf_files(DATA_ROOT)
print("Total EDF files:", len(edf_files))

# ─────────────────────────────────────────────
# *** STEP 1: INSPECT YOUR PATHS FIRST ***
# Run this block, look at the printed parts,
# then update get_label() below accordingly.
# ─────────────────────────────────────────────
print("\n--- Sample path breakdown ---")
for f in edf_files[:5]:
    parts = f.replace("\\", "/").split("/")
    for i, p in enumerate(parts):
        print(f"  [{i}] {p}")
    print()

# ─────────────────────────────────────────────
# LABEL FUNCTION  (FIX: was returning 1 for everything)
# *** Update the index/logic after inspecting paths above ***
# ─────────────────────────────────────────────
def get_label(filepath):
    parts = filepath.replace("\\", "/").split("/")

    # Common ASZED convention: folder just before the phase file
    # encodes session number; map to binary label.
    # ADJUST the index and mapping to match YOUR folder structure.
    session_folder = parts[-2]   # e.g. "1" or "2"

    label_map = {"1": 1, "2": 0}  # <-- update this mapping as needed
    return label_map.get(session_folder, -1)   # -1 = unknown (will be skipped)

# ─────────────────────────────────────────────
# PROCESS DATASET
# ─────────────────────────────────────────────
X_list, y_list = [], []

for filepath in tqdm(edf_files):
    label = get_label(filepath)
    if label == -1:
        continue   # skip files whose label couldn't be determined

    try:
        raw    = load_eeg(filepath)
        epochs = create_epochs(raw)
        feat   = extract_features(epochs)
        X_list.append(feat)
        y_list.append(label)
    except Exception as e:
        print(f"Skipping {filepath}: {e}")

X = np.array(X_list)
y = np.array(y_list)

# ─────────────────────────────────────────────
# VERIFY LABELS BEFORE TRAINING
# ─────────────────────────────────────────────
print("\nUnique labels:", np.unique(y))
print("Class distribution:\n", pd.Series(y).value_counts())

if len(np.unique(y)) < 2:
    raise ValueError(
        "Only one class found! Fix get_label() before training. "
        "Check the path breakdown printed above."
    )

# ─────────────────────────────────────────────
# TRAIN / TEST SPLIT + SCALING
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# TRAIN MODEL
# ─────────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)

# ─────────────────────────────────────────────
# CROSS-VALIDATION
# ─────────────────────────────────────────────
cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=cv)
print("\nCV Accuracy:", scores)
print("Mean:", scores.mean())

# ─────────────────────────────────────────────
# EVALUATE  (FIX: guard against single-class predict_proba crash)
# ─────────────────────────────────────────────
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))

if len(model.classes_) == 2:
    y_prob = model.predict_proba(X_test)[:, 1]
    print("ROC AUC:", roc_auc_score(y_test, y_prob))
else:
    print("WARNING: Only one class seen during training — ROC AUC skipped.")

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ─────────────────────────────────────────────
# SAVE MODEL + SCALER AS .PKL  (joblib)
# ─────────────────────────────────────────────
payload = {
    "model":   model,
    "scaler":  scaler,
    "classes": model.classes_,
    "feature_bands": list(BANDS.keys()),
    "target_channels": TARGET_CHANNELS,
}

joblib.dump(payload, MODEL_PATH)
print(f"\nModel saved to: {MODEL_PATH}")

# ─────────────────────────────────────────────
# HOW TO RELOAD AND USE LATER
# ─────────────────────────────────────────────
# payload  = joblib.load("./data/processed_aszed/eeg_model.pkl")
# model    = payload["model"]
# scaler   = payload["scaler"]
#
# new_feat = extract_features(create_epochs(load_eeg("new_file.edf")))
# new_feat = scaler.transform([new_feat])
# pred     = model.predict(new_feat)
# prob     = model.predict_proba(new_feat)[:, 1]
# print("Prediction:", pred, "  Probability:", prob)