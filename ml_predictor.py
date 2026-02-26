# ml_predictor.py
"""
Improved schizophrenia risk predictor:
- More clinically relevant features (relative power, ratios, asymmetry)
- Better feature naming
- Ready for real dataset retraining
- Returns feature importances
"""

import numpy as np
import pickle
import os
from scipy.integrate import simpson
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

MODEL_PATH = "models/schizophrenia_rf_model.pkl"
SCALER_PATH = "models/schizophrenia_scaler.pkl"
FEATURE_NAMES_PATH = "models/feature_names.pkl"

# ─── Frequency Bands ────────────────────────────────────────────────────────
BANDS = {
    "delta":  (0.5,  4.0),
    "theta":  (4.0,  8.0),
    "alpha":  (8.0, 13.0),
    "beta":   (13.0,30.0),
    "gamma":  (30.0,45.0),
}

ASYMMETRY_PAIRS = [
    ("F3", "F4"), ("F7", "F8"), ("C3", "C4"), ("P3", "P4"), ("O1", "O2")
]


def extract_features(epochs):
    """
    Extract extended PSD-based features:
      - Absolute band power
      - Relative band power
      - Selected band ratios (theta/alpha, delta/alpha)
      - Frontal/posterior asymmetry for selected bands
    """
    psd = epochs.compute_psd(method="welch", fmin=0.5, fmax=45.0, verbose=False)
    psds = psd.get_data()          # (n_epochs, n_channels, n_freqs)
    freqs = psd.freqs

    ch_names = epochs.ch_names
    n_epochs, n_chans, n_freqs = psds.shape

    features = []
    feature_names = []

    for ep in range(n_epochs):
        ep_features = []
        total_power_per_ch = np.zeros(n_chans)

        # 1. Absolute & relative band power
        for ch_idx, ch in enumerate(ch_names):
            psd_ch = psds[ep, ch_idx]
            total_power = simpson(psd_ch, freqs)
            total_power_per_ch[ch_idx] = total_power if total_power > 0 else 1e-12

            for band, (fmin, fmax) in BANDS.items():
                mask = (freqs >= fmin) & (freqs <= fmax)
                if not np.any(mask):
                    band_pow = 0.0
                else:
                    band_pow = simpson(psd_ch[mask], freqs[mask])

                rel_pow = band_pow / total_power_per_ch[ch_idx]

                ep_features.extend([band_pow, rel_pow])
                if ep == 0:
                    feature_names.extend([f"{ch}_{band}_abs", f"{ch}_{band}_rel"])

        # 2. Band ratios (per channel) — always emit both values to keep vector length stable
        for ch_idx, ch in enumerate(ch_names):
            theta_abs = ep_features[ch_idx * 10 + 2]
            alpha_abs = ep_features[ch_idx * 10 + 4]
            delta_abs = ep_features[ch_idx * 10]

            if alpha_abs > 1e-12:
                theta_alpha = theta_abs / alpha_abs
                delta_alpha = delta_abs / alpha_abs
            else:
                theta_alpha = 0.0
                delta_alpha = 0.0

            ep_features.extend([theta_alpha, delta_alpha])
            if ep == 0:
                feature_names.extend([f"{ch}_theta_alpha_ratio", f"{ch}_delta_alpha_ratio"])

        # 3. Asymmetry (selected pairs & bands)
        for left_ch, right_ch in ASYMMETRY_PAIRS:
            if left_ch not in ch_names or right_ch not in ch_names:
                continue
            l_idx = ch_names.index(left_ch)
            r_idx = ch_names.index(right_ch)

            for band_idx, (band, _) in enumerate(BANDS.items()):
                l_pow = ep_features[l_idx * 10 + band_idx * 2]     # abs power offset
                r_pow = ep_features[r_idx * 10 + band_idx * 2]
                if (l_pow + r_pow) > 1e-12:
                    asym = (l_pow - r_pow) / (l_pow + r_pow)
                else:
                    asym = 0.0
                ep_features.append(asym)
                if ep == 0:
                    feature_names.append(f"asym_{left_ch}-{right_ch}_{band}")

        features.append(ep_features)

    X = np.array(features)              # (n_epochs, n_features)
    mean_features = X.mean(axis=0)      # patient-level representation
    return mean_features, feature_names


def load_or_train_model():
    if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH]):
        print("Model files missing. You should train on real data.")
        # For now we raise — in production you might want fallback or synthetic retrain
        raise FileNotFoundError("No trained model found. Train on real EEG data first.")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(FEATURE_NAMES_PATH, "rb") as f:
        feat_names = pickle.load(f)

    return model, scaler, feat_names


def predict_risk(epochs):
    print("[DEBUG] Starting predict_risk")
    features, feat_names = extract_features(epochs)
    print(f"[DEBUG] Extracted {len(features)} features, {len(feat_names)} names")

    try:
        model, scaler, saved_names = load_or_train_model()
        print(f"[DEBUG] Loaded model expecting {len(saved_names)} features")
    except Exception as e:
        print(f"[DEBUG] Model load failed: {e}")
        raise

    if len(features) != len(saved_names):
        print(f"[ERROR] Feature mismatch! Expected {len(saved_names)}, got {len(features)}")
        print("First 10 feat_names:", feat_names[:10])
        raise ValueError(f"Feature count mismatch. Expected {len(saved_names)}, got {len(features)}")

    # Scale and predict
    X = scaler.transform(features.reshape(1, -1))
    risk_prob  = model.predict_proba(X)[0, 1]          # P(schizophrenia)
    risk_score = float(risk_prob * 100.0)               # 0-100 %

    importances = getattr(model, "feature_importances_", None)

    print(f"[DEBUG] Risk score: {risk_score:.1f}%")
    return risk_score, features, feat_names, importances


def get_top_anomalies(features, feature_names, top_n=4):
    z = np.abs(features - np.mean(features)) / (np.std(features) + 1e-10)
    idx = np.argsort(z)[::-1][:top_n]
    return [feature_names[i] for i in idx]


def get_top_important_features(importances, feature_names, top_n=5):
    if importances is None:
        return []
    idx = np.argsort(importances)[::-1][:top_n]
    return [(feature_names[i], importances[i]) for i in idx]



# In ml_predictor.py ─ add this function

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

def train_on_aszed():
    X_path = "./data/processed_aszed/X_aszed.npy"
    y_path = "./data/processed_aszed/y_aszed.npy"

    if not (os.path.exists(X_path) and os.path.exists(y_path)):
        raise FileNotFoundError("Run prepare_aszed_training.py first")

    X = np.load(X_path)
    y = np.load(y_path)

    print(f"Training on {X.shape[0]} subjects | class balance: {np.mean(y):.3f}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    
    # Simple train/val split (subject-level!)
    X_tr, X_val, y_tr, y_val = train_test_split(X_scaled, y, test_size=0.20, stratify=y, random_state=42)
    clf.fit(X_tr, y_tr)

    # Quick evaluation
    pred_proba = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, pred_proba)
    acc = accuracy_score(y_val, pred_proba > 0.5)
    print(f"Validation AUC: {auc:.3f} | Accuracy: {acc:.3f}")

    # Save model and scaler
    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    # Save feature names — load them from the pre-saved .npy file produced by prepare_aszed_training.py
    names_npy = "./data/processed_aszed/feature_names.npy"
    if os.path.exists(names_npy):
        feat_names = list(np.load(names_npy, allow_pickle=True))
    else:
        # Fallback: derive names by running extract_features on a dummy 1-epoch signal
        from eeg_processor import generate_synthetic_eeg
        dummy_epochs = generate_synthetic_eeg(n_epochs=2)
        _, feat_names = extract_features(dummy_epochs)

    with open(FEATURE_NAMES_PATH, "wb") as f:
        pickle.dump(feat_names, f)

    print(f"Model retrained and saved using ASZED data. Feature count: {len(feat_names)}")

def create_placeholder_model():
    """Quick dummy model to unblock the UI – scores will be random until real training."""
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    import pickle
    import os

    # Match observed feature count (253 from your log)
    n_features = 253
    X_dummy = np.random.randn(200, n_features)
    y_dummy = np.random.randint(0, 2, 200)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_dummy)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y_dummy)

    os.makedirs("models", exist_ok=True)
    with open("models/schizophrenia_rf_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("models/schizophrenia_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("models/feature_names.pkl", "wb") as f:
        fake_names = [f"ch_band_{i}" for i in range(n_features)]
        pickle.dump(fake_names, f)

    print("Placeholder model saved. Restart uvicorn — demo should now show a risk score.")
