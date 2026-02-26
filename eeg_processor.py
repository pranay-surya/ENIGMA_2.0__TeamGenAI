import numpy as np
import mne
from mne import create_info, EpochsArray
import tempfile
import os

mne.set_log_level("WARNING")

EEG_CHANNELS_1020 = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6",
    "O1", "O2"
]
N_CHANNELS = len(EEG_CHANNELS_1020)
SFREQ = 256.0  # Hz


def load_eeg_file(filepath: str):
    """Load .edf or .fif EEG file using MNE-Python."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".edf":
        raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    elif ext == ".fif":
        raw = mne.io.read_raw_fif(filepath, preload=True, verbose=False)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .edf or .fif")
    return raw


def preprocess_and_epoch(raw, epoch_duration: float = 2.0):
    """Apply bandpass filter and segment into epochs."""
    raw.filter(l_freq=0.5, h_freq=45.0, method="iir", verbose=False)
    events = mne.make_fixed_length_events(raw, duration=epoch_duration)
    epochs = mne.Epochs(
        raw, events, tmin=0.0, tmax=epoch_duration,
        baseline=None, preload=True, verbose=False
    )
    return epochs


def generate_synthetic_eeg(condition: str = "schizophrenic", n_epochs: int = 30):
    """
    Generate realistic dummy EEG data for demo purposes.
    condition: 'healthy' or 'schizophrenic'
    Returns an MNE Epochs object.
    """
    rng = np.random.default_rng(42)
    epoch_samples = int(SFREQ * 2.0)  # 2-second epochs
    data = np.zeros((n_epochs, N_CHANNELS, epoch_samples))
    t = np.linspace(0, 2.0, epoch_samples, endpoint=False)

    # Band frequencies (Hz) and amplitudes vary by condition
    # Schizophrenia traits: elevated theta/delta, reduced alpha, abnormal gamma
    band_params = {
        "healthy": {
            "delta": (2.0, 0.5e-6),
            "theta": (6.0, 0.5e-6),
            "alpha": (10.0, 3.0e-6),   # strong alpha
            "beta":  (20.0, 1.0e-6),
            "gamma": (40.0, 0.3e-6),
        },
        "schizophrenic": {
            "delta": (2.0, 2.5e-6),    # elevated slow waves
            "theta": (6.0, 2.0e-6),    # elevated theta
            "alpha": (10.0, 0.8e-6),   # reduced alpha
            "beta":  (20.0, 1.2e-6),
            "gamma": (40.0, 1.8e-6),   # abnormal gamma bursts
        },
    }
    params = band_params.get(condition, band_params["schizophrenic"])

    for epoch_idx in range(n_epochs):
        for ch_idx in range(N_CHANNELS):
            signal = np.zeros(epoch_samples)
            for band, (freq, amp) in params.items():
                phase = rng.uniform(0, 2 * np.pi)
                signal += amp * np.sin(2 * np.pi * freq * t + phase)
            # Add pink-ish noise
            noise = rng.standard_normal(epoch_samples) * 0.3e-6
            data[epoch_idx, ch_idx] = signal + noise

    info = create_info(ch_names=EEG_CHANNELS_1020, sfreq=SFREQ, ch_types="eeg")
    info.set_montage("standard_1020", on_missing="ignore")

    events_array = np.column_stack([
        np.arange(n_epochs) * epoch_samples,
        np.zeros(n_epochs, dtype=int),
        np.ones(n_epochs, dtype=int),
    ])

    epochs = EpochsArray(data, info, events=events_array, tmin=0.0, verbose=False)
    return epochs


def process_uploaded_file(filepath: str):
    """Full pipeline: load -> preprocess -> epoch."""
    raw = load_eeg_file(filepath)
    epochs = preprocess_and_epoch(raw)
    return epochs
