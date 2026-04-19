#!/usr/bin/env python3
"""
Download and prepare two additional cleaned public ERP datasets:

1. **NOD-EEG** (OpenNeuro ds005811) — Visual object recognition, ICA-cleaned,
   64 channels, 250 Hz, epochs [-100, +800] ms.  Source: .fif epoch files.

2. **ZuCo 2.0** (OSF 2urht) — Fixation-locked reading EEG, ICA-cleaned (MARA),
   128 channels (105 usable), 500 Hz, word-level epochs.  Source: .mat files.

Both are converted into the same HDF5 / events.csv / metadata.json bundle layout
used by the rest of the Week 15 comparison notebooks.

Required Python packages:
    mne, h5py, numpy, pandas, scipy, openneuro-py, osfclient, requests
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import mne
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "notebooks" / "datasets"


# ---------------------------------------------------------------------------
#  NOD-EEG  (OpenNeuro ds005811)
# ---------------------------------------------------------------------------

NOD_EEG_DATASET_ID = "ds005811"
NOD_EEG_DATASET_KEY = "nod_eeg_public"
NOD_EEG_COMPONENT = "NOD-EEG Visual"
NOD_EEG_SOURCE_URL = "https://openneuro.org/datasets/ds005811"
NOD_EEG_PREFERRED_CHANNELS = ["Oz", "POz", "PO3", "PO4", "O1", "O2"]
NOD_EEG_RECOMMENDED_SORT = [
    "image_category",
    "behavioral_response",
    "session",
    "run",
    "epoch_index",
]
NOD_EEG_DEFAULT_SUBJECTS = ["sub-01", "sub-02"]


def download_nod_eeg_subject(subject: str, download_dir: Path) -> Path:
    """Download a single subject's epoch file and events CSV from OpenNeuro."""
    import openneuro

    # Actual file paths in ds005811 (note: _eeg_ before _epo.fif)
    epoch_include = f"derivatives/preprocessed/epochs/{subject}_eeg_epo.fif"
    events_include = f"derivatives/detailed_events/{subject}_events.csv"

    target = download_dir / NOD_EEG_DATASET_ID
    target.mkdir(parents=True, exist_ok=True)

    for include_path in [epoch_include, events_include]:
        dest = target / include_path
        if dest.exists():
            print(f"  [skip] {dest} already exists")
            continue
        print(f"  [download] {include_path} ...")
        openneuro.download(
            dataset=NOD_EEG_DATASET_ID,
            target_dir=target,
            include=[include_path],
        )
    return target


def build_nod_eeg_bundle(
    output_root: Path,
    subjects: list[str],
    download_dir: Path,
) -> Path:
    """Convert NOD-EEG epoch .fif files to standard bundle format."""
    output_dir = output_root / NOD_EEG_DATASET_KEY
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_path = output_dir / "epochs.hdf5"
    events_path = output_dir / "events.csv"
    metadata_path = output_dir / "metadata.json"

    all_events: list[dict] = []
    subject_trial_counts: list[dict] = []

    ds_root = download_dir / NOD_EEG_DATASET_ID

    with h5py.File(h5_path, "w") as h5:
        h5.attrs["dataset_key"] = NOD_EEG_DATASET_KEY
        h5.attrs["component"] = NOD_EEG_COMPONENT
        h5.attrs["source_component"] = NOD_EEG_SOURCE_URL
        h5.attrs["layout"] = "subjects/<sub>/epochs (channels, time, trial)"
        h5.attrs["reader"] = "mne.read_epochs"
        subjects_group = h5.create_group("subjects")

        for subject in subjects:
            epoch_file = ds_root / "derivatives" / "preprocessed" / "epochs" / f"{subject}_eeg_epo.fif"
            events_file = ds_root / "derivatives" / "detailed_events" / f"{subject}_events.csv"

            if not epoch_file.exists():
                print(f"  [warn] epoch file missing: {epoch_file}, skipping")
                continue

            print(f"  [load] {subject} epochs from {epoch_file.name}")
            epochs = mne.read_epochs(str(epoch_file), preload=True, verbose="ERROR")
            data = epochs.get_data(copy=True).astype(np.float32)  # (trial, channel, time)
            # Transpose for Julia: will be read back as (channel, time, trial)
            data = np.transpose(data, (1, 2, 0))
            ch_names = np.asarray(epochs.ch_names, dtype=h5py.string_dtype(encoding="utf-8"))
            times_s = np.asarray(epochs.times, dtype=np.float32)

            # Try to load detailed events
            events_meta = None
            if events_file.exists():
                events_meta = pd.read_csv(events_file)
                print(f"    events CSV: {len(events_meta)} rows, columns: {list(events_meta.columns)}")

            n_trials = data.shape[2]

            # Build event rows
            for trial_idx in range(n_trials):
                row: dict = {
                    "dataset_key": NOD_EEG_DATASET_KEY,
                    "component": NOD_EEG_COMPONENT,
                    "subject_id": int(subject.replace("sub-", "")),
                    "subject_label": subject,
                    "epoch_index": trial_idx + 1,
                }

                # Add event metadata if available
                if events_meta is not None and trial_idx < len(events_meta):
                    erow = events_meta.iloc[trial_idx]
                    for col in events_meta.columns:
                        val = erow[col]
                        if pd.isna(val):
                            continue
                        col_clean = str(col).strip().lower().replace(" ", "_")
                        if col_clean not in row:
                            row[col_clean] = val
                else:
                    # Fall back to MNE event codes
                    if trial_idx < len(epochs.events):
                        row["event_code"] = int(epochs.events[trial_idx, 2])

                all_events.append(row)

            group = subjects_group.create_group(subject)
            group.create_dataset("epochs", data=data, compression="gzip", compression_opts=4)
            group.create_dataset("times_s", data=times_s)
            group.create_dataset("channel_names", data=ch_names)
            group.attrs["subject_id"] = int(subject.replace("sub-", ""))
            group.attrs["subject_label"] = subject
            group.attrs["sfreq_hz"] = float(epochs.info["sfreq"])
            group.attrs["n_channels"] = int(len(epochs.ch_names))
            group.attrs["n_timepoints"] = int(len(epochs.times))
            group.attrs["n_trials"] = n_trials
            group.attrs["source_set_relpath"] = str(epoch_file.relative_to(output_dir) if epoch_file.is_relative_to(output_dir) else epoch_file.name)
            group.attrs["source_eventlist_relpath"] = str(events_file.relative_to(output_dir) if events_file.is_relative_to(output_dir) else events_file.name)

            subject_trial_counts.append({"subject_label": subject, "n_trials": n_trials})
            print(f"    {subject}: {n_trials} trials, {len(epochs.ch_names)} channels, {len(epochs.times)} timepoints")

    events_df = pd.DataFrame(all_events)
    events_df.sort_values(["subject_id", "epoch_index"], inplace=True)
    events_df.to_csv(events_path, index=False)

    # Determine actual sort columns from events
    sort_cols = []
    for col in NOD_EEG_RECOMMENDED_SORT:
        if col in events_df.columns and events_df[col].dropna().nunique() > 1:
            sort_cols.append(col)
    # Add any other columns with variation
    for col in events_df.columns:
        if col in sort_cols or col in {"dataset_key", "component", "subject_id", "subject_label"}:
            continue
        if events_df[col].dropna().nunique() > 1:
            sort_cols.append(col)

    # Determine preferred channels from what's actually available
    if all_events:
        first_subj = subjects[0]
        epoch_file = ds_root / "derivatives" / "preprocessed" / "epochs" / f"{first_subj}_eeg_epo.fif"
        avail_channels = mne.read_epochs(str(epoch_file), preload=False, verbose="ERROR").ch_names
        preferred = [ch for ch in NOD_EEG_PREFERRED_CHANNELS if ch in avail_channels]
        if len(preferred) < 4:
            for ch in avail_channels:
                if ch not in preferred:
                    preferred.append(ch)
                if len(preferred) >= 4:
                    break
    else:
        preferred = NOD_EEG_PREFERRED_CHANNELS

    metadata = {
        "dataset_key": NOD_EEG_DATASET_KEY,
        "component": NOD_EEG_COMPONENT,
        "source_component": NOD_EEG_SOURCE_URL,
        "source_root_listing": f"https://openneuro.org/datasets/{NOD_EEG_DATASET_ID}",
        "source_processing_scripts": "https://github.com/OpenNeuroDatasets/ds005811",
        "reader_docs": "https://mne.tools/stable/generated/mne.read_epochs.html",
        "selected_subjects": subjects,
        "preferred_channels": preferred[:4],
        "recommended_sort_columns": sort_cols[:6],
        "hdf5_path": "epochs.hdf5",
        "events_csv_path": "events.csv",
        "notes": [
            "Source epoch files are the official NOD-EEG preprocessed .fif files from derivatives/epochs/.",
            "Preprocessing: RANSAC bad channel detection + spline interpolation, Zapline 50Hz removal, ICA artifact rejection (MNE-ICALabel), downsampled to 250Hz, epochs -100 to +800ms with baseline correction.",
            "This conversion only loads the cleaned epochs and stores them in a Julia-friendly layout.",
        ],
        "official_source_examples": {},
        "subject_trial_counts": subject_trial_counts,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"  [done] NOD-EEG bundle: {output_dir}")
    return output_dir


# ---------------------------------------------------------------------------
#  ZuCo 2.0  (OSF 2urht)
# ---------------------------------------------------------------------------

ZUCO2_DATASET_KEY = "zuco2_nr_public"
ZUCO2_COMPONENT = "ZuCo2 Reading Fixations"
ZUCO2_SOURCE_URL = "https://osf.io/2urht/"
ZUCO2_PREFERRED_CHANNELS = [
    "E8",
    "E14",
    "E21",
    "E25",
    "E126",
    "E127",
    "E128",
    "E11",
    "E6",
    "E62",
    "E75",
]  # GSN128 face/peripheral electrodes first, then Fz/FCz/Pz/POz equivalents.
ZUCO2_RECOMMENDED_SORT = [
    "fixation_duration_ms",
    "FFD_ms",
    "GD_ms",
    "TRT_ms",
    "nFixations",
    "word_length",
    "word_index",
    "epoch_index",
]
ZUCO2_DEFAULT_SUBJECTS = ["YAK"]

# Direct download URLs from OSF API (task1 - NR / Matlab files)
ZUCO2_SUBJECT_DOWNLOADS: dict[str, tuple[str, int]] = {
    "YAK": ("https://osf.io/download/8r94t/", 1_235_552_076),
    "YHS": ("https://osf.io/download/kypmc/", 1_290_663_769),
    "YAC": ("https://osf.io/download/6g23c/", 1_340_390_157),
    "YFS": ("https://osf.io/download/zwqjc/", 1_400_000_000),
    "YDR": ("https://osf.io/download/wpn38/", 1_500_000_000),
    "YRH": ("https://osf.io/download/25edz/", 1_500_000_000),
    "YDG": ("https://osf.io/download/g7cnq/", 1_748_510_042),
    "YSD": ("https://osf.io/download/rjhze/", 1_800_000_000),
    "YFR": ("https://osf.io/download/4yfb9/", 1_892_890_796),
    "YLS": ("https://osf.io/download/p9nme/", 1_900_000_000),
    "YIS": ("https://osf.io/download/m85z9/", 2_000_000_000),
    "YRP": ("https://osf.io/download/dhjnk/", 2_319_904_154),
    "YRK": ("https://osf.io/download/mjbhw/", 2_347_425_489),
    "YSL": ("https://osf.io/download/st5nw/", 2_356_341_298),
    "YAG": ("https://osf.io/download/vn74p/", 2_400_000_000),
    "YMD": ("https://osf.io/download/3zk8s/", 2_400_000_000),
    "YMS": ("https://osf.io/download/wczfg/", 2_446_833_984),
    "YTL": ("https://osf.io/download/6ab98/", 2_495_299_295),
}


def download_zuco2_subject(subject: str, download_dir: Path) -> Path:
    """Download a ZuCo 2.0 subject .mat file from OSF via direct URL."""
    import requests

    target_dir = download_dir / "zuco2"
    target_dir.mkdir(parents=True, exist_ok=True)

    if subject not in ZUCO2_SUBJECT_DOWNLOADS:
        raise ValueError(
            f"Unknown ZuCo 2.0 subject: {subject}. "
            f"Available: {sorted(ZUCO2_SUBJECT_DOWNLOADS.keys())}"
        )

    url, expected_size = ZUCO2_SUBJECT_DOWNLOADS[subject]
    filename = f"results{subject}_NR.mat"
    dest = target_dir / filename

    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"  [skip] {filename} already exists ({dest.stat().st_size / 1e9:.1f} GB)")
        return target_dir

    print(f"  [download] {filename} (~{expected_size / 1e9:.1f} GB) from {url}")
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as fp:
        for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
            fp.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r    {downloaded / 1e9:.1f} / {total / 1e9:.1f} GB ({pct:.0f}%)", end="", flush=True)
    print()
    print(f"    -> {dest} ({dest.stat().st_size / 1e9:.1f} GB)")
    return target_dir


def load_zuco2_mat(mat_path: Path) -> tuple[list[np.ndarray], list[dict]]:
    """Load word-level EEG epochs and metadata from a ZuCo 2.0 .mat file.

    Returns (epochs_list, metadata_list) where each entry is one word.
    """
    from scipy.io import loadmat

    try:
        data = loadmat(str(mat_path), squeeze_me=True, simplify_cells=True)
    except Exception:
        # Try HDF5-based .mat (v7.3)
        import h5py as h5
        epochs_list: list[np.ndarray] = []
        meta_list: list[dict] = []
        with h5.File(str(mat_path), "r") as f:
            if "sentenceData" not in f:
                print(f"    [warn] no sentenceData in {mat_path.name}")
                return epochs_list, meta_list
            sent_data = f["sentenceData"]
            for sent_key in sorted(sent_data.keys()):
                sent = sent_data[sent_key]
                if "word" not in sent:
                    continue
                word_group = sent["word"]
                for word_key in sorted(word_group.keys()):
                    word = word_group[word_key]
                    if "rawEEG" not in word:
                        continue
                    raw = np.array(word["rawEEG"], dtype=np.float32)
                    if raw.ndim != 2 or raw.size == 0:
                        continue
                    meta: dict = {"sentence_key": sent_key, "word_key": word_key}
                    for attr in ["FFD", "GD", "TRT", "nFixations"]:
                        if attr in word:
                            val = np.array(word[attr]).flat[0]
                            meta[attr] = float(val)
                    epochs_list.append(raw)
                    meta_list.append(meta)
        return epochs_list, meta_list

    # scipy loadmat path (v5 / v7)
    epochs_list = []
    meta_list = []

    if "sentenceData" not in data:
        print(f"    [warn] no sentenceData in {mat_path.name}")
        return epochs_list, meta_list

    sent_data = data["sentenceData"]
    if not hasattr(sent_data, "__len__"):
        sent_data = [sent_data]

    for sent_idx, sent in enumerate(sent_data):
        # Handle both structured arrays and dicts
        if hasattr(sent, "dtype") and sent.dtype.names:
            word_data = sent["word"] if "word" in sent.dtype.names else None
            content = str(sent["content"]) if "content" in sent.dtype.names else ""
        elif isinstance(sent, dict):
            word_data = sent.get("word")
            content = str(sent.get("content", ""))
        else:
            continue

        if word_data is None:
            continue

        if not hasattr(word_data, "__len__"):
            word_data = [word_data]

        for word_idx, word in enumerate(word_data):
            raw_eeg = None
            meta: dict = {
                "sentence_index": sent_idx,
                "word_index": word_idx,
                "sentence_content": content[:100],
            }

            if hasattr(word, "dtype") and word.dtype.names:
                if "rawEEG" in word.dtype.names:
                    raw_eeg = np.asarray(word["rawEEG"], dtype=np.float32)
                for attr in ["FFD", "GD", "TRT", "nFixations"]:
                    if attr in word.dtype.names:
                        val = word[attr]
                        if hasattr(val, "flat"):
                            val = val.flat[0]
                        meta[attr] = float(val) if not np.isnan(float(val)) else np.nan
                if "content" in word.dtype.names:
                    meta["word_content"] = str(word["content"])
            elif isinstance(word, dict):
                raw_eeg = word.get("rawEEG")
                if raw_eeg is not None:
                    raw_eeg = np.asarray(raw_eeg, dtype=np.float32)
                for attr in ["FFD", "GD", "TRT", "nFixations"]:
                    if attr in word:
                        val = word[attr]
                        if hasattr(val, "flat"):
                            val = val.flat[0]
                        meta[attr] = float(val) if not np.isnan(float(val)) else np.nan
                meta["word_content"] = str(word.get("content", ""))

            if raw_eeg is not None and raw_eeg.ndim == 2 and raw_eeg.size > 0:
                epochs_list.append(raw_eeg)
                meta_list.append(meta)

    return epochs_list, meta_list


def build_zuco2_bundle(
    output_root: Path,
    subjects: list[str],
    download_dir: Path,
) -> Path:
    """Convert ZuCo 2.0 .mat files to standard bundle format.

    ZuCo 2.0 word epochs have variable length (depending on fixation duration).
    We crop/pad all epochs to a fixed window for the ERP image pipeline.
    """
    output_dir = output_root / ZUCO2_DATASET_KEY
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_path = output_dir / "epochs.hdf5"
    events_path = output_dir / "events.csv"
    metadata_path = output_dir / "metadata.json"

    SFREQ = 500.0  # ZuCo sampling rate
    EPOCH_TMIN_S = -0.5
    EPOCH_DURATION_S = 1.5
    EPOCH_SAMPLES = int(round(SFREQ * EPOCH_DURATION_S))
    PRE_SAMPLES = int(round(abs(EPOCH_TMIN_S) * SFREQ))

    all_events: list[dict] = []
    subject_trial_counts: list[dict] = []

    mat_dir = download_dir / "zuco2"

    with h5py.File(h5_path, "w") as h5:
        h5.attrs["dataset_key"] = ZUCO2_DATASET_KEY
        h5.attrs["component"] = ZUCO2_COMPONENT
        h5.attrs["source_component"] = ZUCO2_SOURCE_URL
        h5.attrs["layout"] = "subjects/<sub>/epochs (channels, time, trial)"
        h5.attrs["reader"] = "scipy.io.loadmat / h5py"
        subjects_group = h5.create_group("subjects")

        for subject in subjects:
            subject_label = f"sub-{subject}"

            # Find .mat files for this subject
            mat_files = sorted(mat_dir.glob(f"*{subject}*NR*.mat"))
            if not mat_files:
                mat_files = sorted(mat_dir.glob(f"*{subject}*.mat"))
            if not mat_files:
                print(f"  [warn] no .mat files found for {subject} in {mat_dir}")
                continue

            print(f"  [load] {subject}: {len(mat_files)} .mat files")

            all_word_epochs: list[np.ndarray] = []
            all_word_meta: list[dict] = []

            for mat_file in mat_files:
                print(f"    loading {mat_file.name} ...")
                word_epochs, word_meta = load_zuco2_mat(mat_file)
                print(f"      -> {len(word_epochs)} word epochs")
                all_word_epochs.extend(word_epochs)
                all_word_meta.extend(word_meta)

            if not all_word_epochs:
                print(f"  [warn] no epochs extracted for {subject}")
                continue

            # Determine number of channels from the data
            n_channels_list = [e.shape[1] if e.ndim == 2 else e.shape[0] for e in all_word_epochs]
            most_common_nch = int(pd.Series(n_channels_list).mode().iloc[0])

            # Filter to epochs with consistent channel count
            valid_indices = [i for i, e in enumerate(all_word_epochs)
                           if (e.shape[1] if e.ndim == 2 else e.shape[0]) == most_common_nch]
            all_word_epochs = [all_word_epochs[i] for i in valid_indices]
            all_word_meta = [all_word_meta[i] for i in valid_indices]

            if not all_word_epochs:
                continue

            # Crop/pad to fixed length
            # Each epoch: (time_samples, channels) -> we want (channels, EPOCH_SAMPLES)
            padded_epochs = []
            for epoch in all_word_epochs:
                if epoch.ndim == 2:
                    # (time, channels) -> (channels, time)
                    if epoch.shape[0] == most_common_nch:
                        epoch = epoch  # already (channels, time)
                    else:
                        epoch = epoch.T
                    if epoch.shape[0] != most_common_nch:
                        epoch = epoch.T
                    # Now epoch is (channels, time)
                    n_time = epoch.shape[1]
                else:
                    continue

                padded = np.zeros((most_common_nch, EPOCH_SAMPLES), dtype=np.float32)
                copy_len = min(n_time, EPOCH_SAMPLES)
                padded[:, :copy_len] = epoch[:, :copy_len]
                padded_epochs.append(padded)

            if not padded_epochs:
                continue

            # Stack: (channels, time, trials)
            data = np.stack(padded_epochs, axis=2).astype(np.float32)
            n_trials = data.shape[2]

            # Generate channel names (GSN 128 naming)
            ch_names = [f"E{i+1}" for i in range(most_common_nch)]
            ch_names_h5 = np.asarray(ch_names, dtype=h5py.string_dtype(encoding="utf-8"))

            # Time axis: -500ms to +998ms at 500 Hz.
            times_s = np.linspace(EPOCH_TMIN_S, (EPOCH_SAMPLES - PRE_SAMPLES - 1) / SFREQ, EPOCH_SAMPLES).astype(np.float32)

            # Build event rows
            for trial_idx, meta in enumerate(all_word_meta[:n_trials]):
                row: dict = {
                    "dataset_key": ZUCO2_DATASET_KEY,
                    "component": ZUCO2_COMPONENT,
                    "subject_id": 1,
                    "subject_label": subject_label,
                    "epoch_index": trial_idx + 1,
                }
                if "FFD" in meta and not np.isnan(meta["FFD"]):
                    row["FFD_ms"] = meta["FFD"]
                if "GD" in meta and not np.isnan(meta["GD"]):
                    row["GD_ms"] = meta["GD"]
                    row["fixation_duration_ms"] = meta["GD"]
                elif "FFD" in meta and not np.isnan(meta["FFD"]):
                    row["fixation_duration_ms"] = meta["FFD"]
                if "TRT" in meta and not np.isnan(meta["TRT"]):
                    row["TRT_ms"] = meta["TRT"]
                if "nFixations" in meta and not np.isnan(meta["nFixations"]):
                    row["nFixations"] = int(meta["nFixations"])
                if "word_content" in meta:
                    row["word_content"] = meta["word_content"]
                    row["word_length"] = len(str(meta["word_content"]))
                if "word_index" in meta:
                    row["word_index"] = int(meta["word_index"])
                if "sentence_index" in meta:
                    row["sentence_index"] = int(meta["sentence_index"])

                all_events.append(row)

            group = subjects_group.create_group(subject_label)
            group.create_dataset("epochs", data=data, compression="gzip", compression_opts=4)
            group.create_dataset("times_s", data=times_s)
            group.create_dataset("channel_names", data=ch_names_h5)
            group.attrs["subject_id"] = 1
            group.attrs["subject_label"] = subject_label
            group.attrs["sfreq_hz"] = SFREQ
            group.attrs["n_channels"] = most_common_nch
            group.attrs["n_timepoints"] = EPOCH_SAMPLES
            group.attrs["n_trials"] = n_trials
            group.attrs["source_set_relpath"] = str(mat_files[0].name)
            group.attrs["source_eventlist_relpath"] = ""

            subject_trial_counts.append({"subject_label": subject_label, "n_trials": n_trials})
            print(f"    {subject_label}: {n_trials} trials, {most_common_nch} channels, {EPOCH_SAMPLES} timepoints")

    events_df = pd.DataFrame(all_events)
    if not events_df.empty:
        events_df.sort_values(["subject_id", "epoch_index"], inplace=True)
    events_df.to_csv(events_path, index=False)

    # Determine actual sort columns
    sort_cols = []
    for col in ZUCO2_RECOMMENDED_SORT:
        if col in events_df.columns and events_df[col].dropna().nunique() > 1:
            sort_cols.append(col)

    metadata = {
        "dataset_key": ZUCO2_DATASET_KEY,
        "component": ZUCO2_COMPONENT,
        "source_component": ZUCO2_SOURCE_URL,
        "source_root_listing": "https://osf.io/2urht/",
        "source_processing_scripts": "https://github.com/norahollenstein/zuco-benchmark",
        "reader_docs": "https://osf.io/cqa8j/wiki/Data%20format/",
        "selected_subjects": [f"sub-{s}" for s in subjects],
        "preferred_channels": ZUCO2_PREFERRED_CHANNELS,
        "recommended_sort_columns": sort_cols if sort_cols else ZUCO2_RECOMMENDED_SORT[:4],
        "hdf5_path": "epochs.hdf5",
        "events_csv_path": "events.csv",
        "notes": [
            "Source .mat files are the official ZuCo 2.0 preprocessed release from OSF.",
            "Preprocessing: bandpass 0.5-30Hz, ICA artifact removal via MARA, average re-reference.",
            "Word-level/fixation-related epochs extracted from sentenceData structures, cropped/padded to [-500, +1000] ms.",
            "The notebook path should disable baseline correction, matching fixation-locked sources where the prior saccade can sit in the baseline interval.",
            "Sort variables: fixation_duration_ms (GD when present, otherwise FFD), FFD, GD, TRT, nFixations, word_length.",
        ],
        "official_source_examples": {},
        "subject_trial_counts": subject_trial_counts,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"  [done] ZuCo 2.0 bundle: {output_dir}")
    return output_dir


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=None,
        help="Directory for raw downloads (default: <output-root>/_downloads)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["nod_eeg", "zuco2"],
        choices=["nod_eeg", "zuco2"],
    )
    parser.add_argument(
        "--nod-eeg-subjects",
        nargs="+",
        default=NOD_EEG_DEFAULT_SUBJECTS,
    )
    parser.add_argument(
        "--zuco2-subjects",
        nargs="+",
        default=ZUCO2_DEFAULT_SUBJECTS,
    )
    args = parser.parse_args()

    download_dir = args.download_dir or (args.output_root / "_downloads")
    download_dir.mkdir(parents=True, exist_ok=True)

    if "nod_eeg" in args.datasets:
        print("\n=== NOD-EEG (OpenNeuro ds005811) ===")
        for subject in args.nod_eeg_subjects:
            download_nod_eeg_subject(subject, download_dir)
        build_nod_eeg_bundle(args.output_root, args.nod_eeg_subjects, download_dir)

    if "zuco2" in args.datasets:
        print("\n=== ZuCo 2.0 (OSF 2urht) ===")
        for subject in args.zuco2_subjects:
            download_zuco2_subject(subject, download_dir)
        build_zuco2_bundle(args.output_root, args.zuco2_subjects, download_dir)


if __name__ == "__main__":
    main()
