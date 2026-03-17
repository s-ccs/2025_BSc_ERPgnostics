#!/usr/bin/env python3
"""
Download OpenNeuro ds003517 subject sub-001 EEG files and convert selected
events into a Julia-friendly HDF5/CSV dataset for week 15 self-supervised
learning experiments.

Required Python packages:
    mne, h5py, numpy, pandas, scipy

Primary sources:
    - OpenNeuro ds003517:
      https://openneuro.org/datasets/ds003517
    - Direct public S3 export:
      https://s3.amazonaws.com/openneuro.org/ds003517/
    - Cavanagh & Castellanos (2016):
      https://doi.org/10.1016/j.neuroimage.2016.02.075
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import h5py
import mne
import numpy as np
import pandas as pd


OPENNEURO_DATASET = "ds003517"
OPENNEURO_BASE = f"https://s3.amazonaws.com/openneuro.org/{OPENNEURO_DATASET}"
PAPER_DOI = "10.1016/j.neuroimage.2016.02.075"
DATASET_DOI = "10.18112/openneuro.ds003517.v1.1.0"
REPO_ROOT = Path(__file__).resolve().parents[1]

SUBJECT = "sub-001"
RUNS = (1, 2)
DOWNLOAD_SUFFIXES = (
    "events.tsv",
    "events.json",
    "channels.tsv",
    "electrodes.tsv",
    "coordsystem.json",
    "eeg.json",
    "eeg.set",
    "eeg.fdt",
)

SELECTED_TRIAL_TYPES = (
    "ODDBALL STANDARD",
    "ODDBALL RARE",
    "GAMBLING WIN",
    "GAMBLING LOSS",
    "SHOOT_BUTTON",
    "COLLECT_STAR",
    "MISSILE_HIT_ENEMY",
    "PLAYER_CRASH_WALL",
    "PLAYER_CRASH_ENEMY",
    "COLLECT_AMMO",
)

DROP_TRIAL_TYPES = {
    "STATUS",
    "ODDBALL START",
    "ODDBALL DONE",
    "GAME START",
    "GAME OVER",
    "boundary",
}

TMIN_S = -0.5
TMAX_S = 0.498  # 500 samples at 500 Hz
BASELINE = (-0.2, 0.0)
L_FREQ_HZ = 0.1
H_FREQ_HZ = 20.0


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return

    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def source_relative_path(run: int, suffix: str) -> str:
    prefix = f"{SUBJECT}_task-ContinuousVideoGamePlay_run-{run:02d}"
    if suffix.startswith("eeg."):
        return f"{SUBJECT}/eeg/{prefix}_{suffix}"
    return f"{SUBJECT}/eeg/{prefix}_{suffix}"


def materialize_subject_file(source_root: Path, relpath: str) -> Path:
    destination = source_root / relpath
    if destination.is_symlink() and not destination.exists():
        destination.unlink()

    if destination.is_file():
        return destination

    url = f"{OPENNEURO_BASE}/{relpath}"
    download_file(url, destination)
    return destination


def ensure_subject_files(source_root: Path) -> list[Path]:
    source_root.mkdir(parents=True, exist_ok=True)
    materialized: list[Path] = []
    for run in RUNS:
        for suffix in DOWNLOAD_SUFFIXES:
            relpath = source_relative_path(run, suffix)
            materialized.append(materialize_subject_file(source_root, relpath))
    return materialized


def parse_numeric_column(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace("n/a", np.nan), errors="coerce")


def build_run_epochs(source_root: Path, run: int) -> tuple[mne.Epochs, pd.DataFrame]:
    eeg_dir = source_root / SUBJECT / "eeg"
    set_path = eeg_dir / f"{SUBJECT}_task-ContinuousVideoGamePlay_run-{run:02d}_eeg.set"
    events_path = eeg_dir / f"{SUBJECT}_task-ContinuousVideoGamePlay_run-{run:02d}_events.tsv"

    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")
    drop_channels = [ch for ch in raw.ch_names if ch.upper() in {"VEOG", "HEOG"}]
    if drop_channels:
        raw.drop_channels(drop_channels)

    raw.set_eeg_reference(ref_channels="average", projection=False, verbose="ERROR")
    raw.filter(l_freq=L_FREQ_HZ, h_freq=H_FREQ_HZ, verbose="ERROR")

    events_df = pd.read_csv(events_path, sep="\t")
    events_df = events_df[events_df["trial_type"].isin(SELECTED_TRIAL_TYPES)].copy()
    events_df = events_df[~events_df["trial_type"].isin(DROP_TRIAL_TYPES)].copy()
    events_df["onset"] = parse_numeric_column(events_df["onset"])
    events_df["duration"] = parse_numeric_column(events_df["duration"])
    events_df["response_time"] = parse_numeric_column(events_df["response_time"])
    events_df["sample"] = np.round(events_df["onset"] * raw.info["sfreq"]).astype(int)
    events_df["run"] = run
    events_df["subject"] = SUBJECT
    events_df["source_file"] = set_path.name
    events_df["event_rank_within_type"] = (
        events_df.groupby("trial_type").cumcount().astype(int) + 1
    )

    trial_types = list(dict.fromkeys(events_df["trial_type"].tolist()))
    event_id = {trial_type: idx + 1 for idx, trial_type in enumerate(trial_types)}
    events = np.zeros((len(events_df), 3), dtype=int)
    events[:, 0] = events_df["sample"].to_numpy(dtype=int)
    events[:, 2] = np.array([event_id[tt] for tt in events_df["trial_type"]], dtype=int)

    metadata = events_df[
        [
            "subject",
            "run",
            "trial_type",
            "onset",
            "duration",
            "response_time",
            "sample",
            "value",
            "event_rank_within_type",
            "source_file",
        ]
    ].copy()
    metadata.rename(
        columns={
            "onset": "onset_s",
            "duration": "duration_s",
            "response_time": "response_time_s",
            "sample": "sample_index",
            "value": "trigger_value",
        },
        inplace=True,
    )

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=TMIN_S,
        tmax=TMAX_S,
        baseline=BASELINE,
        preload=True,
        metadata=metadata,
        verbose="ERROR",
    )

    kept = metadata.iloc[epochs.selection].reset_index(drop=True)
    epochs.metadata = kept
    return epochs, kept


def build_dataset(output_dir: Path, source_root: Path) -> None:
    mne.set_log_level("ERROR")

    output_dir.mkdir(parents=True, exist_ok=True)
    if not source_root.is_dir():
        raise FileNotFoundError(f"Expected cloned OpenNeuro dataset at: {source_root}")

    ensure_subject_files(source_root)

    all_epochs: list[mne.Epochs] = []
    metadata_frames: list[pd.DataFrame] = []
    for run in RUNS:
        epochs, metadata = build_run_epochs(source_root, run)
        all_epochs.append(epochs)
        metadata_frames.append(metadata)

    epochs = mne.concatenate_epochs(all_epochs, add_offset=False)
    events_df = pd.concat(metadata_frames, ignore_index=True)
    events_df = events_df.iloc[epochs.selection].reset_index(drop=True)
    events_df["epoch_index"] = np.arange(1, len(events_df) + 1, dtype=int)
    events_df["trial_type_slug"] = events_df["trial_type"].str.lower().str.replace(" ", "_", regex=False)

    data = epochs.get_data(copy=True).astype(np.float32)  # (trial, channel, time)
    data = np.transpose(data, (0, 2, 1))  # Julia reads as (channel, time, trial)
    times_s = np.asarray(epochs.times, dtype=np.float32)
    channel_names = np.asarray(epochs.ch_names, dtype=h5py.string_dtype(encoding="utf-8"))

    h5_path = output_dir / "epochs.hdf5"
    events_path = output_dir / "events.csv"
    readme_path = output_dir / "README.md"
    metadata_path = output_dir / "metadata.json"

    with h5py.File(h5_path, "w") as h5:
        h5.attrs["source_dataset"] = OPENNEURO_DATASET
        h5.attrs["source_dataset_doi"] = DATASET_DOI
        h5.attrs["source_paper_doi"] = PAPER_DOI
        h5.attrs["subject"] = SUBJECT
        h5.attrs["layout"] = "epochs (channels, time, trial) when read from Julia"
        h5.attrs["reader"] = "mne.io.read_raw_eeglab"
        h5.attrs["preprocessing"] = (
            "Dropped VEOG/HEOG, average reference, 0.1-20 Hz filter, "
            "epochs -0.5s..0.498s, baseline -0.2s..0.0s"
        )
        h5.attrs["n_trials"] = int(data.shape[0])
        h5.attrs["n_timepoints"] = int(data.shape[1])
        h5.attrs["n_channels"] = int(data.shape[2])
        h5.attrs["sfreq_hz"] = float(epochs.info["sfreq"])
        h5.create_dataset("epochs", data=data, compression="gzip", compression_opts=4)
        h5.create_dataset("times_s", data=times_s)
        h5.create_dataset("channel_names", data=channel_names)

    events_df.to_csv(events_path, index=False)

    counts = (
        events_df.groupby(["run", "trial_type"]).size().reset_index(name="count")
        .sort_values(["run", "trial_type"])
    )
    metadata = {
        "source_dataset": OPENNEURO_DATASET,
        "source_dataset_doi": DATASET_DOI,
        "source_paper_doi": PAPER_DOI,
        "subject": SUBJECT,
        "selected_trial_types": list(SELECTED_TRIAL_TYPES),
        "tmin_s": TMIN_S,
        "tmax_s": TMAX_S,
        "baseline_s": list(BASELINE),
        "bandpass_hz": [L_FREQ_HZ, H_FREQ_HZ],
        "sfreq_hz": float(epochs.info["sfreq"]),
        "n_trials": int(len(events_df)),
        "n_channels": int(len(epochs.ch_names)),
        "n_timepoints": int(len(epochs.times)),
        "trial_type_counts": counts.to_dict(orient="records"),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    readme_text = f"""# OpenNeuro 8bit EEG Dataset (Derived)

This folder contains a derived subject-level dataset built from OpenNeuro
`{OPENNEURO_DATASET}` for `sub-001`.

Sources
- OpenNeuro dataset DOI: {DATASET_DOI}
- Paper DOI: {PAPER_DOI}
- Public S3 export root: {OPENNEURO_BASE}

Contents
- source dataset root: `{source_root}`
- `epochs.hdf5`: epoched EEG, written so Julia/HDF5 reads it as `(channels, time, trial)`
- `events.csv`: per-epoch metadata aligned to the HDF5 trials
- `metadata.json`: preprocessing summary and trial counts

Preprocessing
- drop `VEOG` and `HEOG`
- average reference
- band-pass filter `{L_FREQ_HZ}` to `{H_FREQ_HZ}` Hz
- epochs from `{TMIN_S}` s to `{TMAX_S}` s around selected events
- baseline correction from `{BASELINE[0]}` s to `{BASELINE[1]}` s
"""
    readme_path.write_text(readme_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=REPO_ROOT / "notebooks" / "datasets" / OPENNEURO_DATASET,
        help="Path to the cloned OpenNeuro dataset root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "notebooks" / "datasets" / f"{OPENNEURO_DATASET}_sub001_derived",
        help="Output directory for derived HDF5/CSV data.",
    )
    args = parser.parse_args()
    build_dataset(args.output_dir.resolve(), args.source_root.resolve())


if __name__ == "__main__":
    main()
