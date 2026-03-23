#!/usr/bin/env python3
"""
Download representative subsets of the previously audited public ERP datasets
and convert them into the shared HDF5/CSV bundle layout used by the week 15
comparison notebook.

The goal here is not to mirror entire upstream repositories locally. Instead,
each bundle materializes enough public source data to compare ERP-image
structure across datasets and sort variables inside the notebook.

Required Python packages:
    mne, h5py, numpy, pandas, scipy
"""

from __future__ import annotations

import argparse
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import h5py
import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "notebooks" / "datasets"
MNE_EDF_DOCS = "https://mne.tools/stable/generated/mne.io.read_raw_edf.html"
MNE_EEGLAB_RAW_DOCS = "https://mne.tools/stable/generated/mne.io.read_raw_eeglab.html"
SCIPY_LOADMAT_DOCS = "https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html"
MOABB_BI_LOADER = "https://raw.githubusercontent.com/NeuroTechX/moabb/develop/moabb/datasets/braininvaders.py"
MOABB_BNCI_LOADER = "https://raw.githubusercontent.com/NeuroTechX/moabb/develop/moabb/datasets/bnci/bnci_2014.py"

BI2013A_URL = "https://zenodo.org/record/2669187/files/subject08.zip"
BI2014A_URL = "https://zenodo.org/record/3266223/files/subject_01.zip"
BI2014B_URL = "https://zenodo.org/record/3267302/files/group_01_mat.zip"
BI2015A_URL = "https://zenodo.org/record/3266930/files/subject_01_mat.zip"
BI2015B_URL = "https://zenodo.org/record/3268762/files/group_01_mat.zip"
CATTAN_VR_URL = "https://zenodo.org/record/2605205/files/subject_01_VR.mat"
CATTAN_PC_URL = "https://zenodo.org/record/2605205/files/subject_01_PC.mat"
BNCI_008_A01_URL = "https://bnci-horizon-2020.eu/database/data-sets/008-2014/A01.mat"
BIGP3_BASE = (
    "https://physionet.org/files/bigp3bci/1.0.0/"
    "bigP3BCI-data/StudyA/A_01/SE001/Train/CB"
)
BIGP3_FILES = [f"A_01_SE001_CB_Train0{i}.edf" for i in range(1, 6)]

BI2013A_CHANNELS = [
    "Fp1",
    "Fp2",
    "F5",
    "AFz",
    "F6",
    "T7",
    "Cz",
    "T8",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "O1",
    "Oz",
    "O2",
]
BI2014A_CHANNELS = [
    "Fp1",
    "Fp2",
    "F3",
    "AFz",
    "F4",
    "T7",
    "Cz",
    "T8",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "O1",
    "Oz",
    "O2",
]
BI32_CHANNELS = [
    "Fp1",
    "Fp2",
    "AFz",
    "F7",
    "F3",
    "F4",
    "F8",
    "FC5",
    "FC1",
    "FC2",
    "FC6",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "CP5",
    "CP1",
    "CP2",
    "CP6",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "PO7",
    "O1",
    "Oz",
    "O2",
    "PO8",
    "PO9",
    "PO10",
]
CATTAN_CHANNELS = [
    "Fp1",
    "Fp2",
    "Fc5",
    "Fz",
    "Fc6",
    "T7",
    "Cz",
    "T8",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "O1",
    "Oz",
    "O2",
]


@dataclass(frozen=True)
class SubjectBundle:
    subject_label: str
    subject_id: int
    channel_names: list[str]
    sfreq_hz: float
    times_s: np.ndarray
    epochs: np.ndarray  # (trial, time, channel)
    events: pd.DataFrame
    source_set_relpath: str
    source_eventlist_relpath: str


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    component: str
    source_component: str
    source_processing_scripts: str
    reader_docs: str
    preferred_channels: tuple[str, ...]
    recommended_sort_columns: tuple[str, ...]
    official_source_examples: dict[str, str]
    prepare: Callable[[Path, "DatasetConfig"], list[SubjectBundle]]


def download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return destination

    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    return destination


def extract_members(archive_path: Path, destination_dir: Path, members: list[str]) -> list[Path]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(archive_path, "r") as zf:
        for member in members:
            target = destination_dir / member
            if not target.is_file():
                zf.extract(member, destination_dir)
            extracted.append(target)
    return extracted


def n_timepoints_for_1s(sfreq_hz: float) -> int:
    return int(round(float(sfreq_hz)))


def epoch_times_s(sfreq_hz: float, n_timepoints: int, *, tmin_s: float = 0.0) -> np.ndarray:
    return (
        np.arange(n_timepoints, dtype=np.float32) / np.float32(sfreq_hz) + np.float32(tmin_s)
    ).astype(np.float32)


def positive_rising_edges(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal)
    active = signal > 0
    return np.flatnonzero(active & np.r_[True, ~active[:-1]])


def code_rising_edges(signal: np.ndarray, valid_codes: set[int] | None = None) -> np.ndarray:
    signal = np.asarray(signal)
    if valid_codes is None:
        active = signal > 0
    else:
        active = np.isin(signal.astype(int), list(valid_codes))
    return np.flatnonzero(active & np.r_[True, ~active[:-1]])


def extract_epochs(
    eeg_channels_samples: np.ndarray,
    onset_samples: np.ndarray,
    sfreq_hz: float,
    *,
    tmin_s: float = -0.2,
    duration_s: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eeg_channels_samples = np.asarray(eeg_channels_samples, dtype=np.float32)
    onset_samples = np.asarray(onset_samples, dtype=int)
    n_timepoints = int(round(float(duration_s) * float(sfreq_hz)))
    pre_samples = int(round(float(tmin_s) * float(sfreq_hz)))
    kept_onsets: list[int] = []
    epochs: list[np.ndarray] = []
    for onset in onset_samples:
        start = onset + pre_samples
        stop = start + n_timepoints
        if start < 0 or stop > eeg_channels_samples.shape[1]:
            continue
        epochs.append(eeg_channels_samples[:, start:stop].T.copy())
        kept_onsets.append(int(onset))

    if not epochs:
        raise RuntimeError("No valid 1-second epochs could be extracted.")

    return (
        np.stack(epochs, axis=0).astype(np.float32),
        np.asarray(kept_onsets, dtype=int),
        (np.arange(n_timepoints, dtype=np.float32) / np.float32(sfreq_hz) + np.float32(tmin_s)).astype(np.float32),
    )


def rarity_condition_labels(codes: np.ndarray) -> list[str]:
    codes = np.asarray(codes, dtype=int)
    positive = codes[codes > 0]
    uniq, counts = np.unique(positive, return_counts=True)
    target_code = int(uniq[np.argmin(counts)])
    return ["target" if int(code) == target_code else "non_target" for code in codes]


def round_channel(signal: np.ndarray) -> np.ndarray:
    return np.rint(np.asarray(signal)).astype(int)


def subject_label(subject_id: int) -> str:
    return f"sub-{subject_id:03d}"


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def write_dataset_bundle(output_dir: Path, config: DatasetConfig, bundles: list[SubjectBundle]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_path = output_dir / "epochs.hdf5"
    events_path = output_dir / "events.csv"
    metadata_path = output_dir / "metadata.json"
    readme_path = output_dir / "README.md"

    events_frames: list[pd.DataFrame] = []

    with h5py.File(h5_path, "w") as h5:
        h5.attrs["dataset_key"] = config.key
        h5.attrs["component"] = config.component
        h5.attrs["source_component"] = config.source_component
        h5.attrs["source_processing_scripts"] = config.source_processing_scripts
        h5.attrs["reader_docs"] = config.reader_docs
        h5.attrs["layout"] = "subjects/<sub>/epochs (channels, time, trial) when read from Julia"
        subjects_group = h5.create_group("subjects")

        for bundle in bundles:
            group = subjects_group.create_group(bundle.subject_label)
            group.create_dataset("epochs", data=bundle.epochs, compression="gzip", compression_opts=4)
            group.create_dataset("times_s", data=bundle.times_s.astype(np.float32))
            group.create_dataset(
                "channel_names",
                data=np.asarray(bundle.channel_names, dtype=h5py.string_dtype(encoding="utf-8")),
            )
            group.attrs["subject_id"] = int(bundle.subject_id)
            group.attrs["subject_label"] = bundle.subject_label
            group.attrs["sfreq_hz"] = float(bundle.sfreq_hz)
            group.attrs["n_channels"] = int(bundle.epochs.shape[2])
            group.attrs["n_timepoints"] = int(bundle.epochs.shape[1])
            group.attrs["n_trials"] = int(bundle.epochs.shape[0])
            group.attrs["source_set_relpath"] = bundle.source_set_relpath
            group.attrs["source_eventlist_relpath"] = bundle.source_eventlist_relpath
            events_frames.append(bundle.events.copy())

    events_df = pd.concat(events_frames, ignore_index=True)
    events_df.to_csv(events_path, index=False)

    metadata = {
        "dataset_key": config.key,
        "component": config.component,
        "source_component": config.source_component,
        "source_processing_scripts": config.source_processing_scripts,
        "reader_docs": config.reader_docs,
        "selected_subjects": [bundle.subject_label for bundle in bundles],
        "preferred_channels": list(config.preferred_channels),
        "recommended_sort_columns": list(config.recommended_sort_columns),
        "official_source_examples": config.official_source_examples,
        "n_subjects": len(bundles),
        "n_trials_total": int(len(events_df)),
        "trial_counts_by_subject": (
            events_df.groupby("subject_label").size().reset_index(name="count").to_dict(orient="records")
        ),
        "sort_columns_present": list(events_df.columns),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, default=json_default), encoding="utf-8")

    readme_lines = [
        f"# {config.component}",
        "",
        "This bundle stores a representative public subset converted for the week 15 ERP-image notebook.",
        "",
        f"- Dataset key: `{config.key}`",
        f"- Source: {config.source_component}",
        f"- Loader reference: {config.source_processing_scripts}",
        f"- Reader docs: {config.reader_docs}",
        f"- Subjects stored: {', '.join(metadata['selected_subjects'])}",
        f"- Trials stored: {metadata['n_trials_total']}",
        "",
        "Layout:",
        "- `epochs.hdf5`: HDF5 with `subjects/<subject>/epochs`, stored as `(trial, time, channel)`",
        "- `events.csv`: per-epoch metadata and sort variables",
        "- `metadata.json`: notebook-facing bundle metadata",
    ]
    readme_path.write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    return output_dir


def finalize_subject_bundle(
    subject_id: int,
    channel_names: list[str],
    sfreq_hz: float,
    epochs_parts: list[np.ndarray],
    event_frames: list[pd.DataFrame],
    source_relpaths: list[str],
    times_s: np.ndarray | None = None,
) -> SubjectBundle:
    if not epochs_parts:
        raise RuntimeError("No epochs were collected for the subject bundle.")
    epochs = np.concatenate(epochs_parts, axis=0).astype(np.float32)
    prepared_frames: list[pd.DataFrame] = []
    for part_idx, frame in enumerate(event_frames, start=1):
        frame_local = frame.copy()
        frame_local["source_part_index"] = int(part_idx)
        frame_local["source_epoch_index"] = np.arange(1, len(frame_local) + 1, dtype=int)
        prepared_frames.append(frame_local)

    events = pd.concat(prepared_frames, ignore_index=True)
    events["epoch_index"] = np.arange(1, len(events) + 1, dtype=int)
    if times_s is None:
        times_s = epoch_times_s(sfreq_hz, epochs.shape[1])
    else:
        times_s = np.asarray(times_s, dtype=np.float32)
        if times_s.shape != (epochs.shape[1],):
            raise RuntimeError(
                f"times_s shape {times_s.shape} does not match epoch time axis {(epochs.shape[1],)}"
            )
    return SubjectBundle(
        subject_label=subject_label(subject_id),
        subject_id=subject_id,
        channel_names=channel_names,
        sfreq_hz=sfreq_hz,
        times_s=times_s,
        epochs=epochs,
        events=events,
        source_set_relpath=source_relpaths[0] if source_relpaths else "",
        source_eventlist_relpath=source_relpaths[-1] if source_relpaths else "",
    )


def braininvaders_event_frame(
    *,
    subject_id: int,
    subject_label_str: str,
    session_label: str,
    run_label: str,
    sample_index: np.ndarray,
    stimulus_code: np.ndarray,
    source_file: str,
    extra_columns: dict[str, Any] | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "subject_id": int(subject_id),
            "subject_label": subject_label_str,
            "session_label": session_label,
            "run_label": run_label,
            "sample_index": np.asarray(sample_index, dtype=int),
            "stimulus_code": np.asarray(stimulus_code, dtype=int),
            "condition": rarity_condition_labels(np.asarray(stimulus_code, dtype=int)),
            "source_file": source_file,
        }
    )
    if extra_columns:
        for key, value in extra_columns.items():
            frame[key] = value
    return frame


def prepare_bi2013a(output_root: Path, config: DatasetConfig) -> list[SubjectBundle]:
    output_dir = output_root / config.key
    source_root = output_dir / "source"
    archive_path = download_file(BI2013A_URL, source_root / "archives" / "subject08.zip")
    members = [f"subject08/Session1/{idx}.mat" for idx in range(1, 5)]
    extracted = extract_members(archive_path, source_root, members)

    epochs_parts: list[np.ndarray] = []
    event_frames: list[pd.DataFrame] = []
    relpaths: list[str] = []
    times_s_ref: np.ndarray | None = None
    for path in sorted(extracted):
        mat = loadmat(path, squeeze_me=True, struct_as_record=False)
        data = np.asarray(mat["data"]).T
        eeg = data[:16, :].astype(np.float32)
        stim = data[16, :]
        onsets = positive_rising_edges(stim)
        epochs, kept_onsets, times_s = extract_epochs(eeg, onsets, 512.0)
        times_s_ref = times_s if times_s_ref is None else times_s_ref
        codes = stim[kept_onsets].astype(int)
        run_label = path.stem
        relpath = str(path.relative_to(output_dir))
        epochs_parts.append(epochs)
        event_frames.append(
            braininvaders_event_frame(
                subject_id=8,
                subject_label_str=subject_label(8),
                session_label="Session1",
                run_label=run_label,
                sample_index=kept_onsets,
                stimulus_code=codes,
                source_file=relpath,
            )
        )
        relpaths.append(relpath)

    return [
        finalize_subject_bundle(
            subject_id=8,
            channel_names=BI2013A_CHANNELS,
            sfreq_hz=512.0,
            epochs_parts=epochs_parts,
            event_frames=event_frames,
            source_relpaths=relpaths,
            times_s=times_s_ref,
        )
    ]


def prepare_bi2014a(output_root: Path, config: DatasetConfig) -> list[SubjectBundle]:
    output_dir = output_root / config.key
    source_root = output_dir / "source"
    archive_path = download_file(BI2014A_URL, source_root / "archives" / "subject_01.zip")
    extracted = extract_members(archive_path, source_root, ["subject_01.mat"])
    path = extracted[0]

    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    data = np.asarray(mat["samples"]).T
    eeg = (data[1:17, :] * 1e-6).astype(np.float32)
    stim = data[-1, :]
    onsets = positive_rising_edges(stim)
    epochs, kept_onsets, times_s = extract_epochs(eeg, onsets, 512.0)
    codes = stim[kept_onsets].astype(int)
    relpath = str(path.relative_to(output_dir))
    frame = braininvaders_event_frame(
        subject_id=1,
        subject_label_str=subject_label(1),
        session_label="session_01",
        run_label="run_01",
        sample_index=kept_onsets,
        stimulus_code=codes,
        source_file=relpath,
    )

    return [
        finalize_subject_bundle(
            subject_id=1,
            channel_names=BI2014A_CHANNELS,
            sfreq_hz=512.0,
            epochs_parts=[epochs],
            event_frames=[frame],
            source_relpaths=[relpath],
            times_s=times_s,
        )
    ]


def prepare_bi2014b(output_root: Path, config: DatasetConfig) -> list[SubjectBundle]:
    output_dir = output_root / config.key
    source_root = output_dir / "source"
    archive_path = download_file(BI2014B_URL, source_root / "archives" / "group_01_mat.zip")
    extracted = extract_members(archive_path, source_root, ["group_01_sujet_01.mat"])
    path = extracted[0]

    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    data = np.asarray(mat["samples"]).T
    eeg = (data[1:33, :] * 1e-6).astype(np.float32)
    stim = data[-1, :]
    onsets = positive_rising_edges(stim)
    epochs, kept_onsets, times_s = extract_epochs(eeg, onsets, 512.0)
    codes = stim[kept_onsets].astype(int)
    relpath = str(path.relative_to(output_dir))
    frame = braininvaders_event_frame(
        subject_id=1,
        subject_label_str=subject_label(1),
        session_label="group_01",
        run_label="subject_01",
        sample_index=kept_onsets,
        stimulus_code=codes,
        source_file=relpath,
    )

    return [
        finalize_subject_bundle(
            subject_id=1,
            channel_names=BI32_CHANNELS,
            sfreq_hz=512.0,
            epochs_parts=[epochs],
            event_frames=[frame],
            source_relpaths=[relpath],
            times_s=times_s,
        )
    ]


def prepare_bi2015a(output_root: Path, config: DatasetConfig) -> list[SubjectBundle]:
    output_dir = output_root / config.key
    source_root = output_dir / "source"
    archive_path = download_file(BI2015A_URL, source_root / "archives" / "subject_01_mat.zip")
    members = [f"subject_01_session_0{idx}.mat" for idx in range(1, 4)]
    extracted = extract_members(archive_path, source_root, members)

    epochs_parts: list[np.ndarray] = []
    event_frames: list[pd.DataFrame] = []
    relpaths: list[str] = []
    times_s_ref: np.ndarray | None = None
    for path in sorted(extracted):
        mat = loadmat(path, squeeze_me=True, struct_as_record=False)
        data = np.asarray(mat["DATA"]).T
        eeg = (data[1:33, :] * 1e-6).astype(np.float32)
        stim = data[-2, :] + data[-1, :]
        onsets = positive_rising_edges(stim)
        epochs, kept_onsets, times_s = extract_epochs(eeg, onsets, 512.0)
        times_s_ref = times_s if times_s_ref is None else times_s_ref
        codes = stim[kept_onsets].astype(int)
        relpath = str(path.relative_to(output_dir))
        epochs_parts.append(epochs)
        event_frames.append(
            braininvaders_event_frame(
                subject_id=1,
                subject_label_str=subject_label(1),
                session_label=path.stem.replace("subject_01_", ""),
                run_label=path.stem,
                sample_index=kept_onsets,
                stimulus_code=codes,
                source_file=relpath,
            )
        )
        relpaths.append(relpath)

    return [
        finalize_subject_bundle(
            subject_id=1,
            channel_names=BI32_CHANNELS,
            sfreq_hz=512.0,
            epochs_parts=epochs_parts,
            event_frames=event_frames,
            source_relpaths=relpaths,
            times_s=times_s_ref,
        )
    ]


def prepare_bi2015b(output_root: Path, config: DatasetConfig) -> list[SubjectBundle]:
    output_dir = output_root / config.key
    source_root = output_dir / "source"
    archive_path = download_file(BI2015B_URL, source_root / "archives" / "group_01_mat.zip")
    members = [f"group_01_s{idx}.mat" for idx in range(1, 5)]
    extracted = extract_members(archive_path, source_root, members)

    epochs_parts: list[np.ndarray] = []
    event_frames: list[pd.DataFrame] = []
    relpaths: list[str] = []
    times_s_ref: np.ndarray | None = None
    for path in sorted(extracted):
        mat = loadmat(path, squeeze_me=True, struct_as_record=False)
        data = np.asarray(mat["mat_data"]).T
        eeg = (data[1:33, :] * 1e-6).astype(np.float32)
        raw_stim = np.asarray(data[-1, :]).astype(int)
        stim = raw_stim.copy()
        stim[(stim >= 60) & (stim <= 85)] = 2
        stim[(stim >= 20) & (stim <= 45)] = 1
        onsets = code_rising_edges(stim, valid_codes={1, 2})
        epochs, kept_onsets, times_s = extract_epochs(eeg, onsets, 512.0)
        times_s_ref = times_s if times_s_ref is None else times_s_ref
        codes = stim[kept_onsets].astype(int)
        source_codes = raw_stim[kept_onsets].astype(int)
        relpath = str(path.relative_to(output_dir))
        frame = pd.DataFrame(
            {
                "subject_id": 1,
                "subject_label": subject_label(1),
                "session_label": "group_01",
                "run_label": path.stem,
                "sample_index": kept_onsets.astype(int),
                "stimulus_code": codes,
                "source_stimulus_code": source_codes,
                "condition": ["target" if int(code) == 2 else "non_target" for code in codes],
                "source_file": relpath,
            }
        )
        epochs_parts.append(epochs)
        event_frames.append(frame)
        relpaths.append(relpath)

    return [
        finalize_subject_bundle(
            subject_id=1,
            channel_names=BI32_CHANNELS,
            sfreq_hz=512.0,
            epochs_parts=epochs_parts,
            event_frames=event_frames,
            source_relpaths=relpaths,
            times_s=times_s_ref,
        )
    ]


def prepare_cattan2019_vr(output_root: Path, config: DatasetConfig) -> list[SubjectBundle]:
    output_dir = output_root / config.key
    source_root = output_dir / "source"
    vr_path = download_file(CATTAN_VR_URL, source_root / "subject_01_VR.mat")
    pc_path = download_file(CATTAN_PC_URL, source_root / "subject_01_PC.mat")

    epochs_parts: list[np.ndarray] = []
    event_frames: list[pd.DataFrame] = []
    relpaths: list[str] = []
    times_s_ref: np.ndarray | None = None
    for path, display_mode in [(pc_path, "pc"), (vr_path, "vr")]:
        mat = loadmat(path, squeeze_me=True, struct_as_record=False)
        data = np.asarray(mat["data"])
        eeg = (data[:, 1:17].T * 1e-6).astype(np.float32)
        stim = (2 * data[:, 18] + data[:, 19]).astype(int)
        onsets = positive_rising_edges(stim)
        epochs, kept_onsets, times_s = extract_epochs(eeg, onsets, 512.0)
        times_s_ref = times_s if times_s_ref is None else times_s_ref
        codes = stim[kept_onsets].astype(int)
        relpath = str(path.relative_to(output_dir))
        frame = pd.DataFrame(
            {
                "subject_id": 1,
                "subject_label": subject_label(1),
                "session_label": display_mode,
                "run_label": path.stem,
                "display_mode": display_mode,
                "sample_index": kept_onsets.astype(int),
                "stimulus_code": codes,
                "condition": rarity_condition_labels(codes),
                "source_file": relpath,
            }
        )
        epochs_parts.append(epochs)
        event_frames.append(frame)
        relpaths.append(relpath)

    return [
        finalize_subject_bundle(
            subject_id=1,
            channel_names=CATTAN_CHANNELS,
            sfreq_hz=512.0,
            epochs_parts=epochs_parts,
            event_frames=event_frames,
            source_relpaths=relpaths,
            times_s=times_s_ref,
        )
    ]


def prepare_bnci_008_2014(output_root: Path, config: DatasetConfig) -> list[SubjectBundle]:
    output_dir = output_root / config.key
    source_root = output_dir / "source"
    mat_path = download_file(BNCI_008_A01_URL, source_root / "A01.mat")
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    run_container = mat["data"]
    try:
        runs = list(run_container)
    except TypeError:
        runs = [run_container]

    epochs_parts: list[np.ndarray] = []
    event_frames: list[pd.DataFrame] = []
    relpath = str(mat_path.relative_to(output_dir))
    times_s_ref: np.ndarray | None = None
    for run_idx, run in enumerate(runs, start=1):
        eeg = (np.asarray(run.X).T * 1e-6).astype(np.float32)
        stim = np.asarray(run.y_stim).astype(int)
        labels = np.asarray(run.y).astype(int)
        trial_starts = np.asarray(run.trial).astype(int)
        onsets = positive_rising_edges(stim)
        epochs, kept_onsets, times_s = extract_epochs(eeg, onsets, 256.0)
        times_s_ref = times_s if times_s_ref is None else times_s_ref
        valid = kept_onsets >= int(trial_starts[0])
        epochs = epochs[valid, :, :]
        kept_onsets = kept_onsets[valid]
        flash_codes = stim[kept_onsets].astype(int)
        label_codes = labels[kept_onsets].astype(int)
        trial_block_index = np.searchsorted(trial_starts, kept_onsets, side="right")
        trial_block_index = np.clip(trial_block_index, 1, len(trial_starts)).astype(int)
        within_trial = pd.Series(trial_block_index).groupby(trial_block_index).cumcount().to_numpy() + 1
        frame = pd.DataFrame(
            {
                "subject_id": 1,
                "subject_label": subject_label(1),
                "session_label": "session_01",
                "run_label": f"run_{run_idx:02d}",
                "trial_block_index": trial_block_index.astype(int),
                "flash_index_within_trial": within_trial.astype(int),
                "sample_index": kept_onsets.astype(int),
                "stimulus_code": flash_codes,
                "target_label_code": label_codes,
                "condition": ["target" if int(code) == 2 else "non_target" for code in label_codes],
                "source_file": relpath,
            }
        )
        epochs_parts.append(epochs)
        event_frames.append(frame)

    return [
        finalize_subject_bundle(
            subject_id=1,
            channel_names=[str(name) for name in np.asarray(runs[0].channels).tolist()],
            sfreq_hz=256.0,
            epochs_parts=epochs_parts,
            event_frames=event_frames,
            source_relpaths=[relpath],
            times_s=times_s_ref,
        )
    ]


def prepare_bigp3bci(output_root: Path, config: DatasetConfig) -> list[SubjectBundle]:
    output_dir = output_root / config.key
    source_root = output_dir / "source" / "bigP3BCI-data" / "StudyA" / "A_01" / "SE001" / "Train" / "CB"

    epochs_parts: list[np.ndarray] = []
    event_frames: list[pd.DataFrame] = []
    relpaths: list[str] = []
    channel_names: list[str] | None = None
    sfreq_hz: float | None = None
    times_s_ref: np.ndarray | None = None

    for file_name in BIGP3_FILES:
        file_path = download_file(f"{BIGP3_BASE}/{file_name}", source_root / file_name)
        raw = mne.io.read_raw_edf(file_path, preload=True, infer_types=True, verbose="ERROR")
        eeg_channels = [name for name in raw.ch_names if name.startswith("EEG_")]
        eeg = raw.get_data(picks=eeg_channels).astype(np.float32)
        stim_begin = raw.get_data(picks=["StimulusBegin"])[0]
        stim_type = round_channel(raw.get_data(picks=["StimulusType"])[0])
        stim_code = round_channel(raw.get_data(picks=["StimulusCode"])[0])
        current_target = round_channel(raw.get_data(picks=["CurrentTarget"])[0])
        onsets = np.flatnonzero((stim_begin > 0.5) & np.r_[True, stim_begin[:-1] <= 0.5])
        epochs, kept_onsets, times_s = extract_epochs(eeg, onsets, float(raw.info["sfreq"]))
        times_s_ref = times_s if times_s_ref is None else times_s_ref

        relpath = str(file_path.relative_to(output_dir))
        frame = pd.DataFrame(
            {
                "subject_id": 1,
                "subject_label": subject_label(1),
                "study_subject_label": "A_01",
                "session_label": "SE001",
                "run_label": file_path.stem.split("_")[-1],
                "study_id": "StudyA",
                "speller_condition": "CB",
                "phase_label": "train",
                "sample_index": kept_onsets.astype(int),
                "stimulus_code": stim_code[kept_onsets].astype(int),
                "current_target": current_target[kept_onsets].astype(int),
                "stimulus_type": stim_type[kept_onsets].astype(int),
                "condition": [
                    "target" if int(code) == 1 else "non_target"
                    for code in stim_type[kept_onsets].astype(int)
                ],
                "source_file": relpath,
            }
        )

        channel_names = [name.removeprefix("EEG_") for name in eeg_channels]
        sfreq_hz = float(raw.info["sfreq"])
        epochs_parts.append(epochs)
        event_frames.append(frame)
        relpaths.append(relpath)

    if channel_names is None or sfreq_hz is None:
        raise RuntimeError("Failed to materialize bigP3BCI source files.")

    return [
        finalize_subject_bundle(
            subject_id=1,
            channel_names=channel_names,
            sfreq_hz=sfreq_hz,
            epochs_parts=epochs_parts,
            event_frames=event_frames,
            source_relpaths=relpaths,
            times_s=times_s_ref,
        )
    ]


DATASETS: dict[str, DatasetConfig] = {
    "bi2013a_public": DatasetConfig(
        key="bi2013a_public",
        component="BI2013a",
        source_component="https://zenodo.org/record/2669187",
        source_processing_scripts=MOABB_BI_LOADER,
        reader_docs=SCIPY_LOADMAT_DOCS,
        preferred_channels=("Pz", "P3", "Oz", "O2", "Cz"),
        recommended_sort_columns=("condition", "run_label", "stimulus_code", "sample_index", "epoch_index"),
        official_source_examples={
            "loader_note": "MOABB loads BI2013a from loadmat(...)[\"data\"].T and uses the last row as the event channel.",
            "subset_note": "This bundle stores subject 08, Session1, runs 1-4 from the public ZIP archive.",
        },
        prepare=prepare_bi2013a,
    ),
    "bi2014a_public": DatasetConfig(
        key="bi2014a_public",
        component="BI2014a",
        source_component="https://zenodo.org/record/3266223",
        source_processing_scripts=MOABB_BI_LOADER,
        reader_docs=SCIPY_LOADMAT_DOCS,
        preferred_channels=("Pz", "P3", "Oz", "O2", "Cz"),
        recommended_sort_columns=("condition", "stimulus_code", "sample_index", "epoch_index"),
        official_source_examples={
            "loader_note": "MOABB loads BI2014a from loadmat(...)[\"samples\"].T, takes rows 1:17 as EEG, and the last row as stimulus codes.",
            "subset_note": "This bundle stores subject 01 from the public subject ZIP archive.",
        },
        prepare=prepare_bi2014a,
    ),
    "bi2014b_public": DatasetConfig(
        key="bi2014b_public",
        component="BI2014b",
        source_component="https://zenodo.org/record/3267302",
        source_processing_scripts=MOABB_BI_LOADER,
        reader_docs=SCIPY_LOADMAT_DOCS,
        preferred_channels=("Pz", "CP1", "CP2", "PO7", "PO8"),
        recommended_sort_columns=("condition", "stimulus_code", "sample_index", "epoch_index"),
        official_source_examples={
            "loader_note": "MOABB loads BI2014b from loadmat(...)[\"samples\"].T and selects subject-specific EEG rows from the pair recording.",
            "subset_note": "This bundle stores group 01, subject 01 from the public group ZIP archive.",
        },
        prepare=prepare_bi2014b,
    ),
    "bi2015a_public": DatasetConfig(
        key="bi2015a_public",
        component="BI2015a",
        source_component="https://zenodo.org/record/3266930",
        source_processing_scripts=MOABB_BI_LOADER,
        reader_docs=SCIPY_LOADMAT_DOCS,
        preferred_channels=("Pz", "CP1", "CP2", "PO7", "PO8"),
        recommended_sort_columns=("condition", "session_label", "run_label", "stimulus_code", "sample_index", "epoch_index"),
        official_source_examples={
            "loader_note": "MOABB loads BI2015a from loadmat(...)[\"DATA\"].T and combines the last two trigger rows into one stimulus channel.",
            "subset_note": "This bundle stores subject 01 sessions 01-03 from the public subject ZIP archive.",
        },
        prepare=prepare_bi2015a,
    ),
    "bi2015b_public": DatasetConfig(
        key="bi2015b_public",
        component="BI2015b",
        source_component="https://zenodo.org/record/3268762",
        source_processing_scripts=MOABB_BI_LOADER,
        reader_docs=SCIPY_LOADMAT_DOCS,
        preferred_channels=("Pz", "CP1", "CP2", "PO7", "PO8"),
        recommended_sort_columns=(
            "condition",
            "run_label",
            "stimulus_code",
            "source_stimulus_code",
            "sample_index",
            "epoch_index",
        ),
        official_source_examples={
            "loader_note": "MOABB loads BI2015b from loadmat(...)[\"mat_data\"].T and remaps source trigger ranges 20-45 and 60-85 to non-target and target events.",
            "subset_note": "This bundle stores group 01 subject 01 runs s1-s4 from the public group ZIP archive.",
        },
        prepare=prepare_bi2015b,
    ),
    "cattan2019_vr_public": DatasetConfig(
        key="cattan2019_vr_public",
        component="Cattan2019 VR/PC",
        source_component="https://zenodo.org/record/2605205",
        source_processing_scripts=MOABB_BI_LOADER,
        reader_docs=SCIPY_LOADMAT_DOCS,
        preferred_channels=("Pz", "P3", "Oz", "O2", "Cz"),
        recommended_sort_columns=("condition", "display_mode", "session_label", "stimulus_code", "sample_index", "epoch_index"),
        official_source_examples={
            "loader_note": "MOABB loads the VR dataset from MAT files and combines two trigger columns into one event channel.",
            "subset_note": "This bundle stores subject 01 in both PC and VR modes from the public MAT releases.",
        },
        prepare=prepare_cattan2019_vr,
    ),
    "bnci_008_2014_public": DatasetConfig(
        key="bnci_008_2014_public",
        component="BNCI 008-2014",
        source_component="https://bnci-horizon-2020.eu/database/data-sets",
        source_processing_scripts=MOABB_BNCI_LOADER,
        reader_docs=SCIPY_LOADMAT_DOCS,
        preferred_channels=("Pz", "Oz", "PO7", "PO8", "Cz"),
        recommended_sort_columns=(
            "condition",
            "trial_block_index",
            "flash_index_within_trial",
            "stimulus_code",
            "sample_index",
            "epoch_index",
        ),
        official_source_examples={
            "loader_note": "MOABB converts BNCI 008-2014 from run.X, run.y, and run.y_stim into flash-locked P300 events.",
            "subset_note": "This bundle stores subject A01 from the official BNCI MAT file.",
        },
        prepare=prepare_bnci_008_2014,
    ),
    "bigp3bci_studya_public": DatasetConfig(
        key="bigp3bci_studya_public",
        component="bigP3BCI StudyA",
        source_component="https://physionet.org/content/bigp3bci/1.0.0/",
        source_processing_scripts="https://physionet.org/content/bigp3bci/1.0.0/",
        reader_docs=MNE_EDF_DOCS,
        preferred_channels=("Pz", "POz", "CPz", "Oz", "PO7"),
        recommended_sort_columns=(
            "condition",
            "current_target",
            "run_label",
            "stimulus_code",
            "sample_index",
            "epoch_index",
        ),
        official_source_examples={
            "loader_note": "The PhysioNet release stores EEG plus event channels such as StimulusBegin, StimulusType, StimulusCode, and CurrentTarget inside EDF+ files.",
            "subset_note": "This bundle stores StudyA / A_01 / SE001 / Train / CB files 01-05.",
        },
        prepare=prepare_bigp3bci,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where dataset bundles should be written.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS.keys()),
        choices=sorted(DATASETS.keys()),
        help="Dataset bundle keys to materialize.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild bundles even if HDF5/CSV/JSON files already exist.",
    )
    return parser.parse_args()


def bundle_exists(output_dir: Path) -> bool:
    return all((output_dir / name).is_file() for name in ["epochs.hdf5", "events.csv", "metadata.json"])


def main() -> None:
    args = parse_args()
    mne.set_log_level("ERROR")

    for dataset_key in args.datasets:
        config = DATASETS[dataset_key]
        output_dir = args.output_root / config.key
        if bundle_exists(output_dir) and not args.force:
            print(f"[skip] {config.key} already exists")
            continue

        print(f"[build] {config.key}")
        bundles = config.prepare(args.output_root, config)
        written_dir = write_dataset_bundle(output_dir, config, bundles)
        print(f"[done] {written_dir}")


if __name__ == "__main__":
    main()
