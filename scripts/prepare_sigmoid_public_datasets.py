#!/usr/bin/env python3
"""
Prepare representative public datasets that can produce sigmoid-like ERP images.

The importers here focus on sources with either eye-movement events or
response-locked reaction-time metadata.  They intentionally materialize a small,
reproducible public subset by default, because several upstream datasets are
large enough that full mirroring would be inappropriate for notebook iteration.

Required Python packages:
    mne, h5py, numpy, pandas
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import h5py
import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat

from prepare_public_shortlist_datasets import (
    DatasetConfig,
    SubjectBundle,
    download_file,
    extract_epochs,
    finalize_subject_bundle,
    subject_label,
    write_dataset_bundle,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "notebooks" / "datasets"
OPENNEURO_S3_ROOT = "https://s3.amazonaws.com/openneuro.org"
MNE_BRAINVISION_DOCS = "https://mne.tools/stable/generated/mne.io.read_raw_brainvision.html"
MNE_EEGLAB_RAW_DOCS = "https://mne.tools/stable/generated/mne.io.read_raw_eeglab.html"
MNE_EDF_DOCS = "https://mne.tools/stable/generated/mne.io.read_raw_edf.html"

RESPONSE_TMIN_S = -1.3
RESPONSE_DURATION_S = 1.6
SACCADE_TMIN_S = -0.5
SACCADE_DURATION_S = 2.5
FIXATION_TMIN_S = -0.5
FIXATION_DURATION_S = 2.5

CENTRAL_RESPONSE_CHANNELS = ("Cz", "CPz", "C3", "C4", "FCz")
EGI_FACE_CHANNELS = ("E8", "E14", "E21", "E25", "E126", "E127", "E128", "E11", "E6")
ROAMM_PREFERRED_CHANNELS = ("Fp1", "Fp2", "AF7", "AF8", "F7", "F8", "FCz", "Cz")
ROAMM_RUN1_URL = "https://files.us.osf.io/v1/resources/kmvgb/providers/osfstorage/697912d765bfc7f516589e97"
SACCADE_VR_EYE_TABLE_URL = "https://files.us.osf.io/v1/resources/trfjw/providers/osfstorage/65cb3cdd9b32ca118d97f217"
SACCADE_VR_SET_URL = "https://files.us.osf.io/v1/resources/trfjw/providers/osfstorage/65ca50839b32ca10a997f14b"
SACCADE_VR_FDT_URL = "https://files.us.osf.io/v1/resources/trfjw/providers/osfstorage/65ca507a435c450fd8da7739"
SACCADE_VR_PREFERRED_CHANNELS = ("FP1", "FPz", "FP2", "F7", "F8", "Fz", "Cz", "Oz")
SACCADE_VR_MAX_EVENTS = 1200
ZUCO2_SOURCE_URL = "https://osf.io/2urht/"
ZUCO2_RAW_FILES = {
    "YAG_NR1_EEG.mat": "https://osf.io/download/xrkb6/",
    "YAG_NR1_ET.mat": "https://osf.io/download/bfn6g/",
}
ZUCO2_PREFERRED_CHANNELS = (
    "E1",
    "E8",
    "E14",
    "E21",
    "E25",
    "E32",
    "E126",
    "E127",
    "E128",
    "E11",
    "E6",
    "E62",
    "E75",
    "E55",
    "E72",
    "E106",
)
RACCOONS_BASE_URL = "https://webdav.data.ru.nl/cls/eeg_et_sentence_reading_dsc_556_v1"
RACCOONS_FILES = {
    "EEG003.mat": f"{RACCOONS_BASE_URL}/EEG/Merged/EEG003.mat",
    "ET_fix_data.tsv": f"{RACCOONS_BASE_URL}/eyetracking/ET_fix_data.tsv",
    "ET_word_data.tsv": f"{RACCOONS_BASE_URL}/eyetracking/ET_word_data.tsv",
    "N400.tsv": f"{RACCOONS_BASE_URL}/EEG/N400.tsv",
    "words.tsv": f"{RACCOONS_BASE_URL}/Stimuli/words.tsv",
}
RACCOONS_PREFERRED_CHANNELS = ("EOG-up", "EOG-down", "EOG-left", "EOG-right", "Fp2", "Fz", "Cz", "Pz", "Oz")
RACCOONS_MAX_EVENTS = 1800
UNFOLD_ZIP_URL = "https://osf.io/download/efs46/"
UNFOLD_SET_MEMBER = "face_saccades_opendata_fig10.set"
UNFOLD_PREFERRED_CHANNELS = ("Fp1", "Fpz", "Fp2", "F7", "F8", "Fz", "Cz", "Oz")
UNFOLD_MAX_EVENTS = 1800
VISUOMOTOR_SOURCE_URL = "https://osf.io/cfdsz/"
EEGET_RSOD_SOURCE_URL = "https://figshare.com/articles/dataset/EEGET-RSOD/26943565"
KILO_WORD_URLS = {
    "KWORD_ERP_LEXICAL_DECISION_DGMH2015.txt": "https://osf.io/download/6gpef/",
    "KWORD_VARIABLES_DGMH2015.txt": "https://osf.io/download/nqc7y/",
}
CONFIDENCE_OPENNEURO_ROOT = "https://s3.amazonaws.com/openneuro.org/ds002739"
CONFIDENCE_RUNS = (1, 2)
VISUOMOTOR_RAW_FILES = {
    "sub1.cdt": "https://osf.io/download/66278a9ad8960707471b1861/",
    "sub1.cdt.ceo": "https://osf.io/download/66278a4380d25c068ef91c1c/",
    "sub1.cdt.dpa": "https://osf.io/download/66278a4cc5851a0698f672d9/",
}
VISUOMOTOR_SMALL_FILES = {
    "readme.pdf": "https://osf.io/download/hdnvw/",
    "EYE_PRE.zip": "https://osf.io/download/4hxt7/",
    "BEH_PRE.zip": "https://osf.io/download/ehgbr/",
    "BEH_RAW.zip": "https://osf.io/download/s6f8p/",
}
VISUOMOTOR_MAX_EVENTS = 1800
EEGET_FILES = {
    "EEG.zip": "https://ndownloader.figshare.com/files/49020166",
    "ET.zip": "https://ndownloader.figshare.com/files/49020133",
    "label.zip": "https://ndownloader.figshare.com/files/49044883",
    "Time Synchronization Data.xlsx": "https://ndownloader.figshare.com/files/49653405",
}
EEGET_MAX_EVENTS = 1800


@dataclass(frozen=True)
class ImportContext:
    output_root: Path
    max_subjects: int
    max_runs: int


def openneuro_url(dataset_id: str, relpath: str) -> str:
    return f"{OPENNEURO_S3_ROOT}/{dataset_id}/{relpath}"


def download_openneuro_file(dataset_id: str, relpath: str, source_root: Path) -> Path:
    return download_file(openneuro_url(dataset_id, relpath), source_root / relpath)


def safe_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", encoding="utf-8-sig", na_values=["n/a", "N/A", ""])


def eeg_channel_names(raw: mne.io.BaseRaw) -> list[str]:
    picks = mne.pick_types(raw.info, eeg=True, eog=True, stim=False, misc=False, exclude=[])
    if len(picks) == 0:
        picks = np.arange(len(raw.ch_names))
    return [raw.ch_names[int(idx)] for idx in picks]


def response_epoch_bundle(
    *,
    subject_id: int,
    subject_label_str: str,
    raw: mne.io.BaseRaw,
    events: pd.DataFrame,
    output_dir: Path,
    source_paths: list[Path],
) -> SubjectBundle:
    channel_names = eeg_channel_names(raw)
    eeg = raw.get_data(picks=channel_names).astype(np.float32)
    epochs, kept_onsets, times_s = extract_epochs(
        eeg,
        events["sample_index"].to_numpy(dtype=int),
        float(raw.info["sfreq"]),
        tmin_s=RESPONSE_TMIN_S,
        duration_s=RESPONSE_DURATION_S,
    )
    keep = events["sample_index"].isin(set(int(v) for v in kept_onsets))
    kept = events.loc[keep].copy().reset_index(drop=True)
    if kept.empty:
        raise RuntimeError(f"No response epochs survived boundary checks for {subject_label_str}.")

    relpaths = [safe_relpath(path, output_dir) for path in source_paths]
    return finalize_subject_bundle(
        subject_id=subject_id,
        channel_names=channel_names,
        sfreq_hz=float(raw.info["sfreq"]),
        epochs_parts=[epochs],
        event_frames=[kept],
        source_relpaths=relpaths,
        times_s=times_s,
    )


def finite_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def prepare_nencki_symfonia_srt(output_root: Path, config: DatasetConfig) -> list[SubjectBundle]:
    dataset_id = "ds004621"
    output_dir = output_root / config.key
    source_root = output_dir / "source"
    subjects = ["sub-01"]
    bundles: list[SubjectBundle] = []

    for subject in subjects:
        numeric_subject_id = int(subject.replace("sub-", ""))
        bundle_subject_label = subject_label(numeric_subject_id)
        eeg_dir = f"{subject}/eeg"
        stem = f"{subject}_task-srt"
        relpaths = [
            f"{eeg_dir}/{stem}_eeg.vhdr",
            f"{eeg_dir}/{stem}_eeg.vmrk",
            f"{eeg_dir}/{stem}_eeg.eeg",
            f"{eeg_dir}/{stem}_events.tsv",
            f"{eeg_dir}/{stem}_channels.tsv",
        ]
        paths = [download_openneuro_file(dataset_id, relpath, source_root) for relpath in relpaths]
        vhdr_path = source_root / relpaths[0]
        events_path = source_root / relpaths[3]

        raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose="ERROR")
        source_events = read_tsv(events_path)
        rows: list[dict[str, object]] = []
        for idx, row in source_events.iterrows():
            if str(row.get("trial_type", "")).lower() != "stimulus":
                continue
            stimulus_onset = finite_float(row.get("onset"))
            if stimulus_onset is None:
                continue
            next_events = source_events.iloc[idx + 1 :]
            next_response = next_events[next_events["trial_type"].astype(str).str.lower() == "response"]
            if next_response.empty:
                continue
            response = next_response.iloc[0]
            response_onset = finite_float(response.get("onset"))
            if response_onset is None:
                continue
            reaction_time_ms = (response_onset - stimulus_onset) * 1000.0
            if reaction_time_ms <= 0 or reaction_time_ms > 1300:
                continue
            rows.append(
                {
                    "dataset_key": config.key,
                    "component": config.component,
                    "subject_id": numeric_subject_id,
                    "subject_label": bundle_subject_label,
                    "source_subject_label": subject,
                    "session_label": "srt",
                    "run_label": "task-srt",
                    "condition": "simple_reaction",
                    "trial_type": "response",
                    "stimulus_code": str(row.get("event_type", "")),
                    "response_code": str(response.get("event_type", "")),
                    "stimulus_onset_s": stimulus_onset,
                    "response_onset_s": response_onset,
                    "reaction_time_ms": reaction_time_ms,
                    "sample_index": int(round(response_onset * float(raw.info["sfreq"]))),
                    "source_file": safe_relpath(vhdr_path, output_dir),
                }
            )

        events = pd.DataFrame(rows)
        bundles.append(
            response_epoch_bundle(
                subject_id=numeric_subject_id,
                subject_label_str=bundle_subject_label,
                raw=raw,
                events=events,
                output_dir=output_dir,
                source_paths=paths,
            )
        )

    return bundles


def prepare_openneuro_gonogo_ds002680(output_root: Path, config: DatasetConfig) -> list[SubjectBundle]:
    dataset_id = "ds002680"
    output_dir = output_root / config.key
    source_root = output_dir / "source"
    subject = "sub-002"
    session = "ses-01"
    run_ids = [1, 2]
    epochs_parts: list[np.ndarray] = []
    event_frames: list[pd.DataFrame] = []
    source_relpaths: list[str] = []
    channel_names_ref: list[str] | None = None
    sfreq_ref: float | None = None
    times_s_ref: np.ndarray | None = None

    for run_id in run_ids:
        eeg_dir = f"{subject}/{session}/eeg"
        stem = f"{subject}_{session}_task-gonogo_run-{run_id}"
        relpaths = [
            f"{eeg_dir}/{stem}_eeg.set",
            f"{eeg_dir}/{stem}_events.tsv",
            f"{eeg_dir}/{stem}_channels.tsv",
        ]
        set_path, events_path, _channels_path = [
            download_openneuro_file(dataset_id, relpath, source_root) for relpath in relpaths
        ]
        raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")
        channel_names = eeg_channel_names(raw)
        if channel_names_ref is None:
            channel_names_ref = channel_names
            sfreq_ref = float(raw.info["sfreq"])
        elif channel_names != channel_names_ref:
            raise RuntimeError(f"Channel mismatch across {config.key} runs.")

        source_events = read_tsv(events_path)
        rows: list[dict[str, object]] = []
        for idx, row in source_events.iterrows():
            if str(row.get("trial_type", "")).lower() != "stimulus":
                continue
            rt_ms = finite_float(row.get("response_time"))
            if rt_ms is None or rt_ms <= 0 or rt_ms > 1300:
                continue
            stimulus_onset = finite_float(row.get("onset"))
            if stimulus_onset is None:
                continue
            response_onset = stimulus_onset + rt_ms / 1000.0
            next_events = source_events.iloc[idx + 1 :]
            next_response = next_events[next_events["trial_type"].astype(str).str.lower() == "response"]
            accuracy = ""
            if not next_response.empty:
                response_row = next_response.iloc[0]
                response_row_onset = finite_float(response_row.get("onset"))
                if response_row_onset is not None and abs(response_row_onset - response_onset) < 0.02:
                    accuracy = str(response_row.get("value", ""))
            rows.append(
                {
                    "dataset_key": config.key,
                    "component": config.component,
                    "subject_id": 2,
                    "subject_label": subject,
                    "session_label": session,
                    "run_label": f"run-{run_id}",
                    "condition": str(row.get("value", "")),
                    "trial_type": str(row.get("trial_type", "")),
                    "stim_file": str(row.get("stim_file", "")),
                    "accuracy": accuracy,
                    "stimulus_onset_s": stimulus_onset,
                    "response_onset_s": response_onset,
                    "reaction_time_ms": rt_ms,
                    "sample_index": int(round(response_onset * float(raw.info["sfreq"]))),
                    "source_file": safe_relpath(set_path, output_dir),
                }
            )

        events = pd.DataFrame(rows)
        eeg = raw.get_data(picks=channel_names).astype(np.float32)
        epochs, kept_onsets, times_s = extract_epochs(
            eeg,
            events["sample_index"].to_numpy(dtype=int),
            float(raw.info["sfreq"]),
            tmin_s=RESPONSE_TMIN_S,
            duration_s=RESPONSE_DURATION_S,
        )
        keep = events["sample_index"].isin(set(int(v) for v in kept_onsets))
        kept = events.loc[keep].copy().reset_index(drop=True)
        if kept.empty:
            continue
        epochs_parts.append(epochs)
        event_frames.append(kept)
        source_relpaths.extend([safe_relpath(set_path, output_dir), safe_relpath(events_path, output_dir)])
        times_s_ref = times_s if times_s_ref is None else times_s_ref

    if channel_names_ref is None or sfreq_ref is None or times_s_ref is None:
        raise RuntimeError(f"No valid runs were prepared for {config.key}.")

    return [
        finalize_subject_bundle(
            subject_id=2,
            channel_names=channel_names_ref,
            sfreq_hz=sfreq_ref,
            epochs_parts=epochs_parts,
            event_frames=event_frames,
            source_relpaths=source_relpaths,
            times_s=times_s_ref,
        )
    ]


def saccade_amplitudes_px(physio_path: Path, saccades: pd.DataFrame) -> np.ndarray:
    physio = pd.read_csv(
        physio_path,
        sep="\t",
        header=None,
        names=["time_s", "gaze_x", "gaze_y"],
        encoding="utf-8-sig",
    )
    time_s = physio["time_s"].to_numpy(dtype=np.float64)
    gaze_x = physio["gaze_x"].replace(0, np.nan).to_numpy(dtype=np.float64)
    gaze_y = physio["gaze_y"].replace(0, np.nan).to_numpy(dtype=np.float64)
    amplitudes: list[float] = []
    for onset_s, duration_s in zip(
        saccades["onset"].to_numpy(dtype=np.float64),
        saccades["duration"].to_numpy(dtype=np.float64),
        strict=False,
    ):
        start_idx = int(np.searchsorted(time_s, onset_s, side="left"))
        stop_idx = int(np.searchsorted(time_s, onset_s + duration_s, side="left"))
        if start_idx >= len(time_s) or stop_idx >= len(time_s):
            amplitudes.append(float("nan"))
            continue
        dx = gaze_x[stop_idx] - gaze_x[start_idx]
        dy = gaze_y[stop_idx] - gaze_y[start_idx]
        amplitudes.append(float(np.hypot(dx, dy)) if np.isfinite(dx) and np.isfinite(dy) else float("nan"))
    return np.asarray(amplitudes, dtype=np.float32)


def prepare_eegeyenet_saccades(output_root: Path, config: DatasetConfig) -> list[SubjectBundle]:
    dataset_id = "ds005872"
    output_dir = output_root / config.key
    source_root = output_dir / "source"
    subject = "sub-EP10"
    session = "ses-01"
    run = "run-01"
    eeg_dir = f"{subject}/{session}/eeg"
    stem = f"{subject}_{session}_task-dots_{run}"
    relpaths = [
        f"{eeg_dir}/{stem}_eeg.edf",
        f"{eeg_dir}/{stem}_events.tsv",
        f"{eeg_dir}/{stem}_channels.tsv",
        f"{eeg_dir}/{stem}_recording-eye1_physio.tsv",
        f"{eeg_dir}/{stem}_recording-eye1_physio.json",
        f"{eeg_dir}/{stem}_recording-eye1_physioevents.tsv",
    ]
    paths = [download_openneuro_file(dataset_id, relpath, source_root) for relpath in relpaths]
    edf_path = source_root / relpaths[0]
    events_path = source_root / relpaths[1]
    physio_path = source_root / relpaths[3]
    physioevents_path = source_root / relpaths[5]

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    channel_names = eeg_channel_names(raw)
    task_events = read_tsv(events_path)
    trial_events = task_events[task_events["trial_type"].astype(str) != "end_cue"].copy()
    trial_onsets = trial_events["onset"].to_numpy(dtype=np.float64)

    physioevents = read_tsv(physioevents_path)
    saccades = physioevents[physioevents["trial_type"].astype(str) == "saccade"].copy().reset_index(drop=True)
    fixations = physioevents[physioevents["trial_type"].astype(str) == "fixation"].copy().reset_index(drop=True)
    saccades["saccade_amplitude_px"] = saccade_amplitudes_px(physio_path, saccades)
    fixation_onsets = fixations["onset"].to_numpy(dtype=np.float64) if not fixations.empty else np.asarray([])
    trial_idx = np.searchsorted(trial_onsets, saccades["onset"].to_numpy(dtype=np.float64), side="right") - 1
    valid = trial_idx >= 0
    saccades = saccades.loc[valid].copy().reset_index(drop=True)
    trial_idx = trial_idx[valid]

    rows: list[dict[str, object]] = []
    for idx, row in saccades.iterrows():
        onset_s = float(row["onset"])
        duration_s = float(row["duration"])
        trial_row = trial_events.iloc[int(trial_idx[idx])]
        trial_onset_s = float(trial_row["onset"])
        previous_fixation_duration_ms = float("nan")
        if len(fixation_onsets):
            fixation_idx = int(np.searchsorted(fixation_onsets, onset_s, side="right") - 1)
            if 0 <= fixation_idx < len(fixations):
                previous_fixation_duration_ms = float(fixations.iloc[fixation_idx]["duration"]) * 1000.0
        rows.append(
            {
                "dataset_key": config.key,
                "component": config.component,
                "subject_id": 10,
                "subject_label": subject,
                "session_label": session,
                "run_label": run,
                "condition": "saccade",
                "trial_type": "saccade",
                "trial_event_type": str(trial_row.get("trial_type", "")),
                "trial_event_value": str(trial_row.get("value", "")),
                "trial_onset_s": trial_onset_s,
                "saccade_onset_s": onset_s,
                "saccade_duration_ms": duration_s * 1000.0,
                "saccade_duration": duration_s * 1000.0,
                "saccade_latency_ms": (onset_s - trial_onset_s) * 1000.0,
                "saccade_amplitude_px": float(row["saccade_amplitude_px"]),
                "saccade_amplitude": float(row["saccade_amplitude_px"]),
                "fixation_duration_ms": previous_fixation_duration_ms,
                "fixation_duration": previous_fixation_duration_ms,
                "sample_index": int(round(onset_s * float(raw.info["sfreq"]))),
                "source_file": safe_relpath(edf_path, output_dir),
            }
        )

    events = pd.DataFrame(rows)
    eeg = raw.get_data(picks=channel_names).astype(np.float32)
    epochs, kept_onsets, times_s = extract_epochs(
        eeg,
        events["sample_index"].to_numpy(dtype=int),
        float(raw.info["sfreq"]),
        tmin_s=SACCADE_TMIN_S,
        duration_s=SACCADE_DURATION_S,
    )
    keep = events["sample_index"].isin(set(int(v) for v in kept_onsets))
    kept = events.loc[keep].copy().reset_index(drop=True)
    if kept.empty:
        raise RuntimeError(f"No saccade epochs survived boundary checks for {config.key}.")
    kept["source_part_index"] = 1
    kept["source_epoch_index"] = np.arange(1, len(kept) + 1, dtype=int)
    kept["epoch_index"] = np.arange(1, len(kept) + 1, dtype=int)

    return [
        SubjectBundle(
            subject_label=subject,
            subject_id=10,
            channel_names=channel_names,
            sfreq_hz=float(raw.info["sfreq"]),
            times_s=times_s,
            epochs=epochs,
            events=kept,
            source_set_relpath=safe_relpath(edf_path, output_dir),
            source_eventlist_relpath=safe_relpath(physioevents_path, output_dir),
        )
    ]


def prepare_roamm_reading_fixations(output_root: Path, config: DatasetConfig) -> list[SubjectBundle]:
    output_dir = output_root / config.key
    source_dir = output_dir / "source" / "subject_ml_data" / "s10014"
    pkl_path = download_file(ROAMM_RUN1_URL, source_dir / "s10014_run1_ml_data.pkl")

    frame = pd.read_pickle(pkl_path)
    eeg_channels = list(frame.columns[:64])
    sfreq = float(frame["sfreq"].dropna().iloc[0])
    subject_id = 10014
    subject = subject_label(subject_id)

    fixations = frame[(frame["is_fix"].astype(bool)) & (frame["first_pass_reading"] == True)].copy()
    fixations["fixation_onset_s"] = fixations["fix_L_tStart"].combine_first(fixations["fix_R_tStart"])
    fixations["fixation_offset_s"] = fixations["fix_L_tEnd"].combine_first(fixations["fix_R_tEnd"])
    fixations["fixation_duration_ms"] = fixations[["fix_L_duration", "fix_R_duration"]].mean(axis=1)
    fixations["fixation_duration"] = fixations["fixation_duration_ms"]
    fixations["gaze_x"] = fixations[["fix_L_xAvg", "fix_R_xAvg"]].mean(axis=1)
    fixations["gaze_y"] = fixations[["fix_L_yAvg", "fix_R_yAvg"]].mean(axis=1)
    fixations["pupil"] = fixations[["fix_L_pupilAvg", "fix_R_pupilAvg"]].mean(axis=1)
    fixations["fixated_word"] = fixations["fix_L_fixed_word"].combine_first(fixations["fix_R_fixed_word"])
    fixations["fixated_word_key"] = fixations["fix_L_fixed_word_key"].combine_first(
        fixations["fix_R_fixed_word_key"]
    )
    fixations = (
        fixations.dropna(subset=["fixation_onset_s", "fixation_duration_ms"])
        .sort_values("fixation_onset_s")
        .drop_duplicates(subset=["fixation_onset_s"])
        .reset_index(drop=True)
    )
    fixations = fixations[fixations["fixation_duration_ms"] >= 80].copy().reset_index(drop=True)
    if fixations.empty:
        raise RuntimeError("No ROAMM first-pass fixation events survived filtering.")

    eeg = frame[eeg_channels].to_numpy(dtype=np.float32).T
    fixations["sample_index"] = np.rint(fixations["fixation_onset_s"].to_numpy(dtype=np.float64) * sfreq).astype(int)
    epochs, kept_onsets, times_s = extract_epochs(
        eeg,
        fixations["sample_index"].to_numpy(dtype=int),
        sfreq,
        tmin_s=SACCADE_TMIN_S,
        duration_s=SACCADE_DURATION_S,
    )
    keep = fixations["sample_index"].isin(set(int(v) for v in kept_onsets))
    kept = fixations.loc[keep].copy().reset_index(drop=True)
    kept["dataset_key"] = config.key
    kept["component"] = config.component
    kept["subject_id"] = subject_id
    kept["subject_label"] = subject
    kept["source_subject_label"] = "s10014"
    kept["session_label"] = "run1"
    kept["run_label"] = "run-1"
    kept["condition"] = np.where(kept["is_mw"] == True, "mind_wandering", "first_pass_reading")
    kept["attention_state"] = np.where(kept["is_mw"] == True, "mind_wandering", "attentive_reading")
    kept["eye"] = "binocular"
    kept["source_file"] = safe_relpath(pkl_path, output_dir)
    event_columns = [
        "dataset_key",
        "component",
        "subject_id",
        "subject_label",
        "source_subject_label",
        "session_label",
        "run_label",
        "condition",
        "attention_state",
        "eye",
        "page_num",
        "story_name",
        "is_mw",
        "mw_onset",
        "mw_offset",
        "mw_dur",
        "fixation_onset_s",
        "fixation_offset_s",
        "fixation_duration_ms",
        "fixation_duration",
        "gaze_x",
        "gaze_y",
        "pupil",
        "fixated_word",
        "fixated_word_key",
        "sample_index",
        "source_file",
    ]

    return [
        finalize_subject_bundle(
            subject_id=subject_id,
            channel_names=eeg_channels,
            sfreq_hz=sfreq,
            epochs_parts=[epochs],
            event_frames=[kept[event_columns]],
            source_relpaths=[safe_relpath(pkl_path, output_dir)],
            times_s=times_s,
        )
    ]


def prepare_saccade_onset_face_vr(output_root: Path, config: DatasetConfig) -> list[SubjectBundle]:
    output_dir = output_root / config.key
    source_root = output_dir / "source"
    eye_path = download_file(
        SACCADE_VR_EYE_TABLE_URL,
        source_root / "eye_movement_classification" / "extendedCombinedTable.csv",
    )
    set_path = download_file(
        SACCADE_VR_SET_URL,
        source_root / "preprocessed_EEG_data" / "new_full_data_1.set",
    )
    fdt_path = download_file(
        SACCADE_VR_FDT_URL,
        source_root / "preprocessed_EEG_data" / "new_full_data_1.fdt",
    )

    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")
    channel_names = eeg_channel_names(raw)
    table = pd.read_csv(eye_path)
    label = table["handLabel2"].to_numpy(dtype=int)
    valid = table["valid"].to_numpy(dtype=int) == 1
    is_saccade = (label == 1) & valid
    starts = np.flatnonzero(is_saccade & np.r_[True, ~is_saccade[:-1]])
    stops = np.flatnonzero(is_saccade & np.r_[~is_saccade[1:], True])

    rows: list[dict[str, object]] = []
    raw_stop_s = float(raw.times[-1])
    for start_idx, stop_idx in zip(starts, stops, strict=False):
        onset_s = float(table.at[int(start_idx), "time"])
        offset_s = float(table.at[int(stop_idx), "time"])
        duration_ms = (offset_s - onset_s) * 1000.0
        if onset_s < 0.5 or onset_s + 1.0 >= raw_stop_s or duration_ms <= 8 or duration_ms > 250:
            continue
        start_vec = table.loc[int(start_idx), ["xcoord", "ycoord", "zcoord"]].to_numpy(dtype=np.float64)
        stop_vec = table.loc[int(stop_idx), ["xcoord", "ycoord", "zcoord"]].to_numpy(dtype=np.float64)
        start_norm = np.linalg.norm(start_vec)
        stop_norm = np.linalg.norm(stop_vec)
        if start_norm == 0 or stop_norm == 0:
            amplitude_deg = float("nan")
        else:
            cos_angle = float(np.clip(np.dot(start_vec, stop_vec) / (start_norm * stop_norm), -1.0, 1.0))
            amplitude_deg = float(np.degrees(np.arccos(cos_angle)))
        rows.append(
            {
                "dataset_key": config.key,
                "component": config.component,
                "subject_id": 1,
                "subject_label": "sub-001",
                "source_subject_label": "new_full_data_1",
                "session_label": "vr_freeviewing",
                "run_label": "run-1",
                "condition": "saccade",
                "trial_type": "saccade",
                "saccade_onset_s": onset_s,
                "saccade_offset_s": offset_s,
                "saccade_duration_ms": duration_ms,
                "saccade_amplitude_deg": amplitude_deg,
                "sample_index": int(round(onset_s * float(raw.info["sfreq"]))),
                "source_file": safe_relpath(set_path, output_dir),
            }
        )
        if len(rows) >= SACCADE_VR_MAX_EVENTS:
            break

    events = pd.DataFrame(rows)
    if events.empty:
        raise RuntimeError("No VR saccade events survived filtering.")
    eeg = raw.get_data(picks=channel_names).astype(np.float32)
    epochs, kept_onsets, times_s = extract_epochs(
        eeg,
        events["sample_index"].to_numpy(dtype=int),
        float(raw.info["sfreq"]),
        tmin_s=SACCADE_TMIN_S,
        duration_s=SACCADE_DURATION_S,
    )
    keep = events["sample_index"].isin(set(int(v) for v in kept_onsets))
    kept = events.loc[keep].copy().reset_index(drop=True)

    return [
        finalize_subject_bundle(
            subject_id=1,
            channel_names=channel_names,
            sfreq_hz=float(raw.info["sfreq"]),
            epochs_parts=[epochs],
            event_frames=[kept],
            source_relpaths=[
                safe_relpath(set_path, output_dir),
                safe_relpath(fdt_path, output_dir),
                safe_relpath(eye_path, output_dir),
            ],
            times_s=times_s,
        )
    ]


def matlab_scalar(value: object, default: float = float("nan")) -> float:
    try:
        arr = np.asarray(value)
        if arr.size == 0:
            return default
        out = float(arr.ravel()[0])
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def mat_struct_array(value: object) -> list[object]:
    arr = np.ravel(value)
    return [item for item in arr]


def hdf5_mat_value(handle: h5py.File, ref: h5py.Reference) -> object:
    obj = handle[ref]
    arr = obj[()]
    matlab_class = obj.attrs.get("MATLAB_class")
    if matlab_class == b"char" or arr.dtype == np.uint16:
        return "".join(chr(int(c)) for c in arr.ravel()).strip()
    if arr.size == 0:
        return float("nan")
    if arr.size == 1:
        return arr.ravel()[0].item()
    return np.asarray(arr).squeeze()


def hdf5_struct_column(handle: h5py.File, group: h5py.Group, column: str) -> list[object]:
    refs = group[column][()]
    return [hdf5_mat_value(handle, ref) for ref in refs.ravel()]


def prepare_zuco2_reading_fixations(output_root: Path, config: DatasetConfig) -> list[SubjectBundle]:
    output_dir = output_root / config.key
    source_root = output_dir / "source"
    local_paths = {name: download_file(url, source_root / name) for name, url in ZUCO2_RAW_FILES.items()}
    eeg_path = local_paths["YAG_NR1_EEG.mat"]
    et_path = local_paths["YAG_NR1_ET.mat"]

    eeg_mat = loadmat(eeg_path, squeeze_me=True, struct_as_record=False, variable_names=["EEG"])
    eeg = eeg_mat["EEG"]
    sfreq = float(eeg.srate)
    all_channel_names = [str(item.labels) for item in np.ravel(eeg.chanlocs)]
    selected = [idx for idx, name in enumerate(all_channel_names) if name in set(ZUCO2_PREFERRED_CHANNELS)]
    if len(selected) < 8:
        selected = list(range(min(32, len(all_channel_names))))
    channel_names = [all_channel_names[idx] for idx in selected]
    eeg_data = np.asarray(eeg.data, dtype=np.float32)[selected, :]

    et_mat = loadmat(et_path, squeeze_me=True, struct_as_record=False, variable_names=["eyeevent", "event"])
    fix = et_mat["eyeevent"].fixations
    sac = et_mat["eyeevent"].saccades
    fix_data = np.asarray(fix.data, dtype=np.float64)
    fix_cols = [str(x) for x in np.ravel(fix.colheader)]
    sac_data = np.asarray(sac.data, dtype=np.float64)
    sac_cols = [str(x) for x in np.ravel(sac.colheader)]
    et_events = np.asarray(et_mat["event"], dtype=np.float64)
    eeg_events = mat_struct_array(eeg.event)

    et_event_times = et_events[:, 0]
    et_event_codes = et_events[:, 1].astype(int)
    eeg_event_times = np.asarray([matlab_scalar(ev.latency) for ev in eeg_events], dtype=np.float64)
    eeg_event_codes = np.asarray([int(str(ev.type).strip()) for ev in eeg_events], dtype=int)
    n_sync = min(len(et_event_codes), len(eeg_event_codes))
    if n_sync < 10 or not np.array_equal(et_event_codes[:n_sync], eeg_event_codes[:n_sync]):
        raise RuntimeError("ZuCo raw EEG/ET trigger streams do not align for YAG_NR1.")
    slope, intercept = np.polyfit(et_event_times[:n_sync], eeg_event_times[:n_sync] - 1.0, deg=1)

    idx_latency = fix_cols.index("latency")
    idx_endtime = fix_cols.index("endtime")
    idx_duration = fix_cols.index("duration")
    idx_x = fix_cols.index("fix_avgpos_x")
    idx_y = fix_cols.index("fix_avgpos_y")
    idx_pupil = fix_cols.index("fix_avgpupilsize")
    sac_latency_idx = sac_cols.index("latency")
    sac_amp_idx = sac_cols.index("sac_amplitude")
    sac_duration_idx = sac_cols.index("duration")

    fix_onsets = fix_data[:, idx_latency]
    sac_onsets = sac_data[:, sac_latency_idx]
    next_saccade_idx = np.searchsorted(sac_onsets, fix_onsets, side="right")
    rows: list[dict[str, object]] = []
    for row_idx, fixation in enumerate(fix_data):
        duration_ms = float(fixation[idx_duration])
        if duration_ms < 80:
            continue
        onset_sample = int(round(float(slope * fixation[idx_latency] + intercept)))
        next_idx = int(next_saccade_idx[row_idx]) if row_idx < len(next_saccade_idx) else -1
        if next_idx >= len(sac_data):
            next_amp = float("nan")
            next_sac_duration = float("nan")
        else:
            next_amp = float(sac_data[next_idx, sac_amp_idx])
            next_sac_duration = float(sac_data[next_idx, sac_duration_idx])
        rows.append(
            {
                "dataset_key": config.key,
                "component": config.component,
                "subject_id": 1,
                "subject_label": "sub-001",
                "source_subject_label": "YAG",
                "session_label": "NR1",
                "run_label": "NR1",
                "condition": "natural_reading",
                "trial_type": "fixation",
                "fixation_onset_s": onset_sample / sfreq,
                "fixation_offset_s": float(slope * fixation[idx_endtime] + intercept) / sfreq,
                "fixation_duration_ms": duration_ms,
                "fixation_duration": duration_ms,
                "gaze_x": float(fixation[idx_x]),
                "gaze_y": float(fixation[idx_y]),
                "pupil": float(fixation[idx_pupil]),
                "saccade_amplitude": next_amp,
                "saccade_duration": next_sac_duration,
                "sample_index": onset_sample,
                "source_file": safe_relpath(eeg_path, output_dir),
            }
        )

    events = pd.DataFrame(rows)
    epochs, kept_onsets, times_s = extract_epochs(
        eeg_data,
        events["sample_index"].to_numpy(dtype=int),
        sfreq,
        tmin_s=FIXATION_TMIN_S,
        duration_s=FIXATION_DURATION_S,
    )
    kept = events[events["sample_index"].isin(set(int(v) for v in kept_onsets))].copy().reset_index(drop=True)
    return [
        finalize_subject_bundle(
            subject_id=1,
            channel_names=channel_names,
            sfreq_hz=sfreq,
            epochs_parts=[epochs],
            event_frames=[kept],
            source_relpaths=[safe_relpath(eeg_path, output_dir), safe_relpath(et_path, output_dir)],
            times_s=times_s,
        )
    ]


def prepare_raccoons_reading(output_root: Path, config: DatasetConfig) -> list[SubjectBundle]:
    output_dir = output_root / config.key
    source_root = output_dir / "source"
    local_paths = {name: download_file(url, source_root / name) for name, url in RACCOONS_FILES.items()}
    eeg_path = local_paths["EEG003.mat"]
    fix_path = local_paths["ET_fix_data.tsv"]
    word_path = local_paths["ET_word_data.tsv"]
    n400_path = local_paths["N400.tsv"]
    lexical_path = local_paths["words.tsv"]

    fix_all = pd.read_csv(fix_path, sep="\t", encoding="latin1")
    word_table = pd.read_csv(word_path, sep="\t", encoding="utf-8")
    n400_table = pd.read_csv(n400_path, sep="\t", encoding="utf-8")
    lexical = pd.read_csv(lexical_path, sep="\t", encoding="latin1")
    fix_all = fix_all[fix_all["participant_id"] == 3].copy().reset_index(drop=True)

    with h5py.File(eeg_path, "r") as handle:
        eeg_group = handle["EEG"]
        sfreq = float(np.asarray(eeg_group["srate"]).ravel()[0])
        all_channel_names = hdf5_struct_column(handle, eeg_group["chanlocs"], "labels")
        selected = list(range(len(all_channel_names)))
        eeg_data = np.asarray(eeg_group["data"][:, selected], dtype=np.float32).T
        ev_group = eeg_group["event"]
        event_types = hdf5_struct_column(handle, ev_group, "type")
        latencies = hdf5_struct_column(handle, ev_group, "latency")
        l_fix_latencies = [float(lat) - 1.0 for typ, lat in zip(event_types, latencies, strict=False) if typ == "L_fixation"]

    if len(l_fix_latencies) != len(fix_all):
        raise RuntimeError(f"Raccoons fixation count mismatch: EEG={len(l_fix_latencies)} table={len(fix_all)}.")

    fix_all["sample_index"] = np.asarray([int(round(v)) for v in l_fix_latencies], dtype=int)
    fix_table = fix_all[fix_all["word_index"].notna() & (fix_all["fix_dur"] >= 80)].copy().reset_index(drop=True)
    fix_table["word_index"] = fix_table["word_index"].astype(int)

    word_features = word_table[word_table["participant_id"] == 3].copy()
    word_features = word_features.merge(
        lexical[["item_id", "word_index", "word_length", "surp", "logfreq", "word_class"]],
        on=["item_id", "word_index"],
        how="left",
    )
    word_features = word_features.merge(n400_table, on=["participant_id", "item_id", "word_index"], how="left")
    merged = fix_table.merge(
        word_features[
            [
                "participant_id",
                "trial_index",
                "item_id",
                "word_index",
                "first_fix_dur",
                "first_pass_dur",
                "total_dur",
                "fix_count",
                "sacc_len",
                "word_length",
                "surp",
                "logfreq",
                "word_class",
                "N400",
            ]
        ],
        on=["participant_id", "trial_index", "item_id", "word_index"],
        how="left",
    )
    merged = merged.iloc[:RACCOONS_MAX_EVENTS].copy().reset_index(drop=True)
    events = pd.DataFrame(
        {
            "dataset_key": config.key,
            "component": config.component,
            "subject_id": 3,
            "subject_label": "sub-003",
            "source_subject_label": "EEG003",
            "session_label": "sentence_reading",
            "run_label": "merged",
            "condition": "sentence_reading",
            "trial_type": "fixation",
            "item_id": merged["item_id"],
            "trial_index": merged["trial_index"],
            "fixation_index": merged["fix_index"],
            "word_index": merged["word_index"],
            "word": merged["word"],
            "fixation_duration_ms": merged["fix_dur"],
            "fixation_duration": merged["fix_dur"],
            "FFD": merged["first_fix_dur"],
            "GD": merged["first_pass_dur"],
            "TRT": merged["total_dur"],
            "saccade_amplitude": merged["sacc_len"],
            "word_length": merged["word_length"],
            "surprisal": merged["surp"],
            "N400_size": merged["N400"],
            "gaze_x": merged["fix_x"],
            "gaze_y": merged["fix_y"],
            "sample_index": merged["sample_index"],
            "source_file": safe_relpath(eeg_path, output_dir),
        }
    )
    epochs, kept_onsets, times_s = extract_epochs(
        eeg_data,
        events["sample_index"].to_numpy(dtype=int),
        sfreq,
        tmin_s=FIXATION_TMIN_S,
        duration_s=FIXATION_DURATION_S,
    )
    kept = events[events["sample_index"].isin(set(int(v) for v in kept_onsets))].copy().reset_index(drop=True)
    return [
        finalize_subject_bundle(
            subject_id=3,
            channel_names=[str(name) for name in all_channel_names],
            sfreq_hz=sfreq,
            epochs_parts=[epochs],
            event_frames=[kept],
            source_relpaths=[
                safe_relpath(eeg_path, output_dir),
                safe_relpath(fix_path, output_dir),
                safe_relpath(word_path, output_dir),
                safe_relpath(n400_path, output_dir),
                safe_relpath(lexical_path, output_dir),
            ],
            times_s=times_s,
        )
    ]


def prepare_unfold_facefreeview(output_root: Path, config: DatasetConfig) -> list[SubjectBundle]:
    output_dir = output_root / config.key
    source_root = output_dir / "source"
    zip_path = download_file(UNFOLD_ZIP_URL, source_root / "Ehinger_Dimigen_unfoldtoolbox_opendata.zip")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in [UNFOLD_SET_MEMBER, "opendata_create_figure10.m"]:
            target = source_root / member
            if not target.is_file():
                zf.extract(member, source_root)
    set_path = source_root / UNFOLD_SET_MEMBER

    mat = loadmat(set_path, squeeze_me=True, struct_as_record=False, variable_names=["EEG"])
    eeg = mat["EEG"]
    sfreq = float(eeg.srate)
    channel_names = [str(item.labels) for item in np.ravel(eeg.chanlocs)]
    eeg_data = np.asarray(eeg.data, dtype=np.float32)
    events_src = mat_struct_array(eeg.event)
    saccade_rows: list[dict[str, object]] = []
    fixation_durations: list[float] = []
    last_fixation_duration = float("nan")
    last_stim_sample = None
    for ev in events_src:
        ev_type = str(ev.type)
        latency = int(round(matlab_scalar(ev.latency))) - 1
        if ev_type == "stimonset":
            last_stim_sample = latency
        elif ev_type == "fixation":
            last_fixation_duration = matlab_scalar(ev.duration) * 1000.0 / sfreq
        elif ev_type == "saccade":
            saccade_duration = matlab_scalar(ev.duration) * 1000.0 / sfreq
            amplitude = matlab_scalar(ev.sac_amplitude)
            if saccade_duration < 8 or amplitude <= 0:
                continue
            latency_ms = float("nan") if last_stim_sample is None else (latency - last_stim_sample) * 1000.0 / sfreq
            saccade_rows.append(
                {
                    "dataset_key": config.key,
                    "component": config.component,
                    "subject_id": 1,
                    "subject_label": "sub-001",
                    "source_subject_label": "face_saccades_opendata_fig10",
                    "session_label": "freeviewing",
                    "run_label": "continuous",
                    "condition": "face_freeview",
                    "face_condition": "face",
                    "trial_type": "saccade",
                    "saccade_onset_s": latency / sfreq,
                    "saccade_duration_ms": saccade_duration,
                    "saccade_duration": saccade_duration,
                    "saccade_amplitude": amplitude,
                    "saccade_angle": matlab_scalar(ev.sac_angle),
                    "saccade_latency_ms": latency_ms,
                    "fixation_duration": last_fixation_duration,
                    "fixation_duration_ms": last_fixation_duration,
                    "sample_index": latency,
                    "source_file": safe_relpath(set_path, output_dir),
                }
            )
            fixation_durations.append(last_fixation_duration)
        if len(saccade_rows) >= UNFOLD_MAX_EVENTS:
            break

    events = pd.DataFrame(saccade_rows)
    epochs, kept_onsets, times_s = extract_epochs(
        eeg_data,
        events["sample_index"].to_numpy(dtype=int),
        sfreq,
        tmin_s=SACCADE_TMIN_S,
        duration_s=SACCADE_DURATION_S,
    )
    kept = events[events["sample_index"].isin(set(int(v) for v in kept_onsets))].copy().reset_index(drop=True)
    return [
        finalize_subject_bundle(
            subject_id=1,
            channel_names=channel_names,
            sfreq_hz=sfreq,
            epochs_parts=[epochs],
            event_frames=[kept],
            source_relpaths=[safe_relpath(set_path, output_dir), safe_relpath(zip_path, output_dir)],
            times_s=times_s,
        )
    ]


def prepare_kilo_word_erp(output_root: Path, config: DatasetConfig) -> list[SubjectBundle]:
    output_dir = output_root / config.key
    source_root = output_dir / "source"
    paths = {name: download_file(url, source_root / name) for name, url in KILO_WORD_URLS.items()}
    erp_path = paths["KWORD_ERP_LEXICAL_DECISION_DGMH2015.txt"]
    variables_path = paths["KWORD_VARIABLES_DGMH2015.txt"]

    variables = pd.read_csv(variables_path, sep="\t")
    erp = pd.read_csv(erp_path, sep="\t")
    time_columns = [column for column in erp.columns if column.endswith("ms") and column[:-2].lstrip("-").isdigit()]
    if not time_columns:
        raise RuntimeError("Kilo-Word ERP table did not expose millisecond time columns.")
    times_s = np.asarray([float(column[:-2]) / 1000.0 for column in time_columns], dtype=np.float32)
    channel_names = [str(name) for name in erp["ELECNAME"].drop_duplicates().tolist()]
    word_ids = variables["WORD#"].astype(str).tolist()

    epochs = np.empty((len(word_ids), len(times_s), len(channel_names)), dtype=np.float32)
    events_rows: list[dict[str, object]] = []
    grouped = {str(word_id): frame for word_id, frame in erp.groupby("WORD#", sort=False)}
    variable_by_word = {str(row["WORD#"]): row for _, row in variables.iterrows()}
    for trial_idx, word_id in enumerate(word_ids):
        frame = grouped.get(word_id)
        if frame is None:
            raise RuntimeError(f"Kilo-Word ERP rows missing for {word_id}.")
        frame = frame.set_index("ELECNAME").reindex(channel_names)
        epochs[trial_idx, :, :] = frame[time_columns].to_numpy(dtype=np.float32).T
        var_row = variable_by_word[word_id]
        events_rows.append(
            {
                "dataset_key": config.key,
                "component": config.component,
                "subject_id": 1,
                "subject_label": "sub-001",
                "source_subject_label": "grand_average",
                "session_label": "lexical_decision",
                "run_label": "grand_average",
                "condition": "word",
                "trial_type": "word_average",
                "word_id": word_id,
                "word": str(var_row["WORD"]),
                "word_frequency": float(var_row["WordFrequency"]),
                "WordFrequency": float(var_row["WordFrequency"]),
                "concreteness": float(var_row["Concreteness"]),
                "Concreteness": float(var_row["Concreteness"]),
                "orthographic_distance": float(var_row["OrthographicDistance"]),
                "number_of_letters": float(var_row["NumberOfLetters"]),
                "word_length": float(var_row["NumberOfLetters"]),
                "bigram_frequency": float(var_row["BigramFrequency"]),
                "visual_complexity": float(var_row["VisualComplexity"]),
                "sample_index": int(trial_idx + 1),
                "source_file": safe_relpath(erp_path, output_dir),
            }
        )

    events = pd.DataFrame(events_rows)
    return [
        finalize_subject_bundle(
            subject_id=1,
            channel_names=channel_names,
            sfreq_hz=250.0,
            epochs_parts=[epochs],
            event_frames=[events],
            source_relpaths=[safe_relpath(erp_path, output_dir), safe_relpath(variables_path, output_dir)],
            times_s=times_s,
        )
    ]


def prepare_confidence_perceptual_decisions(output_root: Path, config: DatasetConfig) -> list[SubjectBundle]:
    output_dir = output_root / config.key
    source_root = output_dir / "source" / "sub-01" / "EEG"
    epochs_parts: list[np.ndarray] = []
    event_frames: list[pd.DataFrame] = []
    relpaths: list[str] = []
    channel_names_ref: list[str] | None = None
    times_s_ref: np.ndarray | None = None
    sfreq_ref: float | None = None

    for run in CONFIDENCE_RUNS:
        data_name = f"EEG_data_sub-01_run-{run:02d}.mat"
        events_name = f"EEG_events_sub-01_run-{run:02d}.mat"
        data_path = download_file(f"{CONFIDENCE_OPENNEURO_ROOT}/sub-01/EEG/{data_name}", source_root / data_name)
        events_path = download_file(
            f"{CONFIDENCE_OPENNEURO_ROOT}/sub-01/EEG/{events_name}",
            source_root / events_name,
        )
        data_mat = loadmat(data_path, squeeze_me=True, struct_as_record=False)
        events_mat = loadmat(events_path, squeeze_me=True, struct_as_record=False)
        eeg_source = data_mat["EEGdata"]
        eeg_data = np.asarray(eeg_source.Y if hasattr(eeg_source, "Y") else eeg_source, dtype=np.float32)
        sfreq = float(np.asarray(data_mat.get("fs", 1000.0)).ravel()[0])
        channel_names = [f"E{idx:02d}" for idx in range(1, eeg_data.shape[0] + 1)]
        channel_names_ref = channel_names if channel_names_ref is None else channel_names_ref
        sfreq_ref = sfreq if sfreq_ref is None else sfreq_ref

        response_samples = np.asarray(events_mat["tresp"], dtype=int).ravel()
        epochs, kept_onsets, times_s = extract_epochs(
            eeg_data,
            response_samples,
            sfreq,
            tmin_s=RESPONSE_TMIN_S,
            duration_s=RESPONSE_DURATION_S,
        )
        times_s_ref = times_s if times_s_ref is None else times_s_ref
        keep_mask = np.isin(response_samples, kept_onsets)
        n_events = len(response_samples)
        frame = pd.DataFrame(
            {
                "dataset_key": config.key,
                "component": config.component,
                "subject_id": 1,
                "subject_label": "sub-001",
                "source_subject_label": "sub-01",
                "session_label": "confidence",
                "run_label": f"run-{run:02d}",
                "condition": "perceptual_decision",
                "trial_type": "response",
                "sample_index": response_samples,
                "stimulus_sample_index": np.asarray(events_mat["tstim"], dtype=int).ravel(),
                "decision_duration_ms": np.asarray(events_mat["RT"], dtype=float).ravel(),
                "reaction_time_ms": np.asarray(events_mat["RT"], dtype=float).ravel(),
                "confidence_rating": np.asarray(events_mat["confidence"], dtype=float).ravel(),
                "accuracy": np.asarray(events_mat["accuracy"], dtype=int).ravel(),
                "choice": np.asarray(events_mat["choice"], dtype=int).ravel(),
                "dotdirection": np.asarray(events_mat["dotdirection"], dtype=int).ravel(),
                "source_file": safe_relpath(data_path, output_dir),
            }
        )
        if len(frame) != n_events:
            raise RuntimeError(f"Confidence event table length mismatch for run {run}.")
        epochs_parts.append(epochs)
        event_frames.append(frame.loc[keep_mask].copy().reset_index(drop=True))
        relpaths.extend([safe_relpath(data_path, output_dir), safe_relpath(events_path, output_dir)])

    if channel_names_ref is None or sfreq_ref is None or times_s_ref is None:
        raise RuntimeError("No confidence EEG runs were imported.")
    return [
        finalize_subject_bundle(
            subject_id=1,
            channel_names=channel_names_ref,
            sfreq_hz=sfreq_ref,
            epochs_parts=epochs_parts,
            event_frames=event_frames,
            source_relpaths=relpaths,
            times_s=times_s_ref,
        )
    ]


def hdf5_cell_numeric(handle: h5py.File, dataset: h5py.Dataset, row: int, col: int) -> np.ndarray:
    ref = dataset[row, col]
    arr = np.asarray(handle[ref])
    if arr.size == 0 or np.array_equal(arr.ravel(), np.asarray([0, 0])):
        return np.empty((0, 0), dtype=np.float64)
    return np.asarray(arr, dtype=np.float64)


def hdf5_cell_scalar(handle: h5py.File, dataset: h5py.Dataset, row: int, col: int) -> float:
    arr = hdf5_cell_numeric(handle, dataset, row, col)
    if arr.size == 0:
        return float("nan")
    return float(arr.ravel()[0])


def prepare_visuomotor_chenguang(output_root: Path, config: DatasetConfig) -> list[SubjectBundle]:
    output_dir = output_root / config.key
    source_root = output_dir / "source"
    for name, url in VISUOMOTOR_SMALL_FILES.items():
        download_file(url, source_root / name)
    raw_root = source_root / "EEG_RAW"
    for name, url in VISUOMOTOR_RAW_FILES.items():
        download_file(url, raw_root / name)

    eye_zip = source_root / "EYE_PRE.zip"
    eye_path = source_root / "EYE_PRE" / "sub1.mat"
    if not eye_path.is_file():
        with zipfile.ZipFile(eye_zip, "r") as zf:
            zf.extract("sub1.mat", eye_path.parent)

    beh_zip = source_root / "BEH_PRE.zip"
    beh_path = source_root / "BEH_PRE" / "sub1.mat"
    if not beh_path.is_file():
        with zipfile.ZipFile(beh_zip, "r") as zf:
            zf.extract("sub1.mat", beh_path.parent)

    raw_path = raw_root / "sub1.cdt"
    raw = mne.io.read_raw_curry(raw_path, preload=True, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    selected_channels = [name for name in raw.ch_names if name != "Trigger"]
    eeg_data = raw.get_data(picks=selected_channels).astype(np.float32)
    annotation_rows = pd.DataFrame(
        {
            "description": [str(desc) for desc in raw.annotations.description],
            "onset_s": np.asarray(raw.annotations.onset, dtype=float),
        }
    )

    rows: list[dict[str, object]] = []
    with h5py.File(eye_path, "r") as handle, h5py.File(beh_path, "r") as beh_handle:
        eye_group = handle["EYE"]
        beh_group = beh_handle["BEH"]
        fix_cells = eye_group["Fix"]
        sac_cells = eye_group["Sac"]
        tag_cells = eye_group["Tag"]
        rt1_cells = beh_group["Rt1"]
        rt2_cells = beh_group["Rt2"]
        rt3_cells = beh_group["Rt3"]
        acc_cells = beh_group["Acc"]
        for block_idx in range(fix_cells.shape[0]):
            trial_events = annotation_rows[annotation_rows["description"] == str(81 + block_idx)].reset_index(drop=True)
            n_trials = min(fix_cells.shape[1], len(trial_events))
            for trial_idx in range(n_trials):
                trial_onset_s = float(trial_events.loc[trial_idx, "onset_s"])
                fixes = hdf5_cell_numeric(handle, fix_cells, block_idx, trial_idx)
                sacs = hdf5_cell_numeric(handle, sac_cells, block_idx, trial_idx)
                tag = hdf5_cell_numeric(handle, tag_cells, block_idx, trial_idx)
                if fixes.size == 0:
                    continue
                if fixes.shape[0] == 6:
                    fixes = fixes.T
                if sacs.size != 0 and sacs.shape[0] == 9:
                    sacs = sacs.T
                target_position = float(tag.ravel()[0]) if tag.size else float("nan")
                rt1_s = hdf5_cell_scalar(beh_handle, rt1_cells, block_idx, trial_idx)
                rt2_s = hdf5_cell_scalar(beh_handle, rt2_cells, block_idx, trial_idx)
                rt3_s = hdf5_cell_scalar(beh_handle, rt3_cells, block_idx, trial_idx)
                accuracy = hdf5_cell_scalar(beh_handle, acc_cells, block_idx, trial_idx)
                reaction_time_ms = rt1_s * 1000.0 if 0 < rt1_s < 30 else float("nan")
                movement_time_ms = rt2_s * 1000.0 if 0 < rt2_s < 30 else float("nan")
                response_time_ms = rt3_s * 1000.0 if 0 < rt3_s < 30 else float("nan")
                sac_starts = sacs[:, 0] if sacs.size else np.asarray([], dtype=float)
                for fix_idx, fix in enumerate(fixes):
                    start_ms, end_ms, duration_ms, gaze_x, gaze_y, pupil = [float(value) for value in fix[:6]]
                    if duration_ms < 50:
                        continue
                    next_sac_duration = float("nan")
                    next_sac_amplitude = float("nan")
                    next_sac_idx = int(np.searchsorted(sac_starts, start_ms, side="right")) if sac_starts.size else -1
                    if 0 <= next_sac_idx < len(sacs):
                        next_sac_duration = float(sacs[next_sac_idx, 2])
                        next_sac_amplitude = float(sacs[next_sac_idx, 7])
                    onset_sample = int(round((trial_onset_s + start_ms / 1000.0) * sfreq))
                    rows.append(
                        {
                            "dataset_key": config.key,
                            "component": config.component,
                            "subject_id": 1,
                            "subject_label": "sub-001",
                            "source_subject_label": "sub1",
                            "session_label": "visuomotor",
                            "run_label": f"block-{block_idx + 1:02d}",
                            "condition": "visuomotor",
                            "trial_type": "fixation",
                            "block_index": block_idx + 1,
                            "trial_index": trial_idx + 1,
                            "fixation_index": fix_idx + 1,
                            "target_position": target_position,
                            "fixation_onset_s": trial_onset_s + start_ms / 1000.0,
                            "fixation_duration_ms": duration_ms,
                            "fixation_duration": duration_ms,
                            "saccade_duration": next_sac_duration,
                            "saccade_amplitude": next_sac_amplitude,
                            "reaction_time": reaction_time_ms,
                            "reaction_time_ms": reaction_time_ms,
                            "movement_time_ms": movement_time_ms,
                            "response_time_ms": response_time_ms,
                            "accuracy": accuracy,
                            "gaze_x": gaze_x,
                            "gaze_y": gaze_y,
                            "pupil": pupil,
                            "sample_index": onset_sample,
                            "source_file": safe_relpath(raw_path, output_dir),
                        }
                    )
                    if len(rows) >= VISUOMOTOR_MAX_EVENTS:
                        break
                if len(rows) >= VISUOMOTOR_MAX_EVENTS:
                    break
            if len(rows) >= VISUOMOTOR_MAX_EVENTS:
                break

    events = pd.DataFrame(rows)
    if events.empty:
        raise RuntimeError("No Visuomotor fixation events were extracted.")
    epochs, kept_onsets, times_s = extract_epochs(
        eeg_data,
        events["sample_index"].to_numpy(dtype=int),
        sfreq,
        tmin_s=FIXATION_TMIN_S,
        duration_s=FIXATION_DURATION_S,
    )
    kept = events[events["sample_index"].isin(set(int(v) for v in kept_onsets))].copy().reset_index(drop=True)
    return [
        finalize_subject_bundle(
            subject_id=1,
            channel_names=selected_channels,
            sfreq_hz=sfreq,
            epochs_parts=[epochs],
            event_frames=[kept],
            source_relpaths=[
                safe_relpath(raw_path, output_dir),
                safe_relpath(raw_root / "sub1.cdt.ceo", output_dir),
                safe_relpath(raw_root / "sub1.cdt.dpa", output_dir),
                safe_relpath(eye_path, output_dir),
                safe_relpath(beh_path, output_dir),
            ],
            times_s=times_s,
        )
    ]


def prepare_eeget_rsod(output_root: Path, config: DatasetConfig) -> list[SubjectBundle]:
    output_dir = output_root / config.key
    source_root = output_dir / "source"
    local_paths = {name: download_file(url, source_root / name) for name, url in EEGET_FILES.items()}
    extracted_root = source_root / "extracted"
    eeg_path = extracted_root / "EEG" / "P01.edf"
    et_path = extracted_root / "ET" / "P01.txt"
    if not eeg_path.is_file():
        with zipfile.ZipFile(local_paths["EEG.zip"], "r") as zf:
            zf.extract("EEG/P01.edf", extracted_root)
    if not et_path.is_file():
        with zipfile.ZipFile(local_paths["ET.zip"], "r") as zf:
            zf.extract("ET/P01.txt", extracted_root)

    label_names: set[str] = set()
    with zipfile.ZipFile(local_paths["label.zip"], "r") as zf:
        for member in zf.namelist():
            if member.startswith("label/") and member.endswith(".xml"):
                label_names.add(Path(member).stem)

    sync_table = pd.read_excel(local_paths["Time Synchronization Data.xlsx"], sheet_name=0)
    sync_row = sync_table[sync_table["Participant No."].astype(str).str.upper() == "P01"]
    if sync_row.empty:
        raise RuntimeError("EEGET time synchronization table does not contain P01.")
    sync_b = float(sync_row.iloc[0]["b"])

    raw = mne.io.read_raw_edf(eeg_path, preload=True, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    channel_names = list(raw.ch_names)
    eeg_data = raw.get_data(picks=channel_names).astype(np.float32)

    usecols = [
        "Time",
        "Type",
        "Trial",
        "L POR X [px]",
        "L POR Y [px]",
        "R POR X [px]",
        "R POR Y [px]",
        "L Pupil Diameter [mm]",
        "R Pupil Diameter [mm]",
        "L Event Info",
        "R Event Info",
        "Stimulus",
    ]
    et = pd.read_csv(
        et_path,
        encoding="gb18030",
        comment="#",
        usecols=usecols,
        on_bad_lines="skip",
    )
    samples = et[et["Type"].astype(str) == "SMP"].copy()
    samples["Stimulus"] = samples["Stimulus"].astype(str)
    samples = samples[samples["Stimulus"].str.match(r"^\d{5}\.jpg$")].copy()
    samples["event_label"] = samples["L Event Info"].fillna(samples["R Event Info"]).astype(str)
    samples = samples[samples["event_label"].isin(["Fixation", "Saccade"])].copy()
    samples["segment_id"] = (
        (samples["event_label"] != samples["event_label"].shift())
        | (samples["Stimulus"] != samples["Stimulus"].shift())
        | (samples["Trial"] != samples["Trial"].shift())
    ).cumsum()

    segment_rows: list[dict[str, object]] = []
    for _, segment in samples.groupby("segment_id", sort=False):
        event_label = str(segment["event_label"].iloc[0])
        start_time = float(segment["Time"].iloc[0])
        end_time = float(segment["Time"].iloc[-1])
        duration_ms = max(4.0, (end_time - start_time) / 1000.0 + 4.0)
        left_x = pd.to_numeric(segment["L POR X [px]"], errors="coerce").replace(0, np.nan)
        left_y = pd.to_numeric(segment["L POR Y [px]"], errors="coerce").replace(0, np.nan)
        right_x = pd.to_numeric(segment["R POR X [px]"], errors="coerce").replace(0, np.nan)
        right_y = pd.to_numeric(segment["R POR Y [px]"], errors="coerce").replace(0, np.nan)
        gaze_x = float(np.nanmean(np.vstack([left_x.to_numpy(dtype=float), right_x.to_numpy(dtype=float)])))
        gaze_y = float(np.nanmean(np.vstack([left_y.to_numpy(dtype=float), right_y.to_numpy(dtype=float)])))
        pupil_left = pd.to_numeric(segment["L Pupil Diameter [mm]"], errors="coerce").replace(0, np.nan)
        pupil_right = pd.to_numeric(segment["R Pupil Diameter [mm]"], errors="coerce").replace(0, np.nan)
        pupil = float(np.nanmean(np.vstack([pupil_left.to_numpy(dtype=float), pupil_right.to_numpy(dtype=float)])))
        x_start = float(np.nanmean([left_x.iloc[0], right_x.iloc[0]]))
        y_start = float(np.nanmean([left_y.iloc[0], right_y.iloc[0]]))
        x_end = float(np.nanmean([left_x.iloc[-1], right_x.iloc[-1]]))
        y_end = float(np.nanmean([left_y.iloc[-1], right_y.iloc[-1]]))
        amplitude_px = float(math.hypot(x_end - x_start, y_end - y_start))
        stimulus = str(segment["Stimulus"].iloc[0])
        stem = Path(stimulus).stem
        segment_rows.append(
            {
                "event_label": event_label,
                "start_time": start_time,
                "end_time": end_time,
                "duration_ms": duration_ms,
                "trial": int(segment["Trial"].iloc[0]),
                "stimulus": stimulus,
                "target_present": stem in label_names,
                "gaze_x": gaze_x,
                "gaze_y": gaze_y,
                "pupil": pupil,
                "amplitude_px": amplitude_px,
            }
        )

    segments = pd.DataFrame(segment_rows)
    if segments.empty:
        raise RuntimeError("No EEGET fixation/saccade segments were parsed from P01 ET file.")
    saccades = segments[segments["event_label"] == "Saccade"].copy().reset_index(drop=True)
    fixation_rows: list[dict[str, object]] = []
    for _, fix in segments[segments["event_label"] == "Fixation"].iterrows():
        if float(fix["duration_ms"]) < 50:
            continue
        later_saccades = saccades[
            (saccades["trial"] == fix["trial"])
            & (saccades["stimulus"] == fix["stimulus"])
            & (saccades["start_time"] > fix["start_time"])
        ]
        if later_saccades.empty:
            saccade_duration = float("nan")
            saccade_amplitude = float("nan")
        else:
            next_saccade = later_saccades.iloc[0]
            saccade_duration = float(next_saccade["duration_ms"])
            saccade_amplitude = float(next_saccade["amplitude_px"])
        onset_sample = int(round((float(fix["start_time"]) - sync_b) / 2000.0))
        fixation_rows.append(
            {
                "dataset_key": config.key,
                "component": config.component,
                "subject_id": 1,
                "subject_label": "sub-001",
                "source_subject_label": "P01",
                "session_label": "visual_search",
                "run_label": "continuous",
                "condition": "target_present" if bool(fix["target_present"]) else "target_absent",
                "trial_type": "fixation",
                "trial_index": int(fix["trial"]),
                "stimulus": str(fix["stimulus"]),
                "target_present": bool(fix["target_present"]),
                "fixation_duration_ms": float(fix["duration_ms"]),
                "fixation_duration": float(fix["duration_ms"]),
                "saccade_duration": saccade_duration,
                "saccade_amplitude": saccade_amplitude,
                "gaze_x": float(fix["gaze_x"]),
                "gaze_y": float(fix["gaze_y"]),
                "pupil": float(fix["pupil"]),
                "sample_index": onset_sample,
                "source_file": safe_relpath(eeg_path, output_dir),
            }
        )
        if len(fixation_rows) >= EEGET_MAX_EVENTS:
            break

    events = pd.DataFrame(fixation_rows)
    if events.empty:
        raise RuntimeError("No EEGET fixation rows survived duration filtering.")
    epochs, kept_onsets, times_s = extract_epochs(
        eeg_data,
        events["sample_index"].to_numpy(dtype=int),
        sfreq,
        tmin_s=FIXATION_TMIN_S,
        duration_s=FIXATION_DURATION_S,
    )
    kept = events[events["sample_index"].isin(set(int(v) for v in kept_onsets))].copy().reset_index(drop=True)
    return [
        finalize_subject_bundle(
            subject_id=1,
            channel_names=channel_names,
            sfreq_hz=sfreq,
            epochs_parts=[epochs],
            event_frames=[kept],
            source_relpaths=[
                safe_relpath(eeg_path, output_dir),
                safe_relpath(et_path, output_dir),
                safe_relpath(local_paths["Time Synchronization Data.xlsx"], output_dir),
                safe_relpath(local_paths["label.zip"], output_dir),
            ],
            times_s=times_s,
        )
    ]


def unavailable_importer(message: str) -> Callable[[Path, DatasetConfig], list[SubjectBundle]]:
    def _prepare(_output_root: Path, _config: DatasetConfig) -> list[SubjectBundle]:
        raise RuntimeError(message)

    return _prepare


DATASETS: dict[str, DatasetConfig] = {
    "nencki_symfonia_srt": DatasetConfig(
        key="nencki_symfonia_srt",
        component="Nencki-Symfonia Simple Reaction Task",
        source_component="https://openneuro.org/datasets/ds004621",
        source_processing_scripts="https://github.com/OpenNeuroDatasets/ds004621",
        reader_docs=MNE_BRAINVISION_DOCS,
        preferred_channels=CENTRAL_RESPONSE_CHANNELS,
        recommended_sort_columns=("reaction_time_ms", "condition", "epoch_index"),
        official_source_examples={
            "events": "sub-01/eeg/sub-01_task-srt_events.tsv pairs stimulus and response rows.",
            "epoching": "Response-locked epochs use [-1.3, +0.3] s so stimulus onset remains visible when sorting by RT.",
        },
        prepare=prepare_nencki_symfonia_srt,
    ),
    "openneuro_gonogo_ds002680": DatasetConfig(
        key="openneuro_gonogo_ds002680",
        component="OpenNeuro ds002680 Go-Nogo",
        source_component="https://openneuro.org/datasets/ds002680/versions/1.2.0",
        source_processing_scripts="https://github.com/OpenNeuroDatasets/ds002680",
        reader_docs=MNE_EEGLAB_RAW_DOCS,
        preferred_channels=CENTRAL_RESPONSE_CHANNELS,
        recommended_sort_columns=("reaction_time_ms", "condition", "accuracy", "run_label", "epoch_index"),
        official_source_examples={
            "events": "Stimulus rows provide response_time in ms; response rows provide correctness.",
            "epoching": "Response-locked epochs use [-1.3, +0.3] s so stimulus onset remains visible when sorting by RT.",
        },
        prepare=prepare_openneuro_gonogo_ds002680,
    ),
    "eegeyenet_saccades": DatasetConfig(
        key="eegeyenet_saccades",
        component="EEGEyeNet Saccades",
        source_component="https://openneuro.org/datasets/ds005872",
        source_processing_scripts="https://github.com/OpenNeuroDatasets/ds005872",
        reader_docs=MNE_EDF_DOCS,
        preferred_channels=EGI_FACE_CHANNELS,
        recommended_sort_columns=(
            "saccade_amplitude_px",
            "saccade_latency_ms",
            "saccade_duration_ms",
            "epoch_index",
        ),
        official_source_examples={
            "events": "recording-eye1_physioevents.tsv marks saccades; recording-eye1_physio.tsv supplies gaze positions.",
            "epoching": "Saccade-locked epochs use [-0.5, +1.0] s with EGI face/peripheral channels preferred.",
        },
        prepare=prepare_eegeyenet_saccades,
    ),
    "roamm_reading_fixations": DatasetConfig(
        key="roamm_reading_fixations",
        component="ROAMM Reading Fixations",
        source_component="https://data-brain-mind.github.io/tutorials/reading-observed-at-mindless-moments-roamm-a-simultaneous-eeg-and-eye-tracking-dataset-of-natural-reading-with-attention-annotations/",
        source_processing_scripts="https://osf.io/kmvgb/",
        reader_docs="https://pandas.pydata.org/docs/reference/api/pandas.read_pickle.html",
        preferred_channels=ROAMM_PREFERRED_CHANNELS,
        recommended_sort_columns=(
            "fixation_duration_ms",
            "condition",
            "pupil",
            "gaze_x",
            "gaze_y",
            "epoch_index",
        ),
        official_source_examples={
            "events": "s10014_run1_ml_data.pkl stores aligned 256 Hz EEG and expanded fixation labels.",
            "epoching": "First-pass fixation-locked epochs use [-0.5, +1.0] s and keep frontal/peripheral channels.",
        },
        prepare=prepare_roamm_reading_fixations,
    ),
    "saccade_onset_face_vr": DatasetConfig(
        key="saccade_onset_face_vr",
        component="Saccade-Onset Face Perception VR",
        source_component="https://pmc.ncbi.nlm.nih.gov/articles/PMC12418071/",
        source_processing_scripts="https://github.com/debnolte/saccade-onset_ERPs_of-face_perception_free-viewing_VR",
        reader_docs=MNE_EEGLAB_RAW_DOCS,
        preferred_channels=SACCADE_VR_PREFERRED_CHANNELS,
        recommended_sort_columns=("saccade_duration_ms", "saccade_amplitude_deg", "epoch_index"),
        official_source_examples={
            "events": "extendedCombinedTable.csv supplies hand-corrected eye-movement labels and gaze vectors.",
            "epoching": "A deterministic subset of subject new_full_data_1 is saccade-locked to [-0.5, +1.0] s.",
        },
        prepare=prepare_saccade_onset_face_vr,
    ),
    "02_new_eegeyenet_saccades": DatasetConfig(
        key="02_new_eegeyenet_saccades",
        component="EEGEyeNet minimally processed saccades",
        source_component="https://osf.io/ktv7m/ and https://openneuro.org/datasets/ds005872",
        source_processing_scripts="https://github.com/ardkastrati/EEGEyeNet",
        reader_docs=MNE_EDF_DOCS,
        preferred_channels=EGI_FACE_CHANNELS,
        recommended_sort_columns=(
            "saccade_amplitude",
            "saccade_duration",
            "saccade_latency_ms",
            "fixation_duration",
            "epoch_index",
        ),
        official_source_examples={
            "events": "BIDS physioevents.tsv marks minimally processed eye-movement saccades and fixations.",
            "epoching": "Saccade-locked epochs use [-0.5, +2.0] s; peripheral EGI channels are preferred.",
        },
        prepare=prepare_eegeyenet_saccades,
    ),
    "02_new_zuco2_reading_fixations": DatasetConfig(
        key="02_new_zuco2_reading_fixations",
        component="ZuCo 2.0 raw natural-reading fixations",
        source_component=ZUCO2_SOURCE_URL,
        source_processing_scripts="https://github.com/norahollenstein/zuco-benchmark",
        reader_docs="https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html",
        preferred_channels=ZUCO2_PREFERRED_CHANNELS,
        recommended_sort_columns=(
            "fixation_duration",
            "saccade_amplitude",
            "saccade_duration",
            "gaze_x",
            "pupil",
            "epoch_index",
        ),
        official_source_examples={
            "version": "Raw data / task1 NR / YAG_NR1_EEG.mat and YAG_NR1_ET.mat; no ocular ICA removal.",
            "epoching": "Fixation-locked epochs use [-0.5, +2.0] s from continuous raw EEG.",
        },
        prepare=prepare_zuco2_reading_fixations,
    ),
    "02_new_raccoons_reading": DatasetConfig(
        key="02_new_raccoons_reading",
        component="Raccoons Dutch sentence-reading raw-synced fixations",
        source_component="https://data.ru.nl/collections/ru/cls/eeg_et_sentence_reading_dsc_556",
        source_processing_scripts="https://link.springer.com/article/10.1007/s10579-023-09684-x",
        reader_docs="https://docs.h5py.org/",
        preferred_channels=RACCOONS_PREFERRED_CHANNELS,
        recommended_sort_columns=(
            "fixation_duration",
            "surprisal",
            "N400_size",
            "word_length",
            "saccade_amplitude",
            "epoch_index",
        ),
        official_source_examples={
            "version": "EEG/Merged/EEG003.mat is raw EEG+ET synchronized by EYEEEG; ICA-corrected files are not used.",
            "epoching": "Left-eye fixation events are mapped to ET fixation/word tables and epoched to [-0.5, +2.0] s.",
        },
        prepare=prepare_raccoons_reading,
    ),
    "02_new_roamm_reading": DatasetConfig(
        key="02_new_roamm_reading",
        component="ROAMM reading fixations with attention labels",
        source_component="https://data-brain-mind.github.io/tutorials/reading-observed-at-mindless-moments-roamm-a-simultaneous-eeg-and-eye-tracking-dataset-of-natural-reading-with-attention-annotations/",
        source_processing_scripts="https://osf.io/kmvgb/",
        reader_docs="https://pandas.pydata.org/docs/reference/api/pandas.read_pickle.html",
        preferred_channels=ROAMM_PREFERRED_CHANNELS,
        recommended_sort_columns=(
            "fixation_duration",
            "attention_state",
            "pupil",
            "gaze_x",
            "gaze_y",
            "epoch_index",
        ),
        official_source_examples={
            "events": "s10014_run1_ml_data.pkl stores aligned EEG and expanded fixation labels.",
            "epoching": "First-pass fixation-locked epochs use [-0.5, +2.0] s and keep frontal/peripheral channels.",
        },
        prepare=prepare_roamm_reading_fixations,
    ),
    "02_new_unfold_facefreeview": DatasetConfig(
        key="02_new_unfold_facefreeview",
        component="Unfold face free-viewing saccades",
        source_component="https://osf.io/wbz7x/",
        source_processing_scripts="https://github.com/unfoldtoolbox/unfold",
        reader_docs=MNE_EEGLAB_RAW_DOCS,
        preferred_channels=UNFOLD_PREFERRED_CHANNELS,
        recommended_sort_columns=(
            "fixation_duration",
            "saccade_amplitude",
            "saccade_duration",
            "face_condition",
            "saccade_latency_ms",
            "epoch_index",
        ),
        official_source_examples={
            "version": "face_saccades_opendata_fig10.set contains synchronized continuous EEG and face/saccade/button events; deconvolved residuals are not used.",
            "epoching": "Saccade-locked raw continuous epochs use [-0.5, +2.0] s.",
        },
        prepare=prepare_unfold_facefreeview,
    ),
    "02_new_eeget_rsod": DatasetConfig(
        key="02_new_eeget_rsod",
        component="EEGET-RSOD raw visual-search fixations",
        source_component=EEGET_RSOD_SOURCE_URL,
        source_processing_scripts="https://github.com/Bing-1997/EEGET_RSOD",
        reader_docs="manual raw EEG + ET zip import",
        preferred_channels=("Fp1", "Fp2", "F7", "F8", "FCz", "Cz", "Oz"),
        recommended_sort_columns=("fixation_duration", "saccade_amplitude", "target_present", "epoch_index"),
        official_source_examples={
            "version": "Figshare EEG.zip and ET.zip raw files are used; EEG_clean data.zip is intentionally excluded because ocular ICA was manually removed.",
            "synchronization": "Time Synchronization Data.xlsx supplies b for P01; fixation onsets use EEG sample = (ET_time - b) / 2000.",
        },
        prepare=prepare_eeget_rsod,
    ),
    "02_new_visuomotor_chenguang": DatasetConfig(
        key="02_new_visuomotor_chenguang",
        component="Chenguang visuomotor raw eye-linked task",
        source_component=VISUOMOTOR_SOURCE_URL,
        source_processing_scripts="https://github.com/Chenguang918/visuomotor",
        reader_docs="https://mne.tools/stable/generated/mne.io.read_raw_curry.html",
        preferred_channels=("FP1", "FP2", "F7", "F8", "FCZ", "CZ", "OZ"),
        recommended_sort_columns=("saccade_duration", "fixation_duration", "reaction_time", "epoch_index"),
        official_source_examples={
            "version": "OSF EEG/RAW Curry .cdt files are used; EEG/PRE task files are excluded because the pipeline includes ICA removal.",
            "events": "EYE/PRE supplies fixation/saccade events per trial and block; raw EEG annotation codes 81-86 anchor the six task blocks.",
        },
        prepare=prepare_visuomotor_chenguang,
    ),
    "mrc_ox_gonogo": DatasetConfig(
        key="mrc_ox_gonogo",
        component="MRC-Ox Go/No-Go/Conflict",
        source_component="https://data.mrc.ox.ac.uk/data-set/go-no-go",
        source_processing_scripts="https://data.mrc.ox.ac.uk/data-set/go-no-go",
        reader_docs="manual access",
        preferred_channels=CENTRAL_RESPONSE_CHANNELS,
        recommended_sort_columns=("reaction_time_ms", "condition", "epoch_index"),
        official_source_examples={},
        prepare=unavailable_importer(
            "The MRC-Ox download page requires registration/login before files are exposed; place the raw files locally before conversion."
        ),
    ),
    "confidence_perceptual_decisions": DatasetConfig(
        key="confidence_perceptual_decisions",
        component="Confidence in Perceptual Decisions",
        source_component="https://openneuro.org/datasets/ds002739",
        source_processing_scripts="https://github.com/OpenNeuroDatasets/ds002739",
        reader_docs="https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html",
        preferred_channels=("E01", "E02", "E03", "E04", "E05"),
        recommended_sort_columns=("decision_duration_ms", "confidence_rating", "accuracy", "epoch_index"),
        official_source_examples={
            "version": "OpenNeuro ds002739 sub-01 run-01 and run-02 EEG_data/EEG_events MATLAB files are used.",
            "epoching": "Two sub-01 runs are response-locked to [-1.3, +0.3] s so stimulus onset remains visible when sorting by decision duration.",
        },
        prepare=prepare_confidence_perceptual_decisions,
    ),
    "kilo_word_erp": DatasetConfig(
        key="kilo_word_erp",
        component="Kilo-Word ERP Database",
        source_component="https://osf.io/72b89/",
        source_processing_scripts="https://osf.io/72b89/",
        reader_docs="https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html",
        preferred_channels=("Cz", "CPz", "Pz", "POz"),
        recommended_sort_columns=("word_frequency", "concreteness", "word_length", "visual_complexity", "epoch_index"),
        official_source_examples={
            "version": "OSF KWORD_ERP_LEXICAL_DECISION_DGMH2015.txt and KWORD_VARIABLES_DGMH2015.txt are word-level averaged ERP tables.",
            "note": "Rows are word averages rather than raw single-trial EEG; each word is materialized as one ERP-image trial for notebook compatibility.",
        },
        prepare=prepare_kilo_word_erp,
    ),
}


def annotate_metadata(output_dir: Path, notes: list[str]) -> None:
    metadata_path = output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["notes"] = notes
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def build_dataset(output_root: Path, dataset_key: str) -> Path:
    config = DATASETS[dataset_key]
    bundles = config.prepare(output_root, config)
    output_dir = write_dataset_bundle(output_root / config.key, config, bundles)
    if dataset_key in {"nencki_symfonia_srt", "openneuro_gonogo_ds002680"}:
        annotate_metadata(
            output_dir,
            [
                "Response-locked representative public subset.",
                "Epochs are intentionally long before the response so the stimulus onset can form a sigmoid-like trace when sorted by reaction_time_ms.",
                "No baseline correction is applied in the notebook preview path.",
            ],
        )
    elif dataset_key in {"eegeyenet_saccades", "02_new_eegeyenet_saccades"}:
        annotate_metadata(
            output_dir,
            [
                "Saccade-locked representative EEGEyeNet minimally processed subset.",
                "Saccade amplitude is computed in gaze-pixel units from the accompanying physio gaze stream.",
                "The epoch window is long enough to include the next eye-movement transition in the post-onset image.",
                "No baseline correction is applied in the notebook preview path.",
            ],
        )
    elif dataset_key in {"roamm_reading_fixations", "02_new_roamm_reading"}:
        annotate_metadata(
            output_dir,
            [
                "Fixation-locked representative ROAMM subset from subject s10014 run 1.",
                "The source pickle already aligns 64-channel EEG and eye-tracking labels at 256 Hz.",
                "Only first-pass reading fixations are retained; no baseline correction is applied in the notebook preview path.",
            ],
        )
    elif dataset_key == "02_new_zuco2_reading_fixations":
        annotate_metadata(
            output_dir,
            [
                "Fixation-locked ZuCo 2.0 raw-data subset from subject YAG, natural-reading run 1.",
                "Uses raw EEG and raw ET files instead of Automagic/MARA ICA-corrected word-level exports.",
                "Only peripheral/edge channels are materialized to prioritize trials and long post-onset windows.",
            ],
        )
    elif dataset_key == "02_new_raccoons_reading":
        annotate_metadata(
            output_dir,
            [
                "Fixation-locked Dutch sentence-reading subset from raw synchronized EEG/ET participant EEG003.",
                "Uses EEG/Merged raw synchronized data, not EEG/Preprocessed ICA-corrected files.",
                f"The importer keeps the first {RACCOONS_MAX_EVENTS} valid word fixations to cap bundle size.",
            ],
        )
    elif dataset_key == "02_new_unfold_facefreeview":
        annotate_metadata(
            output_dir,
            [
                "Saccade-locked raw continuous Unfold face-freeviewing subset.",
                "Uses synchronized EEGLAB data directly; deconvolved predictions/residuals are not used.",
                f"The importer keeps the first {UNFOLD_MAX_EVENTS} valid saccades to cap bundle size.",
            ],
        )
    elif dataset_key == "saccade_onset_face_vr":
        annotate_metadata(
            output_dir,
            [
                "Saccade-locked representative VR free-viewing subset from preprocessed EEG subject new_full_data_1.",
                f"The importer keeps the first {SACCADE_VR_MAX_EVENTS} valid hand-labeled saccade events to cap bundle size.",
                "Saccade amplitude is computed as the angular distance between start/end gaze vectors; no baseline correction is applied in the notebook preview path.",
            ],
        )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory where dataset folders will be created.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "02_new_eegeyenet_saccades",
            "02_new_zuco2_reading_fixations",
            "02_new_raccoons_reading",
            "02_new_roamm_reading",
            "02_new_unfold_facefreeview",
        ],
        choices=sorted(DATASETS.keys()),
        help="Dataset keys to prepare.",
    )
    args = parser.parse_args()

    for dataset_key in args.datasets:
        print(f"[build] {dataset_key}")
        build_dataset(args.output_root, dataset_key)


if __name__ == "__main__":
    main()
