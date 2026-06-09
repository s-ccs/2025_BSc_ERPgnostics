#!/usr/bin/env python3
"""
Download additional public sources requested for week 15 and prepare the
bundle-compatible subsets that are useful for ERP-image comparisons.

This script does three things:
1. Materialize raw EYE-EEG example sources locally.
2. Build a fixation-locked bundle from the EYE-EEG reading dataset.
3. Build an ERP-BCI bundle from PhysioNet erpbci subject s01.
"""

from __future__ import annotations

import argparse
import os
import re
import zipfile
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from prepare_public_shortlist_datasets import (
    DEFAULT_OUTPUT_ROOT,
    DatasetConfig,
    MNE_EDF_DOCS,
    MNE_EEGLAB_RAW_DOCS,
    download_file,
    extract_epochs,
    finalize_subject_bundle,
    subject_label,
    write_dataset_bundle,
)


EYE_EEG_PAGE = "https://www.eyetracking-eeg.org/testdata.html"
EYE_EEG_FREEVIEWING_URL = "https://www.eyetracking-eeg.org/testdata/freeviewing.zip"
EYE_EEG_READING_URL = "https://www.eyetracking-eeg.org/testdata/reading.zip"
EYE_EEG_SCENEVIEW_URL = "https://www.eyetracking-eeg.org/testdata/sceneviewing_tobii.zip"
ERPBCI_ROOT = "https://physionet.org/files/erpbci/1.0.0"
ERPBCI_PAGE = "https://physionet.org/content/erpbci/1.0.0/"
ERPBCI_RUN_LABELS = [f"rc{idx:02d}" for idx in range(1, 6)]
FIXATION_TMIN_S = -0.5
FIXATION_DURATION_S = 1.5
MIN_FIXATION_DURATION_MS = 80.0


def extract_all(archive_path: Path, destination_dir: Path) -> list[Path]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(archive_path, "r") as zf:
        for member in zf.namelist():
            target = destination_dir / member
            if member.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not target.is_file():
                zf.extract(member, destination_dir)
            extracted.append(target)
    return extracted


def ensure_readme(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return os.path.relpath(path, root)


def prepare_eye_eeg_source_dir(
    output_root: Path,
    *,
    dataset_key: str,
    title: str,
    url: str,
    notes: list[str],
) -> Path:
    out_dir = output_root / dataset_key
    source_dir = out_dir / "source"
    archive_path = download_file(url, source_dir / "archive.zip")
    extract_all(archive_path, source_dir)
    ensure_readme(
        out_dir / "README.md",
        [
            f"# {title}",
            "",
            f"- Official source: {EYE_EEG_PAGE}",
            f"- Download URL: {url}",
            "- Files are kept as source material for later synchronized EEG/ET imports.",
            *notes,
        ],
    )
    return out_dir


def parse_reading_fixations(asc_path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    sync_times: list[int] = []
    fixation_rows: list[dict[str, float | int | str]] = []

    with asc_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if parts[0] == "MSG" and "MYKEYWORD" in line:
                sync_times.append(int(parts[1]))
            elif parts[0] == "EFIX" and len(parts) >= 8:
                eye = parts[1]
                if eye != "R":
                    continue
                fixation_rows.append(
                    {
                        "eye": eye,
                        "et_onset": int(parts[2]),
                        "et_offset": int(parts[3]),
                        "fixation_duration_ms": int(parts[4]),
                        "gaze_x": float(parts[5]),
                        "gaze_y": float(parts[6]),
                        "pupil": float(parts[7]),
                    }
                )

    fixations = pd.DataFrame(fixation_rows)
    fixations = fixations[fixations["fixation_duration_ms"] >= 80].reset_index(drop=True)
    return np.asarray(sync_times, dtype=np.int64), fixations


def normalized_trigger_code(value: object) -> int | None:
    text = str(value).strip()
    if not text or text.lower() == "boundary":
        return None
    match = re.search(r"(-?\d+)", text)
    if match is None:
        return None
    return int(match.group(1))


def collapse_sample_triggers(
    frame: pd.DataFrame,
    *,
    time_column: str,
    trigger_column: str,
    max_same_code_gap_s: float = 0.02,
) -> pd.DataFrame:
    trigger = pd.to_numeric(frame[trigger_column], errors="coerce").fillna(0).astype(int).to_numpy()
    time_s = pd.to_numeric(frame[time_column], errors="coerce").to_numpy(dtype=np.float64)
    rows: list[dict[str, float | int]] = []
    last_code = 0
    last_time = np.nan
    for code, t_s in zip(trigger, time_s, strict=False):
        if code == 0 or not np.isfinite(t_s):
            last_code = 0
            last_time = t_s
            continue
        starts_new = (
            last_code == 0
            or code != last_code
            or not np.isfinite(last_time)
            or (t_s - last_time) > max_same_code_gap_s
        )
        if starts_new:
            rows.append({"trigger_code": int(code), "et_time_s": float(t_s)})
        last_code = int(code)
        last_time = float(t_s)
    return pd.DataFrame(rows)


def eeg_trigger_frame(raw: mne.io.BaseRaw) -> pd.DataFrame:
    rows = []
    for ann in raw.annotations:
        code = normalized_trigger_code(ann["description"])
        if code is None:
            continue
        rows.append({"trigger_code": code, "eeg_onset_s": float(ann["onset"])})
    return pd.DataFrame(rows)


def fit_et_to_eeg_time(et_events: pd.DataFrame, eeg_events: pd.DataFrame, source_name: str) -> tuple[float, float]:
    if len(et_events) != len(eeg_events):
        raise RuntimeError(
            f"{source_name} sync mismatch: ET={len(et_events)} vs EEG={len(eeg_events)}. "
            "Check trigger parsing before epoch extraction."
        )
    et_codes = et_events["trigger_code"].astype(int).to_numpy()
    eeg_codes = eeg_events["trigger_code"].astype(int).to_numpy()
    if not np.array_equal(et_codes, eeg_codes):
        first_mismatch = int(np.flatnonzero(et_codes != eeg_codes)[0])
        raise RuntimeError(
            f"{source_name} sync code mismatch at index {first_mismatch}: "
            f"ET={et_codes[first_mismatch]} vs EEG={eeg_codes[first_mismatch]}"
        )
    slope, intercept = np.polyfit(
        et_events["et_time_s"].to_numpy(dtype=np.float64),
        eeg_events["eeg_onset_s"].to_numpy(dtype=np.float64),
        deg=1,
    )
    return float(slope), float(intercept)


def extract_idt_fixations(
    samples: pd.DataFrame,
    *,
    time_s_column: str,
    x_column: str,
    y_column: str,
    pupil_column: str | None = None,
    trial_column: str | None = None,
    min_duration_ms: float = MIN_FIXATION_DURATION_MS,
    dispersion_threshold_px: float = 80.0,
    smooth_samples: int = 5,
) -> pd.DataFrame:
    time_s = pd.to_numeric(samples[time_s_column], errors="coerce").to_numpy(dtype=np.float64)
    x = pd.to_numeric(samples[x_column], errors="coerce").replace(0, np.nan).to_numpy(dtype=np.float64)
    y = pd.to_numeric(samples[y_column], errors="coerce").replace(0, np.nan).to_numpy(dtype=np.float64)
    if smooth_samples > 1:
        x = pd.Series(x).rolling(smooth_samples, center=True, min_periods=1).median().to_numpy(dtype=np.float64)
        y = pd.Series(y).rolling(smooth_samples, center=True, min_periods=1).median().to_numpy(dtype=np.float64)

    pupil = None
    if pupil_column is not None and pupil_column in samples.columns:
        pupil = pd.to_numeric(samples[pupil_column], errors="coerce").replace(0, np.nan).to_numpy(dtype=np.float64)
    trial_values = None
    if trial_column is not None and trial_column in samples.columns:
        trial_values = samples[trial_column].to_numpy()

    min_duration_s = float(min_duration_ms) / 1000.0
    rows: list[dict[str, float | int | str]] = []
    n_samples = len(samples)
    idx = 0
    while idx < n_samples:
        if not (np.isfinite(time_s[idx]) and np.isfinite(x[idx]) and np.isfinite(y[idx])):
            idx += 1
            continue

        stop = idx
        while stop < n_samples and (time_s[stop] - time_s[idx]) < min_duration_s:
            stop += 1
        if stop >= n_samples:
            break

        window_x = x[idx : stop + 1]
        window_y = y[idx : stop + 1]
        if np.any(~np.isfinite(window_x)) or np.any(~np.isfinite(window_y)):
            idx += 1
            continue

        dispersion = (np.nanmax(window_x) - np.nanmin(window_x)) + (np.nanmax(window_y) - np.nanmin(window_y))
        if dispersion > dispersion_threshold_px:
            idx += 1
            continue

        extend = stop + 1
        while extend < n_samples and np.isfinite(x[extend]) and np.isfinite(y[extend]):
            candidate_x = x[idx : extend + 1]
            candidate_y = y[idx : extend + 1]
            dispersion = (np.nanmax(candidate_x) - np.nanmin(candidate_x)) + (
                np.nanmax(candidate_y) - np.nanmin(candidate_y)
            )
            if dispersion > dispersion_threshold_px:
                break
            extend += 1

        end = extend - 1
        duration_ms = (time_s[end] - time_s[idx]) * 1000.0
        if duration_ms >= min_duration_ms:
            row: dict[str, float | int | str] = {
                "et_onset_s": float(time_s[idx]),
                "et_offset_s": float(time_s[end]),
                "fixation_duration_ms": float(duration_ms),
                "gaze_x": float(np.nanmean(x[idx : end + 1])),
                "gaze_y": float(np.nanmean(y[idx : end + 1])),
            }
            if pupil is not None:
                row["pupil"] = float(np.nanmean(pupil[idx : end + 1]))
            if trial_values is not None:
                row["trial_block_index"] = trial_values[idx]
            rows.append(row)
        idx = extend

    fixations = pd.DataFrame(rows)
    if fixations.empty:
        raise RuntimeError("No valid fixations could be extracted from eye-tracker samples.")
    return fixations


def prepare_sample_fixation_bundle(
    *,
    output_root: Path,
    config: DatasetConfig,
    source_dataset_key: str,
    set_name: str,
    et_name: str,
    sample_reader,
    component_session_label: str,
    source_note: str,
) -> list:
    output_dir = output_root / config.key
    source_dir = output_root / source_dataset_key / "source"
    set_path = source_dir / set_name
    et_path = source_dir / et_name
    if not set_path.is_file() or not et_path.is_file():
        raise FileNotFoundError(
            f"Missing local EYE-EEG source files for {config.key}. "
            f"Expected {set_path} and {et_path}. Run this script once without "
            "the fixation dataset or keep the source bundles in notebooks/datasets."
        )

    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")
    samples, et_events, fixations = sample_reader(et_path)
    eeg_events = eeg_trigger_frame(raw)
    slope, intercept = fit_et_to_eeg_time(et_events, eeg_events, config.key)

    fixations["eeg_onset_s"] = slope * fixations["et_onset_s"].astype(np.float64) + intercept
    fixations["sample_index"] = np.rint(fixations["eeg_onset_s"] * raw.info["sfreq"]).astype(int)
    fixations["fixation_index"] = np.arange(1, len(fixations) + 1, dtype=int)

    eeg_channels = list(raw.ch_names)
    eeg = raw.get_data(picks=eeg_channels).astype(np.float32)
    epochs, kept_onsets, times_s = extract_epochs(
        eeg,
        fixations["sample_index"].to_numpy(dtype=int),
        raw.info["sfreq"],
        tmin_s=FIXATION_TMIN_S,
        duration_s=FIXATION_DURATION_S,
    )
    keep_mask = fixations["sample_index"].isin(set(int(v) for v in kept_onsets))
    kept = fixations.loc[keep_mask].copy().reset_index(drop=True)
    kept["source_file"] = safe_relpath(set_path, output_dir)
    kept["subject_id"] = 1
    kept["subject_label"] = subject_label(1)
    kept["session_label"] = component_session_label
    kept["run_label"] = "fixation_locked"
    kept["condition"] = "fixation"
    kept["eye"] = "binocular"
    kept["source_note"] = source_note

    return [
        finalize_subject_bundle(
            subject_id=1,
            channel_names=eeg_channels,
            sfreq_hz=float(raw.info["sfreq"]),
            epochs_parts=[epochs],
            event_frames=[
                kept[
                    [
                        "subject_id",
                        "subject_label",
                        "session_label",
                        "run_label",
                        "trial_block_index",
                        "fixation_index",
                        "fixation_duration_ms",
                        "eye",
                        "gaze_x",
                        "gaze_y",
                        "pupil",
                        "sample_index",
                        "condition",
                        "source_note",
                        "source_file",
                    ]
                ]
            ],
            source_relpaths=[safe_relpath(set_path, output_dir), safe_relpath(et_path, output_dir)],
            times_s=times_s,
        )
    ]


def read_freeviewing_samples(et_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    samples = pd.read_csv(et_path, sep="\t", comment="#")
    samples["et_time_s"] = pd.to_numeric(samples["Time"], errors="coerce") / 1_000_000.0
    samples["gaze_x"] = samples[["L POR X [px]", "R POR X [px]"]].replace(0, np.nan).mean(axis=1)
    samples["gaze_y"] = samples[["L POR Y [px]", "R POR Y [px]"]].replace(0, np.nan).mean(axis=1)
    samples["pupil"] = samples[["L Dia [mm]", "R Dia [mm]"]].replace(0, np.nan).mean(axis=1)
    et_events = collapse_sample_triggers(samples, time_column="et_time_s", trigger_column="Trigger")
    fixations = extract_idt_fixations(
        samples,
        time_s_column="et_time_s",
        x_column="gaze_x",
        y_column="gaze_y",
        pupil_column="pupil",
        trial_column="Trial",
        dispersion_threshold_px=80.0,
    )
    return samples, et_events, fixations


def read_tobii_scene_samples(et_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    samples = pd.read_csv(et_path, sep="\t")
    samples["et_time_s"] = pd.to_numeric(samples["Recording timestamp"], errors="coerce") / 1_000_000.0
    samples["gaze_x"] = samples[["Gaze2d_Left.x", "Gaze2d_Right.x"]].replace(0, np.nan).mean(axis=1)
    samples["gaze_y"] = samples[["Gaze2d_Left.y", "Gaze2d_Right.y"]].replace(0, np.nan).mean(axis=1)
    samples["pupil"] = samples[["PupilDiam_Left", "PupilDiam_Right"]].replace(0, np.nan).mean(axis=1)
    et_events = samples[pd.to_numeric(samples["Event value"], errors="coerce").fillna(0).astype(int) != 0][
        ["et_time_s", "Event value"]
    ].copy()
    et_events.rename(columns={"Event value": "trigger_code"}, inplace=True)
    et_events["trigger_code"] = et_events["trigger_code"].astype(int)
    fixations = extract_idt_fixations(
        samples,
        time_s_column="et_time_s",
        x_column="gaze_x",
        y_column="gaze_y",
        pupil_column="pupil",
        dispersion_threshold_px=80.0,
    )
    fixations["trial_block_index"] = np.searchsorted(
        et_events.loc[et_events["trigger_code"].isin([1, 2, 3]), "et_time_s"].to_numpy(),
        fixations["et_onset_s"].to_numpy(),
        side="right",
    ).astype(int)
    return samples, et_events.reset_index(drop=True), fixations


def prepare_eye_eeg_freeviewing_fixations(output_root: Path, config: DatasetConfig):
    prepare_eye_eeg_source_dir(
        output_root,
        dataset_key="eye_eeg_freeviewing_source",
        title="EYE-EEG Freeviewing Source",
        url=EYE_EEG_FREEVIEWING_URL,
        notes=[
            "- This source is converted into `eye_eeg_freeviewing_fixations` by extracting I-DT fixations from the SMI sample stream.",
            "- EEG trigger codes S103/S12/S1/S99/S203 are aligned to collapsed eye-tracker trigger samples.",
        ],
    )
    return prepare_sample_fixation_bundle(
        output_root=output_root,
        config=config,
        source_dataset_key="eye_eeg_freeviewing_source",
        set_name="EEG_freeviewing_25channels.set",
        et_name="eyetracker_freeviewing.txt",
        sample_reader=read_freeviewing_samples,
        component_session_label="freeviewing",
        source_note="SMI iView samples; I-DT fixation extraction after trigger-based EEG/ET synchronization.",
    )


def prepare_eye_eeg_sceneviewing_tobii_fixations(output_root: Path, config: DatasetConfig):
    prepare_eye_eeg_source_dir(
        output_root,
        dataset_key="eye_eeg_sceneviewing_tobii_source",
        title="EYE-EEG Sceneviewing Tobii Source",
        url=EYE_EEG_SCENEVIEW_URL,
        notes=[
            "- This source is converted into `eye_eeg_sceneviewing_tobii_fixations` by extracting I-DT fixations from Tobii samples.",
            "- MYKEYWORD event values are aligned to matching EEG trigger annotations.",
        ],
    )
    return prepare_sample_fixation_bundle(
        output_root=output_root,
        config=config,
        source_dataset_key="eye_eeg_sceneviewing_tobii_source",
        set_name="tobii_sceneviewing_eeg.set",
        et_name="tobii_sceneviewing_eyetrack_ascii.txt",
        sample_reader=read_tobii_scene_samples,
        component_session_label="sceneviewing_tobii",
        source_note="Tobii TX-300 samples; I-DT fixation extraction after MYKEYWORD EEG/ET synchronization.",
    )


def prepare_eye_eeg_reading_fixations(output_root: Path, config: DatasetConfig):
    output_dir = output_root / config.key
    source_dir = output_dir / "source"
    archive_path = download_file(EYE_EEG_READING_URL, source_dir / "reading.zip")
    extract_all(archive_path, source_dir)

    set_path = source_dir / "reading_eeg.set"
    asc_path = source_dir / "reading_eyelink.asc"
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")

    eeg_sync_onsets_s = np.asarray(
        [float(ann["onset"]) for ann in raw.annotations if str(ann["description"]).strip() == "3"],
        dtype=np.float64,
    )
    et_sync_times, fixations = parse_reading_fixations(asc_path)
    if len(et_sync_times) != len(eeg_sync_onsets_s):
        raise RuntimeError(
            f"EYE-EEG reading sync mismatch: ET={len(et_sync_times)} vs EEG={len(eeg_sync_onsets_s)}"
        )

    slope, intercept = np.polyfit(et_sync_times.astype(np.float64), eeg_sync_onsets_s, deg=1)
    fixations["eeg_onset_s"] = slope * fixations["et_onset"].astype(np.float64) + intercept
    fixations["sample_index"] = np.rint(fixations["eeg_onset_s"] * raw.info["sfreq"]).astype(int)

    eeg_channels = [ch for ch in raw.ch_names if ch not in {"A1", "A2"}]
    eeg = raw.get_data(picks=eeg_channels).astype(np.float32)
    epochs, kept_onsets, times_s = extract_epochs(
        eeg,
        fixations["sample_index"].to_numpy(dtype=int),
        raw.info["sfreq"],
        tmin_s=-0.5,
        duration_s=1.5,
    )

    keep_mask = fixations["sample_index"].isin(set(int(v) for v in kept_onsets))
    kept = fixations.loc[keep_mask].copy().reset_index(drop=True)

    task_block_onsets_s = eeg_sync_onsets_s
    kept["trial_block_index"] = np.searchsorted(task_block_onsets_s, kept["eeg_onset_s"], side="right").astype(int)
    kept["source_file"] = str(set_path.relative_to(output_dir))
    kept["subject_id"] = 1
    kept["subject_label"] = subject_label(1)
    kept["session_label"] = "reading"
    kept["run_label"] = "fixation_locked"
    kept["condition"] = "fixation"

    scalp_channels = list(eeg_channels)
    return [
        finalize_subject_bundle(
            subject_id=1,
            channel_names=scalp_channels,
            sfreq_hz=float(raw.info["sfreq"]),
            epochs_parts=[epochs],
            event_frames=[
                kept[
                    [
                        "subject_id",
                        "subject_label",
                        "session_label",
                        "run_label",
                        "trial_block_index",
                        "fixation_duration_ms",
                        "eye",
                        "gaze_x",
                        "gaze_y",
                        "pupil",
                        "sample_index",
                        "condition",
                        "source_file",
                    ]
                ]
            ],
            source_relpaths=[str(set_path.relative_to(output_dir)), str(asc_path.relative_to(output_dir))],
            times_s=times_s,
        )
    ]


def parse_erpbci_flash_events(raw: mne.io.BaseRaw, run_label: str) -> pd.DataFrame:
    target_char = None
    rows: list[dict[str, object]] = []
    for ann in raw.annotations:
        onset_s = float(ann["onset"])
        desc = str(ann["description"]).strip()
        if desc.startswith("#Tgt"):
            match = re.match(r"#Tgt(.?)_", desc)
            if match:
                target_char = match.group(1)
            continue
        if desc.startswith("#"):
            continue
        if not desc:
            continue
        if target_char is None:
            raise RuntimeError(f"Missing target annotation before flash events in {run_label}")

        rows.append(
            {
                "run_label": run_label,
                "sample_index": int(round(onset_s * raw.info["sfreq"])),
                "flash_string": desc,
                "target_char": target_char,
                "condition": "target" if target_char in desc else "non_target",
            }
        )
    frame = pd.DataFrame(rows)
    frame["flash_index_within_run"] = np.arange(1, len(frame) + 1, dtype=int)
    return frame


def prepare_erpbci_public(output_root: Path, config: DatasetConfig):
    output_dir = output_root / config.key
    source_dir = output_dir / "source" / "s01"

    epochs_parts = []
    event_frames = []
    relpaths: list[str] = []
    times_s_ref: np.ndarray | None = None

    for run_label in ERPBCI_RUN_LABELS:
        edf_path = download_file(f"{ERPBCI_ROOT}/s01/{run_label}.edf", source_dir / f"{run_label}.edf")
        raw = mne.io.read_raw_edf(edf_path, preload=True, infer_types=True, verbose="ERROR")

        ref_signal = raw.get_data(picks=["EARL", "EARR"]).mean(axis=0, keepdims=True)
        eeg_channels = [ch for ch in raw.ch_names if ch not in {"EARL", "EARR", "VEOGL", "VEOGR", "HEOGL", "HEOGR"}]
        eeg = raw.get_data(picks=eeg_channels).astype(np.float32) - ref_signal.astype(np.float32)

        eeg_info = mne.create_info(ch_names=eeg_channels, sfreq=raw.info["sfreq"], ch_types=["eeg"] * len(eeg_channels))
        eeg_raw = mne.io.RawArray(eeg, eeg_info, verbose="ERROR")
        eeg_raw.filter(l_freq=0.1, h_freq=20.0, verbose="ERROR")
        eeg_raw.resample(256, verbose="ERROR")

        events = parse_erpbci_flash_events(raw, run_label)
        sample_scale = eeg_raw.info["sfreq"] / raw.info["sfreq"]
        events["sample_index"] = np.rint(events["sample_index"].astype(np.float64) * sample_scale).astype(int)

        epochs, kept_onsets, times_s = extract_epochs(
            eeg_raw.get_data().astype(np.float32),
            events["sample_index"].to_numpy(dtype=int),
            eeg_raw.info["sfreq"],
        )
        times_s_ref = times_s if times_s_ref is None else times_s_ref
        keep_mask = events["sample_index"].isin(set(int(v) for v in kept_onsets))
        kept = events.loc[keep_mask].copy().reset_index(drop=True)
        kept["source_file"] = str(edf_path.relative_to(output_dir))
        kept["subject_id"] = 1
        kept["subject_label"] = subject_label(1)
        kept["session_label"] = "s01"

        epochs_parts.append(epochs)
        event_frames.append(
            kept[
                [
                    "subject_id",
                    "subject_label",
                    "session_label",
                    "run_label",
                    "flash_index_within_run",
                    "target_char",
                    "flash_string",
                    "sample_index",
                    "condition",
                    "source_file",
                ]
            ]
        )
        relpaths.append(str(edf_path.relative_to(output_dir)))

    return [
        finalize_subject_bundle(
            subject_id=1,
            channel_names=eeg_channels,
            sfreq_hz=256.0,
            epochs_parts=epochs_parts,
            event_frames=event_frames,
            source_relpaths=relpaths,
            times_s=times_s_ref,
        )
    ]


DATASETS = {
    "eye_eeg_reading_fixations": DatasetConfig(
        key="eye_eeg_reading_fixations",
        component="EYE-EEG Reading Fixations",
        source_component=EYE_EEG_PAGE,
        source_processing_scripts=EYE_EEG_PAGE,
        reader_docs=MNE_EEGLAB_RAW_DOCS,
        preferred_channels=("LO1", "LO2", "IO1", "IO2", "POz", "Pz", "PO7", "PO8", "Oz"),
        recommended_sort_columns=(
            "fixation_duration_ms",
            "trial_block_index",
            "gaze_x",
            "gaze_y",
            "pupil",
            "sample_index",
            "epoch_index",
        ),
        official_source_examples={
            "sync_note": "The reading example uses MYKEYWORD messages in the EyeLink ASC file and repeated EEG event code 3 for synchronization.",
            "bundle_note": "This bundle uses right-eye EFIX events, linearly mapped into EEG time and epoched fixation-locked.",
            "julia_import": "Julia reference: NeuroAnalyzer.jl supports EEGLAB imports via import_set(); the eye tracker ASC file still needs custom parsing and synchronization.",
        },
        prepare=prepare_eye_eeg_reading_fixations,
    ),
    "eye_eeg_freeviewing_fixations": DatasetConfig(
        key="eye_eeg_freeviewing_fixations",
        component="EYE-EEG Freeviewing Fixations",
        source_component=EYE_EEG_PAGE,
        source_processing_scripts=EYE_EEG_PAGE,
        reader_docs=MNE_EEGLAB_RAW_DOCS,
        preferred_channels=("LO1", "LO2", "IO1", "IO2", "Fz", "Cz", "Pz", "Oz"),
        recommended_sort_columns=(
            "fixation_duration_ms",
            "trial_block_index",
            "gaze_x",
            "gaze_y",
            "pupil",
            "sample_index",
            "epoch_index",
        ),
        official_source_examples={
            "sync_note": "The freeviewing sample stream carries SMI trigger values that match EEG annotations S103/S12/S1/S99/S203.",
            "bundle_note": "This bundle keeps the EOG channels and uses I-DT fixation extraction on binocular SMI gaze samples.",
            "epoch_window": f"Fixation-locked epochs use tmin={FIXATION_TMIN_S}s and duration={FIXATION_DURATION_S}s without baseline correction in the notebook path.",
        },
        prepare=prepare_eye_eeg_freeviewing_fixations,
    ),
    "eye_eeg_sceneviewing_tobii_fixations": DatasetConfig(
        key="eye_eeg_sceneviewing_tobii_fixations",
        component="EYE-EEG Sceneviewing Tobii Fixations",
        source_component=EYE_EEG_PAGE,
        source_processing_scripts=EYE_EEG_PAGE,
        reader_docs=MNE_EEGLAB_RAW_DOCS,
        preferred_channels=("LO1", "LO2", "AFz"),
        recommended_sort_columns=(
            "fixation_duration_ms",
            "trial_block_index",
            "gaze_x",
            "gaze_y",
            "pupil",
            "sample_index",
            "epoch_index",
        ),
        official_source_examples={
            "sync_note": "The Tobii text export contains MYKEYWORD event values that match EEG trigger annotations.",
            "bundle_note": "This bundle keeps all provided EEG/EOG channels and uses I-DT fixation extraction on binocular Tobii gaze samples.",
            "epoch_window": f"Fixation-locked epochs use tmin={FIXATION_TMIN_S}s and duration={FIXATION_DURATION_S}s without baseline correction in the notebook path.",
        },
        prepare=prepare_eye_eeg_sceneviewing_tobii_fixations,
    ),
    "erpbci_public": DatasetConfig(
        key="erpbci_public",
        component="PhysioNet ERPBCI",
        source_component=ERPBCI_PAGE,
        source_processing_scripts=ERPBCI_PAGE,
        reader_docs=MNE_EDF_DOCS,
        preferred_channels=("Pz", "POz", "CPz", "Oz", "P3"),
        recommended_sort_columns=(
            "condition",
            "run_label",
            "target_char",
            "flash_string",
            "flash_index_within_run",
            "sample_index",
            "epoch_index",
        ),
        official_source_examples={
            "import_note": "The PhysioNet ERPBCI release stores flash intensifications in EDF annotations and recommends software re-referencing using EARL and EARR.",
            "bundle_note": "This bundle uses subject s01 runs rc01-rc05, re-references to the ear electrodes, filters 0.1-20 Hz, resamples to 256 Hz, and epochs flash events.",
            "julia_import": "Julia reference: NeuroAnalyzer.jl supports EDF imports via import_edf(); annotation parsing and target/non-target labeling still need custom event handling.",
        },
        prepare=prepare_erpbci_public,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS.keys()),
        choices=sorted(DATASETS.keys()),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def bundle_exists(output_dir: Path) -> bool:
    return all((output_dir / name).is_file() for name in ["epochs.hdf5", "events.csv", "metadata.json"])


def main() -> None:
    args = parse_args()
    mne.set_log_level("ERROR")

    prepare_eye_eeg_source_dir(
        args.output_root,
        dataset_key="eye_eeg_freeviewing_source",
        title="EYE-EEG Freeviewing Source",
        url=EYE_EEG_FREEVIEWING_URL,
        notes=[
            "- EEG is provided as EEGLAB .set/.fdt and the eye tracker export as plain text.",
            "- Only five task trials are present, so this source is kept raw for later custom ET-driven epoching.",
        ],
    )
    prepare_eye_eeg_source_dir(
        args.output_root,
        dataset_key="eye_eeg_sceneviewing_tobii_source",
        title="EYE-EEG Sceneviewing Tobii Source",
        url=EYE_EEG_SCENEVIEW_URL,
        notes=[
            "- EEG is provided as EEGLAB .set and Tobii samples as plain text with MYKEYWORD sync messages.",
            "- No precomputed fixation/saccade event stream is included, so this source is kept raw for later Tobii-specific event parsing.",
        ],
    )

    for dataset_key in args.datasets:
        config = DATASETS[dataset_key]
        output_dir = args.output_root / config.key
        if bundle_exists(output_dir) and not args.force:
            print(f"[skip] {config.key} already exists")
            continue
        print(f"[build] {config.key}")
        bundles = config.prepare(args.output_root, config)
        write_dataset_bundle(output_dir, config, bundles)
        print(f"[done] {output_dir}")


if __name__ == "__main__":
    main()
