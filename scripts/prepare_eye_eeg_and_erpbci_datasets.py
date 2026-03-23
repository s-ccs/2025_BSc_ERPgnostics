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

    eeg_channels = [
        ch for ch in raw.ch_names
        if ch not in {"A1", "A2", "LO1", "LO2", "IO1", "IO2"}
    ]
    eeg = raw.get_data(picks=eeg_channels).astype(np.float32)
    epochs, kept_onsets, times_s = extract_epochs(
        eeg,
        fixations["sample_index"].to_numpy(dtype=int),
        raw.info["sfreq"],
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
        preferred_channels=("POz", "Pz", "PO7", "PO8", "Oz"),
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
