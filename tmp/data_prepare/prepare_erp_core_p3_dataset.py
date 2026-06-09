#!/usr/bin/env python3
"""
Download processed ERP CORE P3 subject files from OSF and convert them into
an internal HDF5/CSV layout for the Julia notebooks in `notebooks/model_test`.

Required Python packages:
    mne, h5py, numpy, pandas, scipy

Primary sources used by this script:
    - ERP CORE OSF P3 component: https://osf.io/etdkz/
    - ERP CORE GitHub P3 processing scripts:
      https://github.com/lucklab/ERP_CORE/tree/master/P3
    - MNE EEGLAB epoch reader:
      https://mne.tools/stable/generated/mne.read_epochs_eeglab.html
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import h5py
import mne
import numpy as np
import pandas as pd


OSF_P3_ALL_DATA_ROOT = (
    "https://files.osf.io/v1/resources/etdkz/providers/osfstorage/5f247351b084f60115c9aa10/"
)
OSF_P3_COMPONENT = "https://osf.io/etdkz/"
ERP_CORE_GITHUB_P3 = "https://github.com/lucklab/ERP_CORE/tree/master/P3"
MNE_EEGLAB_DOCS = "https://mne.tools/stable/generated/mne.read_epochs_eeglab.html"

EPOCH_SET_TEMPLATE = (
    "{sid}_P3_shifted_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins_epoch_interp_ar.set"
)
EPOCH_FDT_TEMPLATE = (
    "{sid}_P3_shifted_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins_epoch_interp_ar.fdt"
)
EVENTLIST_TEMPLATE = "{sid}_P3_Eventlist_For_RTs.txt"

EVENT_LINE_RE = re.compile(
    r'^\s*(\d+)\s+(\d+)\s+(-?\d+)\s+"([^"]*)"\s+'
    r'(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+'
    r'([01]+)\s+([01]+)\s+(\d+)\s+\[(.*?)\]\s*$'
)


@dataclass(frozen=True)
class StimulusEvent:
    epoch_index: int
    stimulus_code: int
    response_code: int
    reaction_time_ms: float
    condition: str
    block_target_code: int
    trial_stimulus_code: int
    stimulus_onset_s: float
    response_onset_s: float
    source_event_item: int


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url) as response:
        return json.load(response)


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


def subject_label(subject_id: int) -> str:
    return f"sub-{subject_id:03d}"


def parse_eventlist_for_rt(path: Path) -> list[StimulusEvent]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = EVENT_LINE_RE.match(line)
            if not match:
                continue
            (
                item,
                bepoch,
                ecode,
                _label,
                onset_s,
                diff_ms,
                _dura_ms,
                _b_flags,
                _a_flags,
                _enable,
                bins_raw,
            ) = match.groups()
            bins = [int(token) for token in bins_raw.split()] if bins_raw.strip() else []
            rows.append(
                {
                    "item": int(item),
                    "bepoch": int(bepoch),
                    "ecode": int(ecode),
                    "onset_s": float(onset_s),
                    "diff_ms": float(diff_ms),
                    "bins": bins,
                }
            )

    stimuli: list[StimulusEvent] = []
    for idx, row in enumerate(rows[:-1]):
        if row["bepoch"] <= 0:
            continue
        if row["ecode"] < 10 or row["ecode"] > 99:
            continue

        after_ar_bins = [b for b in row["bins"] if b in {1, 2}]
        if not after_ar_bins:
            continue

        response = rows[idx + 1]
        if response["ecode"] not in {201, 202}:
            raise RuntimeError(
                f"Expected response after stimulus item {row['item']} in {path}, "
                f"got event code {response['ecode']}"
            )

        code = int(row["ecode"])
        block_target = code // 10
        trial_stimulus = code % 10
        condition = "rare" if after_ar_bins[0] == 1 else "frequent"

        stimuli.append(
            StimulusEvent(
                epoch_index=int(row["bepoch"]),
                stimulus_code=code,
                response_code=int(response["ecode"]),
                reaction_time_ms=float(response["diff_ms"]),
                condition=condition,
                block_target_code=block_target,
                trial_stimulus_code=trial_stimulus,
                stimulus_onset_s=float(row["onset_s"]),
                response_onset_s=float(response["onset_s"]),
                source_event_item=int(row["item"]),
            )
        )

    stimuli.sort(key=lambda event: event.epoch_index)
    return stimuli


def build_dataset(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_root = output_dir / "source" / "P3_All_Data_and_Scripts"
    subjects_source_dir = source_root / "subjects"
    root_listing = fetch_json(OSF_P3_ALL_DATA_ROOT)

    download_named_items(
        root_listing["data"],
        source_root,
        names=["README_P3.txt", "P3_Subject_Summary.xlsx", "Participant_Demographics.xlsx"],
    )

    behavior_item = next(
        item for item in root_listing["data"] if item["attributes"]["name"] == "Behavior_Measurements"
    )
    behavior_listing = fetch_json(behavior_item["links"]["move"])
    download_named_items(
        behavior_listing["data"],
        source_root / "Behavior_Measurements",
        names=["BDF_P3_RTs.txt", "Script14_Calculate_RTs_and_Accuracy.m"],
    )

    subject_items = {
        int(item["attributes"]["name"]): item
        for item in root_listing["data"]
        if item["attributes"]["kind"] == "folder" and item["attributes"]["name"].isdigit()
    }

    events_rows: list[dict] = []
    h5_path = output_dir / "epochs.hdf5"
    metadata_path = output_dir / "metadata.json"
    events_path = output_dir / "events_rt.csv"
    readme_path = output_dir / "README.md"

    with h5py.File(h5_path, "w") as h5:
        h5.attrs["source_component"] = OSF_P3_COMPONENT
        h5.attrs["source_root_listing"] = OSF_P3_ALL_DATA_ROOT
        h5.attrs["layout"] = "subjects/<sub>/epochs (channels, time, trial)"
        h5.attrs["reader"] = "mne.read_epochs_eeglab"
        subjects_group = h5.create_group("subjects")

        for subject_id in range(1, 41):
            subject_item = subject_items[subject_id]
            subject_listing = fetch_json(subject_item["links"]["move"])
            subject_source_dir = subjects_source_dir / subject_label(subject_id)

            required_names = [
                EPOCH_SET_TEMPLATE.format(sid=subject_id),
                EPOCH_FDT_TEMPLATE.format(sid=subject_id),
                EVENTLIST_TEMPLATE.format(sid=subject_id),
            ]
            download_named_items(subject_listing["data"], subject_source_dir, names=required_names)

            set_path = subject_source_dir / required_names[0]
            eventlist_path = subject_source_dir / required_names[2]
            epochs = mne.read_epochs_eeglab(set_path, verbose="ERROR")
            data = epochs.get_data(copy=True).astype(np.float32)  # (trial, channel, time)
            # HDF5 written from NumPy is read back by Julia/HDF5.jl with reversed
            # axis order. Store the array as (trial, time, channel) so Julia sees
            # it as (channel, time, trial), which matches the rest of this repo.
            data = np.transpose(data, (0, 2, 1))  # Julia reads this as (channel, time, trial)
            ch_names = np.asarray(epochs.ch_names, dtype=h5py.string_dtype(encoding="utf-8"))
            times_s = np.asarray(epochs.times, dtype=np.float32)

            rt_events = parse_eventlist_for_rt(eventlist_path)
            if len(rt_events) != len(epochs):
                raise RuntimeError(
                    f"Subject {subject_id}: event list count {len(rt_events)} "
                    f"does not match epochs {len(epochs)}"
                )

            group = subjects_group.create_group(subject_label(subject_id))
            group.create_dataset("epochs", data=data, compression="gzip", compression_opts=4)
            group.create_dataset("times_s", data=times_s)
            group.create_dataset("channel_names", data=ch_names)
            group.attrs["subject_id"] = subject_id
            group.attrs["subject_label"] = subject_label(subject_id)
            group.attrs["sfreq_hz"] = float(epochs.info["sfreq"])
            group.attrs["n_channels"] = int(len(epochs.ch_names))
            group.attrs["n_timepoints"] = int(len(epochs.times))
            group.attrs["n_trials"] = int(len(epochs))
            group.attrs["source_set_relpath"] = str(set_path.relative_to(output_dir))
            group.attrs["source_eventlist_relpath"] = str(eventlist_path.relative_to(output_dir))

            for event in rt_events:
                events_rows.append(
                    {
                        "subject_id": subject_id,
                        "subject_label": subject_label(subject_id),
                        "epoch_index": event.epoch_index,
                        "stimulus_code": event.stimulus_code,
                        "response_code": event.response_code,
                        "reaction_time_ms": event.reaction_time_ms,
                        "condition": event.condition,
                        "block_target_code": event.block_target_code,
                        "trial_stimulus_code": event.trial_stimulus_code,
                        "stimulus_onset_s": event.stimulus_onset_s,
                        "response_onset_s": event.response_onset_s,
                        "sort_variable": "reaction_time_ms",
                        "source_event_item": event.source_event_item,
                        "source_set_relpath": str(set_path.relative_to(output_dir)),
                        "source_eventlist_relpath": str(eventlist_path.relative_to(output_dir)),
                    }
                )

    events_df = pd.DataFrame(events_rows)
    events_df.sort_values(["subject_id", "epoch_index"], inplace=True)
    events_df.to_csv(events_path, index=False)

    metadata = {
        "source_component": OSF_P3_COMPONENT,
        "source_root_listing": OSF_P3_ALL_DATA_ROOT,
        "source_processing_scripts": ERP_CORE_GITHUB_P3,
        "reader_docs": MNE_EEGLAB_DOCS,
        "subjects": [subject_label(i) for i in range(1, 41)],
        "hdf5_path": h5_path.name,
        "events_csv_path": events_path.name,
        "notes": [
            "Epoch source files are the processed ERP CORE P3 EEGLAB epochs "
            "('*_epoch_interp_ar.set/.fdt').",
            "Reaction times are derived from '*_P3_Eventlist_For_RTs.txt' because "
            "that file aligns 1:1 with the retained epochs.",
            "The Julia notebook performs RT sorting, per-timepoint z-scoring, "
            "Gaussian low-pass filtering, and resizing on top of these derived files.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    readme_text = f"""# ERP CORE P3 Dataset

This folder contains the processed ERP CORE P3 material used by
`notebooks/model_test/erp_core_p3.ipynb`.

## Contents

- `epochs.hdf5`: per-subject epoch tensors in `subjects/<sub>/epochs`
  with shape `(channel, time, trial)`
- `events_rt.csv`: one row per retained epoch with calculated reaction time
- `metadata.json`: source links and layout metadata
- `source/`: downloaded ERP CORE source files used to build the derived files

## Source

- OSF P3 component: {OSF_P3_COMPONENT}
- OSF root listing used for downloads: {OSF_P3_ALL_DATA_ROOT}
- ERP CORE P3 processing scripts: {ERP_CORE_GITHUB_P3}
- MNE EEGLAB epoch reader docs: {MNE_EEGLAB_DOCS}

## Notes

- Source epoch files are the processed EEGLAB files
  `*_P3_shifted_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins_epoch_interp_ar.set/.fdt`.
- Reaction times are calculated from `*_P3_Eventlist_For_RTs.txt`, which aligns
  directly with the retained epochs in the processed epoch file.
- The Julia notebook applies the remaining image pipeline steps:
  reaction-time sorting, per-timepoint z-scoring, Gaussian low-pass filtering,
  and resizing to the model input size.
"""
    readme_path.write_text(readme_text, encoding="utf-8")


def download_named_items(items: list[dict], target_dir: Path, names: list[str]) -> None:
    by_name = {item["attributes"]["name"]: item for item in items}
    for name in names:
        item = by_name.get(name)
        if item is None:
            raise FileNotFoundError(f"Missing {name} in OSF listing for {target_dir}")
        download_file(item["links"]["download"], target_dir / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("notebooks/model_test/real_data_sets/erp_core_p3"),
        help="Target dataset directory",
    )
    args = parser.parse_args()
    build_dataset(args.output_dir)


if __name__ == "__main__":
    main()
