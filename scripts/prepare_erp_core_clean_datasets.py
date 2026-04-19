#!/usr/bin/env python3
"""
Download already-cleaned ERP CORE components from OSF and convert them into a
compact HDF5/CSV layout under `notebooks/datasets`.

The selected source files are the processed EEGLAB epoch files ending in
`_interp_ar.set/.fdt`, which are created by the official ERP CORE artifact
rejection pipeline. No additional artifact cleaning is performed here.

Required Python packages:
    mne, h5py, numpy, pandas
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import mne
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "notebooks" / "datasets"
DEFAULT_COMPONENTS = ("p3", "n170", "lrp")
DEFAULT_SUBJECTS = (1, 2, 3, 4)

MNE_EEGLAB_DOCS = "https://mne.tools/stable/generated/mne.read_epochs_eeglab.html"
ERP_CORE_REPO_ROOT = "https://github.com/lucklab/ERP_CORE/tree/master"

EVENT_LINE_RE = re.compile(
    r'^\s*(\d+)\s+(\d+)\s+(-?\d+)\s+"([^"]*)"\s+'
    r'(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+'
    r'([01]+)\s+([01]+)\s+(\d+)\s+\[(.*?)\]\s*$'
)


@dataclass(frozen=True)
class ComponentConfig:
    key: str
    title: str
    node_id: str
    component_url: str
    root_listing_url: str
    github_dir: str
    epoch_set_template: str
    epoch_fdt_template: str
    eventlist_template: str
    recommended_sort_columns: tuple[str, ...]
    preferred_channels: tuple[str, ...]
    example_source_relpaths: tuple[str, ...]
    artifact_script_name: str = "Script6_Artifact_Rejection.m"
    requires_response: bool = True
    root_named_files: tuple[str, ...] = ()
    behavior_named_files: tuple[str, ...] = ()
    known_missing_subject_ids: tuple[int, ...] = ()

    @property
    def output_dirname(self) -> str:
        return f"erp_core_{self.key}_clean"

    @property
    def github_component_url(self) -> str:
        return f"{ERP_CORE_REPO_ROOT}/{self.github_dir}"


COMPONENTS: dict[str, ComponentConfig] = {
    "p3": ComponentConfig(
        key="p3",
        title="P3",
        node_id="etdkz",
        component_url="https://osf.io/etdkz/",
        root_listing_url="https://files.osf.io/v1/resources/etdkz/providers/osfstorage/5f247351b084f60115c9aa10/",
        github_dir="P3",
        epoch_set_template="{sid}_P3_shifted_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins_epoch_interp_ar.set",
        epoch_fdt_template="{sid}_P3_shifted_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins_epoch_interp_ar.fdt",
        eventlist_template="{sid}_P3_Eventlist_For_RTs.txt",
        recommended_sort_columns=("reaction_time_ms", "condition", "trial_stimulus_code", "epoch_index"),
        preferred_channels=("Pz", "CPz", "POz", "Cz"),
        root_named_files=("README_P3.txt", "P3_Subject_Summary.xlsx", "Participant_Demographics.xlsx"),
        behavior_named_files=("BDF_P3_RTs.txt", "Script14_Calculate_RTs_and_Accuracy.m"),
        example_source_relpaths=(
            "source/P3_All_Data_and_Scripts/Behavior_Measurements/Script14_Calculate_RTs_and_Accuracy.m",
            "source/P3_All_Data_and_Scripts/EEG_ERP_Processing/Script6_Artifact_Rejection.m",
            "source/P3_All_Data_and_Scripts/Behavior_Measurements/BDF_P3_RTs.txt",
        ),
    ),
    "n170": ComponentConfig(
        key="n170",
        title="N170",
        node_id="pfde9",
        component_url="https://osf.io/pfde9/",
        root_listing_url="https://files.osf.io/v1/resources/pfde9/providers/osfstorage/5f2479095f705a010e619b0a/",
        github_dir="N170",
        epoch_set_template="{sid}_N170_shifted_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins_epoch_interp_ar.set",
        epoch_fdt_template="{sid}_N170_shifted_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins_epoch_interp_ar.fdt",
        eventlist_template="{sid}_N170_Eventlist_For_RTs.txt",
        recommended_sort_columns=("reaction_time_ms", "condition", "stimulus_code", "epoch_index"),
        preferred_channels=("PO8", "P8", "PO7", "P7"),
        root_named_files=("N170_Subject_Summary.xlsx", "License.txt"),
        behavior_named_files=("BDF_N170_RTs.txt", "Script14_Calculate_RTs_and_Accuracy.m"),
        example_source_relpaths=(
            "source/N170_All_Data_and_Scripts/Behavior_Measurements/Script14_Calculate_RTs_and_Accuracy.m",
            "source/N170_All_Data_and_Scripts/EEG_ERP_Processing/Script6_Artifact_Rejection.m",
            "source/N170_All_Data_and_Scripts/Behavior_Measurements/BDF_N170_RTs.txt",
        ),
    ),
    "n400": ComponentConfig(
        key="n400",
        title="N400",
        node_id="29xpq",
        component_url="https://osf.io/29xpq/",
        root_listing_url="https://api.osf.io/v2/nodes/29xpq/files/osfstorage/5f248dcb9c909400fd48877d/",
        github_dir="N400",
        epoch_set_template="{sid}_N400_shifted_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins_epoch_interp_ar.set",
        epoch_fdt_template="{sid}_N400_shifted_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins_epoch_interp_ar.fdt",
        eventlist_template="{sid}_N400_Eventlist_For_RTs.txt",
        recommended_sort_columns=("reaction_time_ms", "condition", "stimulus_code", "epoch_index"),
        preferred_channels=("CPz", "Pz", "Cz", "POz"),
        example_source_relpaths=(
            "source/N400_All_Data_and_Scripts/Behavior_Measurements/Script14_Calculate_RTs_and_Accuracy.m",
            "source/N400_All_Data_and_Scripts/EEG_ERP_Processing/Script6_Artifact_Rejection.m",
            "source/N400_All_Data_and_Scripts/Behavior_Measurements/BDF_N400_RTs.txt",
        ),
        artifact_script_name="Script6_Artifact_Rejection.m",
        root_named_files=(
            "README_N400.txt",
            "N400_Subject_Summary.xlsx",
            "Participant_Demographics.xlsx",
            "License.txt",
            "N400_Event_Code_Scheme.xlsx",
        ),
        behavior_named_files=("BDF_N400_RTs.txt", "Script14_Calculate_RTs_and_Accuracy.m"),
    ),
    "n2pc": ComponentConfig(
        key="n2pc",
        title="N2pc",
        node_id="yefrq",
        component_url="https://osf.io/yefrq/",
        root_listing_url="https://api.osf.io/v2/nodes/yefrq/files/osfstorage/612d2ecaaf610c0028e0010b/",
        github_dir="N2pc",
        epoch_set_template="{sid}_N2pc_shifted_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins_epoch_interp_ar.set",
        epoch_fdt_template="{sid}_N2pc_shifted_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins_epoch_interp_ar.fdt",
        eventlist_template="{sid}_N2pc_Eventlist_For_RTs.txt",
        recommended_sort_columns=("reaction_time_ms", "condition", "stimulus_code", "epoch_index"),
        preferred_channels=("PO7", "PO8", "P7", "P8"),
        example_source_relpaths=(
            "source/N2pc_All_Data_and_Scripts/Behavior_Measurements/Script14_Calculate_RTs_and_Accuracy.m",
            "source/N2pc_All_Data_and_Scripts/EEG_ERP_Processing/Script6_Artifact_Rejection.m",
            "source/N2pc_All_Data_and_Scripts/Behavior_Measurements/BDF_N2pc_RTs.txt",
        ),
        artifact_script_name="Script6_Artifact_Rejection.m",
        root_named_files=(
            "README_N2pc.txt",
            "N2pc_Subject_Summary.xlsx",
            "License.txt",
            "N2pc_Event_Code_Scheme.xlsx",
            "N2pc Analysis Procedures.pdf",
        ),
        behavior_named_files=("BDF_N2pc_RTs.txt", "Script14_Calculate_RTs_and_Accuracy.m"),
    ),
    "mmn": ComponentConfig(
        key="mmn",
        title="MMN",
        node_id="5q4xs",
        component_url="https://osf.io/5q4xs/",
        root_listing_url="https://api.osf.io/v2/nodes/5q4xs/files/osfstorage/5f248e8db084f6011bc9da61/",
        github_dir="MMN",
        epoch_set_template="{sid}_MMN_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins_epoch_interp_ar.set",
        epoch_fdt_template="{sid}_MMN_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins_epoch_interp_ar.fdt",
        eventlist_template="{sid}_MMN_Eventlist_Bins.txt",
        recommended_sort_columns=("condition", "stimulus_code", "epoch_index", "bin_id"),
        preferred_channels=("FCz", "Fz", "Cz", "CPz"),
        example_source_relpaths=(
            "source/MMN_All_Data_and_Scripts/EEG_ERP_Processing/Script6_Artifact_Rejection.m",
        ),
        artifact_script_name="Script6_Artifact_Rejection.m",
        requires_response=False,
        root_named_files=(
            "README_MMN.txt",
            "MMN_Subject_Summary.xlsx",
            "Participant_Demographics.xlsx",
            "License.txt",
            "MMN_Event_Code_Scheme.xlsx",
        ),
    ),
    "lrp": ComponentConfig(
        key="lrp",
        title="LRP",
        node_id="28e6c",
        component_url="https://osf.io/28e6c/",
        root_listing_url="https://api.osf.io/v2/nodes/28e6c/files/osfstorage/5f248f3bb084f60123c9ab4d/",
        github_dir="LRP",
        epoch_set_template="{sid}_LRP_shifted_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins_epoch_interp_ar.set",
        epoch_fdt_template="{sid}_LRP_shifted_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins_epoch_interp_ar.fdt",
        eventlist_template="{sid}_LRP_Eventlist_For_RTs.txt",
        recommended_sort_columns=("reaction_time_ms", "condition", "stimulus_code", "epoch_index"),
        preferred_channels=("Cz", "CPz", "C3", "C4", "FCz"),
        example_source_relpaths=(
            "source/LRP_All_Data_and_Scripts/Behavior_Measurements/Script14_Calculate_RTs_and_Accuracy.m",
            "source/LRP_All_Data_and_Scripts/EEG_ERP_Processing/Script6_Artifact_Rejection.m",
            "source/LRP_All_Data_and_Scripts/Behavior_Measurements/BDF_LRP_RTs.txt",
        ),
        artifact_script_name="Script6_Artifact_Rejection.m",
        root_named_files=(
            "README_LRP.txt",
            "LRP_Subject_Summary.xlsx",
            "Participant_Demographics.xlsx",
            "License.txt",
            "LRP_Event_Code_Scheme.xlsx",
        ),
        behavior_named_files=("BDF_LRP_RTs.txt", "Script14_Calculate_RTs_and_Accuracy.m"),
        known_missing_subject_ids=(2,),
    ),
    "ern": ComponentConfig(
        key="ern",
        title="ERN",
        node_id="q6gwp",
        component_url="https://osf.io/q6gwp/",
        root_listing_url="https://api.osf.io/v2/nodes/q6gwp/files/osfstorage/5f246c9fb084f6011bc9af2d/",
        github_dir="ERN",
        epoch_set_template="{sid}_ERN_shifted_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins_epoch_interp_ar.set",
        epoch_fdt_template="{sid}_ERN_shifted_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins_epoch_interp_ar.fdt",
        eventlist_template="{sid}_ERN_Eventlist_For_RTs.txt",
        recommended_sort_columns=("reaction_time_ms", "condition", "stimulus_code", "epoch_index"),
        preferred_channels=("FCz", "Cz", "Fz", "CPz"),
        example_source_relpaths=(
            "source/ERN_All_Data_and_Scripts/Behavior_Measurements/Script14_Calculate_RTs_and_Accuracy.m",
            "source/ERN_All_Data_and_Scripts/EEG_ERP_Processing/Script6_Artifact_Rejection.m",
            "source/ERN_All_Data_and_Scripts/Behavior_Measurements/BDF_ERN_RTs.txt",
        ),
        artifact_script_name="Script6_Artifact_Rejection.m",
        root_named_files=(
            "README_ERN.txt",
            "ERN_Subject_Summary.xlsx",
            "Participant_Demographics.xlsx",
            "License.txt",
            "ERN_Event_Code_Scheme.xlsx",
        ),
        behavior_named_files=("BDF_ERN_RTs.txt", "Script14_Calculate_RTs_and_Accuracy.m"),
    ),
}


def fetch_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url) as response:
        payload = json.load(response)
    if not isinstance(payload, dict) or "data" not in payload or "links" not in payload:
        return payload

    data = list(payload.get("data", []))
    next_url = payload.get("links", {}).get("next")
    while next_url:
        with urllib.request.urlopen(next_url) as response:
            next_payload = json.load(response)
        data.extend(next_payload.get("data", []))
        next_url = next_payload.get("links", {}).get("next")

    payload["data"] = data
    return payload


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


def download_named_items(items: list[dict[str, Any]], target_dir: Path, names: tuple[str, ...]) -> None:
    if not names:
        return
    by_name = {item["attributes"]["name"]: item for item in items}
    for name in names:
        item = by_name.get(name)
        if item is None:
            raise FileNotFoundError(f"Missing {name} in OSF listing for {target_dir}")
        download_file(item["links"]["download"], target_dir / name)


def subject_label(subject_id: int) -> str:
    return f"sub-{subject_id:03d}"


def parse_bin_labels(path: Path) -> dict[int, str]:
    labels: dict[int, str] = {}
    current_bin: int | None = None
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        inline_match = re.match(r"^bin\s+(\d+)\s*,\s*#\s*[^,]*,\s*(.+?)\s*$", line, re.IGNORECASE)
        if inline_match:
            labels[int(inline_match.group(1))] = re.sub(r"\s+", " ", inline_match.group(2)).strip()
            current_bin = None
            continue
        match = re.match(r"^bin\s+(\d+)$", line, re.IGNORECASE)
        if match:
            current_bin = int(match.group(1))
            continue
        if current_bin is not None and current_bin not in labels:
            labels[current_bin] = re.sub(r"\s+", " ", line)
            current_bin = None
    return labels


def split_bin_label(label: str) -> tuple[str, str]:
    cleaned = re.sub(r"\s+(after|before)\s+AR$", "", label, flags=re.IGNORECASE).strip()
    parts = [part.strip() for part in cleaned.split(",") if part.strip()]
    condition = parts[0] if parts else cleaned
    accuracy = parts[1] if len(parts) > 1 else ""
    return condition, accuracy


def p3_extras(stimulus_code: int) -> dict[str, Any]:
    block_target_code = stimulus_code // 10
    trial_stimulus_code = stimulus_code % 10
    return {
        "block_target_code": block_target_code,
        "trial_stimulus_code": trial_stimulus_code,
        "is_target_match": bool(block_target_code == trial_stimulus_code),
    }


def n170_extras(stimulus_code: int) -> dict[str, Any]:
    if 1 <= stimulus_code <= 40:
        return {"stimulus_family": "Faces", "stimulus_exemplar_index": stimulus_code}
    if 41 <= stimulus_code <= 80:
        return {"stimulus_family": "Cars", "stimulus_exemplar_index": stimulus_code - 40}
    if 101 <= stimulus_code <= 140:
        return {"stimulus_family": "Scrambled Faces", "stimulus_exemplar_index": stimulus_code - 100}
    if 141 <= stimulus_code <= 180:
        return {"stimulus_family": "Scrambled Cars", "stimulus_exemplar_index": stimulus_code - 140}
    return {"stimulus_family": "Unknown", "stimulus_exemplar_index": None}


def component_specific_extras(config: ComponentConfig, stimulus_code: int) -> dict[str, Any]:
    if config.key == "p3":
        return p3_extras(stimulus_code)
    if config.key == "n170":
        return n170_extras(stimulus_code)
    return {}


def read_eventlist_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
    return rows


def parse_eventlist_for_rt(path: Path, bin_labels: dict[int, str], config: ComponentConfig) -> list[dict[str, Any]]:
    rows = read_eventlist_rows(path)
    stimuli: list[dict[str, Any]] = []
    for idx, row in enumerate(rows[:-1]):
        if row["bepoch"] <= 0:
            continue
        stimulus_code = int(row["ecode"])
        if stimulus_code < 1:
            continue

        candidate_bins = [
            b for b in row["bins"] if "after ar" in bin_labels.get(b, "").lower()
        ]
        if not candidate_bins:
            continue

        response = rows[idx + 1]
        response_code = int(response["ecode"])
        if response_code not in {201, 202}:
            raise RuntimeError(
                f"Expected response after stimulus item {row['item']} in {path}, "
                f"got event code {response_code}"
            )

        bin_id = int(candidate_bins[0])
        bin_label = bin_labels.get(bin_id, f"bin {bin_id}")
        condition_label, accuracy_label = split_bin_label(bin_label)
        accuracy_norm = accuracy_label.lower() if accuracy_label else (
            "correct" if response_code == 201 else "incorrect"
        )

        event_row = {
            "epoch_index": int(row["bepoch"]),
            "bin_id": bin_id,
            "bin_label": bin_label,
            "stimulus_code": stimulus_code,
            "response_code": response_code,
            "reaction_time_ms": float(response["diff_ms"]),
            "condition": condition_label,
            "accuracy": accuracy_norm,
            "stimulus_onset_s": float(row["onset_s"]),
            "response_onset_s": float(response["onset_s"]),
            "source_event_item": int(row["item"]),
        }
        event_row.update(component_specific_extras(config, stimulus_code))
        stimuli.append(event_row)

    stimuli.sort(key=lambda event: int(event["epoch_index"]))
    return stimuli


def parse_lrp_eventlist_for_rt(
    path: Path,
    bin_labels: dict[int, str],
    config: ComponentConfig,
) -> list[dict[str, Any]]:
    rows = read_eventlist_rows(path)
    events: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if row["bepoch"] <= 0 or not row["bins"]:
            continue
        if idx == 0:
            raise RuntimeError(f"LRP response item {row['item']} in {path} has no preceding stimulus row")
        stimulus = rows[idx - 1]
        if stimulus["bepoch"] != 0:
            raise RuntimeError(
                f"LRP response item {row['item']} in {path} is not preceded by a continuous stimulus event"
            )

        stimulus_code = int(stimulus["ecode"])
        response_code = int(row["ecode"])
        if stimulus_code not in {11, 12, 21, 22}:
            raise RuntimeError(
                f"Unexpected LRP stimulus code {stimulus_code} before response item {row['item']} in {path}"
            )
        if response_code not in {111, 121, 212, 222}:
            raise RuntimeError(
                f"Unexpected LRP response code {response_code} at response item {row['item']} in {path}"
            )

        candidate_bins = [b for b in row["bins"] if b in bin_labels]
        if not candidate_bins:
            continue
        # Bins 1-2 are all-trial response side bins; bins 3-6 preserve compatibility.
        bin_id = int(next((b for b in candidate_bins if b not in {1, 2}), candidate_bins[0]))
        bin_label = bin_labels.get(bin_id, f"bin {bin_id}")
        condition_label, accuracy_label = split_bin_label(bin_label)
        condition = condition_label or bin_label
        response_side = "left" if response_code in {111, 121} else "right"
        flanker_compatibility = (
            "compatible" if stimulus_code in {11, 12} else "incompatible"
        )

        event_row = {
            "epoch_index": int(row["bepoch"]),
            "bin_id": bin_id,
            "bin_label": bin_label,
            "stimulus_code": stimulus_code,
            "response_code": response_code,
            "reaction_time_ms": float(row["diff_ms"]),
            "condition": condition,
            "accuracy": accuracy_label.lower() if accuracy_label else "correct",
            "stimulus_onset_s": float(stimulus["onset_s"]),
            "response_onset_s": float(row["onset_s"]),
            "source_event_item": int(row["item"]),
            "response_side": response_side,
            "flanker_compatibility": flanker_compatibility,
        }
        events.append(event_row)

    events.sort(key=lambda event: int(event["epoch_index"]))
    return events


def parse_eventlist_without_response(
    path: Path,
    bin_labels: dict[int, str],
    config: ComponentConfig,
) -> list[dict[str, Any]]:
    rows = read_eventlist_rows(path)
    events: list[dict[str, Any]] = []
    for row in rows:
        stimulus_code = int(row["ecode"])
        if stimulus_code < 1:
            continue
        candidate_bins = [b for b in row["bins"] if b in bin_labels]
        if not candidate_bins:
            continue

        bin_id = int(candidate_bins[0])
        bin_label = bin_labels.get(bin_id, f"bin {bin_id}")
        condition_label, _accuracy_label = split_bin_label(bin_label)
        epoch_index = int(row["bepoch"]) if row["bepoch"] > 0 else len(events) + 1

        event_row = {
            "epoch_index": epoch_index,
            "bin_id": bin_id,
            "bin_label": bin_label,
            "stimulus_code": stimulus_code,
            "response_code": 0,
            "reaction_time_ms": float("nan"),
            "condition": condition_label,
            "accuracy": "n/a",
            "stimulus_onset_s": float(row["onset_s"]),
            "response_onset_s": float("nan"),
            "source_event_item": int(row["item"]),
        }
        event_row.update(component_specific_extras(config, stimulus_code))
        events.append(event_row)

    events.sort(key=lambda event: int(event["epoch_index"]))
    return events


def parse_epochs_without_response(epochs: mne.BaseEpochs, config: ComponentConfig) -> list[dict[str, Any]]:
    id_to_name = {int(code): name for name, code in epochs.event_id.items()}
    sfreq = float(epochs.info["sfreq"])
    events: list[dict[str, Any]] = []
    for epoch_index, event in enumerate(epochs.events, start=1):
        sample_idx = int(event[0])
        event_code = int(event[2])
        event_name = id_to_name.get(event_code, f"event_{event_code}")
        event_row = {
            "epoch_index": epoch_index,
            "bin_id": event_code,
            "bin_label": event_name,
            "stimulus_code": event_code,
            "response_code": 0,
            "reaction_time_ms": float("nan"),
            "condition": event_name,
            "accuracy": "n/a",
            "stimulus_onset_s": sample_idx / sfreq,
            "response_onset_s": float("nan"),
            "source_event_item": epoch_index,
        }
        event_row.update(component_specific_extras(config, event_code))
        events.append(event_row)
    return events


def read_source_snippets(output_dir: Path, relpaths: tuple[str, ...]) -> dict[str, str]:
    snippets: dict[str, str] = {}
    for relpath in relpaths:
        path = output_dir / relpath
        if not path.is_file():
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        keep: list[str] = []
        for idx, line in enumerate(lines, start=1):
            if (
                "pop_loadset" in line
                or "pop_exporteegeventlist" in line
                or "interp_ar" in line
                or "artifact rejection" in line.lower()
            ):
                keep.append(f"{idx}: {line.rstrip()}")
        snippets[relpath] = "\n".join(keep[:12])
    return snippets


def download_component_support_files(
    config: ComponentConfig,
    root_listing: dict[str, Any],
    source_root: Path,
) -> None:
    download_named_items(root_listing["data"], source_root, config.root_named_files)

    by_name = {item["attributes"]["name"]: item for item in root_listing["data"]}
    for folder_name, file_names in {
        "Behavior_Measurements": config.behavior_named_files,
        "EEG_ERP_Processing": (config.artifact_script_name,),
    }.items():
        if not file_names:
            continue
        folder_item = by_name.get(folder_name)
        if folder_item is None:
            raise FileNotFoundError(f"Missing {folder_name} in root listing for {config.title}")
        folder_listing = fetch_json(folder_item["links"]["move"])
        download_named_items(folder_listing["data"], source_root / folder_name, file_names)


def build_component_dataset(config: ComponentConfig, output_root: Path, subject_ids: list[int]) -> Path:
    output_dir = output_root / config.output_dirname
    output_dir.mkdir(parents=True, exist_ok=True)

    source_root = output_dir / "source" / f"{config.title}_All_Data_and_Scripts"
    root_listing = fetch_json(config.root_listing_url)
    download_component_support_files(config, root_listing, source_root)
    subject_items = {
        int(item["attributes"]["name"]): item
        for item in root_listing["data"]
        if item["attributes"]["kind"] == "folder" and item["attributes"]["name"].isdigit()
    }

    events_rows: list[dict[str, Any]] = []
    subject_trial_counts: list[dict[str, Any]] = []
    processed_subject_ids: list[int] = []

    h5_path = output_dir / "epochs.hdf5"
    events_path = output_dir / "events.csv"
    metadata_path = output_dir / "metadata.json"
    readme_path = output_dir / "README.md"

    with h5py.File(h5_path, "w") as h5:
        h5.attrs["dataset_key"] = config.output_dirname
        h5.attrs["component"] = config.title
        h5.attrs["source_component"] = config.component_url
        h5.attrs["source_root_listing"] = config.root_listing_url
        h5.attrs["layout"] = "subjects/<sub>/epochs (channels, time, trial)"
        h5.attrs["reader"] = "mne.read_epochs_eeglab"
        subjects_group = h5.create_group("subjects")

        for subject_id in subject_ids:
            subject_item = subject_items.get(subject_id)
            if subject_item is None:
                if subject_id in config.known_missing_subject_ids:
                    print(
                        f"  [warn] {config.title} subject {subject_id} is absent from the OSF listing; skipping"
                    )
                    continue
                raise FileNotFoundError(f"Subject {subject_id} missing in {config.title} listing")

            subject_source_dir = source_root / "subjects" / subject_label(subject_id)
            subject_listing = fetch_json(subject_item["links"]["move"])
            required_names = (
                config.epoch_set_template.format(sid=subject_id),
                config.epoch_fdt_template.format(sid=subject_id),
                config.eventlist_template.format(sid=subject_id),
            )
            download_named_items(subject_listing["data"], subject_source_dir, required_names)

            set_path = subject_source_dir / required_names[0]
            eventlist_path = subject_source_dir / required_names[2]
            epochs = mne.read_epochs_eeglab(set_path, verbose="ERROR")
            data = epochs.get_data(copy=True).astype(np.float32)  # (trial, channel, time)
            data = np.transpose(data, (0, 2, 1))  # Julia reads back as (channel, time, trial)
            ch_names = np.asarray(epochs.ch_names, dtype=h5py.string_dtype(encoding="utf-8"))
            times_s = np.asarray(epochs.times, dtype=np.float32)

            if config.behavior_named_files:
                bin_labels = parse_bin_labels(
                    source_root / "Behavior_Measurements" / config.behavior_named_files[0]
                )
            else:
                bin_labels = parse_bin_labels(eventlist_path)

            if config.key == "lrp":
                event_rows = parse_lrp_eventlist_for_rt(eventlist_path, bin_labels, config)
            elif config.requires_response:
                event_rows = parse_eventlist_for_rt(eventlist_path, bin_labels, config)
            else:
                event_rows = parse_epochs_without_response(epochs, config)
            if len(event_rows) != len(epochs):
                raise RuntimeError(
                    f"{config.title} subject {subject_id}: event list count {len(event_rows)} "
                    f"does not match epochs {len(epochs)}"
                )

            subject_key = subject_label(subject_id)
            group = subjects_group.create_group(subject_key)
            group.create_dataset("epochs", data=data, compression="gzip", compression_opts=4)
            group.create_dataset("times_s", data=times_s)
            group.create_dataset("channel_names", data=ch_names)
            group.attrs["subject_id"] = subject_id
            group.attrs["subject_label"] = subject_key
            group.attrs["sfreq_hz"] = float(epochs.info["sfreq"])
            group.attrs["n_channels"] = int(len(epochs.ch_names))
            group.attrs["n_timepoints"] = int(len(epochs.times))
            group.attrs["n_trials"] = int(len(epochs))
            group.attrs["source_set_relpath"] = str(set_path.relative_to(output_dir))
            group.attrs["source_eventlist_relpath"] = str(eventlist_path.relative_to(output_dir))

            subject_trial_counts.append(
                {"subject_label": subject_key, "n_trials": int(len(epochs))}
            )
            processed_subject_ids.append(subject_id)

            for event in event_rows:
                row = {
                    "dataset_key": config.output_dirname,
                    "component": config.title,
                    "subject_id": subject_id,
                    "subject_label": subject_key,
                    "epoch_index": int(event["epoch_index"]),
                    "bin_id": int(event["bin_id"]),
                    "bin_label": str(event["bin_label"]),
                    "stimulus_code": int(event["stimulus_code"]),
                    "response_code": int(event["response_code"]),
                    "reaction_time_ms": float(event["reaction_time_ms"]),
                    "condition": str(event["condition"]),
                    "accuracy": str(event["accuracy"]),
                    "stimulus_onset_s": float(event["stimulus_onset_s"]),
                    "response_onset_s": float(event["response_onset_s"]),
                    "source_event_item": int(event["source_event_item"]),
                    "source_set_relpath": str(set_path.relative_to(output_dir)),
                    "source_eventlist_relpath": str(eventlist_path.relative_to(output_dir)),
                }
                for key, value in event.items():
                    if key in row:
                        continue
                    row[key] = value
                events_rows.append(row)

    events_df = pd.DataFrame(events_rows)
    events_df.sort_values(["subject_id", "epoch_index"], inplace=True)
    events_df.to_csv(events_path, index=False)

    metadata = {
        "dataset_key": config.output_dirname,
        "component": config.title,
        "source_component": config.component_url,
        "source_root_listing": config.root_listing_url,
        "source_processing_scripts": config.github_component_url,
        "reader_docs": MNE_EEGLAB_DOCS,
        "selected_subjects": [subject_label(subject_id) for subject_id in processed_subject_ids],
        "preferred_channels": list(config.preferred_channels),
        "recommended_sort_columns": list(config.recommended_sort_columns),
        "hdf5_path": h5_path.name,
        "events_csv_path": events_path.name,
        "notes": [
            "Source epoch files are the official ERP CORE processed EEGLAB epoch files ending in '_interp_ar.set/.fdt'.",
            "Those files already passed the component-specific ERP CORE ICA + interpolation + artifact rejection pipeline.",
            "This conversion only loads the cleaned epochs, extracts the matching event list metadata, and stores them in a Julia-friendly layout.",
        ],
        "official_source_examples": read_source_snippets(output_dir, config.example_source_relpaths),
        "subject_trial_counts": subject_trial_counts,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    readme_text = f"""# {config.title} ERP CORE Clean Dataset

This folder contains a compact derivative of the official, already-cleaned ERP CORE
`{config.title}` component for the Week 15 comparison notebook.

## Contents

- `epochs.hdf5`: per-subject epoch tensors in `subjects/<sub>/epochs` with shape `(channel, time, trial)`
- `events.csv`: one row per retained epoch with RT metadata and sort columns
- `metadata.json`: source links, example loader snippets, preferred channels, and selected subjects
- `source/`: the official RT/event scripts plus the downloaded cleaned EEGLAB files for the selected subjects

## Official sources

- OSF component: {config.component_url}
- OSF root listing: {config.root_listing_url}
- ERP CORE scripts: {config.github_component_url}
- MNE EEGLAB reader docs: {MNE_EEGLAB_DOCS}

## Cleaning status

- Only the official processed files `*_interp_ar.set/.fdt` are used here.
- No additional artifact cleaning is performed by this repo for these datasets.
"""
    readme_path.write_text(readme_text, encoding="utf-8")
    return output_dir


def parse_subjects(values: list[str] | None) -> list[int]:
    if not values:
        return list(DEFAULT_SUBJECTS)
    out: list[int] = []
    for value in values:
        if ":" in value:
            start_s, stop_s = value.split(":", 1)
            start = int(start_s)
            stop = int(stop_s)
            out.extend(range(start, stop + 1))
        else:
            out.append(int(value))
    seen: set[int] = set()
    deduped: list[int] = []
    for subject_id in out:
        if subject_id in seen:
            continue
        seen.add(subject_id)
        deduped.append(subject_id)
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory where dataset folders will be created",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        default=list(DEFAULT_COMPONENTS),
        choices=sorted(COMPONENTS.keys()),
        help="ERP CORE components to prepare",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=[str(subject_id) for subject_id in DEFAULT_SUBJECTS],
        help="Subject ids to download, e.g. '1 2 3 4' or ranges like '1:4'",
    )
    args = parser.parse_args()

    subject_ids = parse_subjects(args.subjects)
    for component_key in args.components:
        config = COMPONENTS[component_key]
        build_component_dataset(config, args.output_root, subject_ids)


if __name__ == "__main__":
    main()
