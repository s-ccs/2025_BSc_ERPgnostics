#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import parse_qs, urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
WEEK21_DIR = REPO_ROOT / "notebooks" / "week_21"
LS_DB = REPO_ROOT / ".label-studio-data" / "label_studio.sqlite3"

EXPORT_ROOTS = [
    WEEK21_DIR / "labelstudio_export_test",
    WEEK21_DIR / "labelstudio_export_model_prioritized_200",
    WEEK21_DIR / "labelstudio_export_pattern_positive_followup_1000",
]

ANNOTATIONS_ALL_CSV = WEEK21_DIR / "labelstudio_annotations_all.csv"
ANNOTATION_SUMMARY_CSV = WEEK21_DIR / "labelstudio_annotation_summary_by_dataset.csv"
POSITIVE_SORTS_CSV = WEEK21_DIR / "labelstudio_positive_sort_variables.csv"
REFERENCE_ANNOTATIONS_CSV = WEEK21_DIR / "fixations_dataset_reference_annotations.csv"

CLASS_ID = {
    "no_class": 0,
    "sigmoid": 1,
    "one_sided_fan": 2,
    "two_sided_fan": 3,
    "diverging_bar": 4,
    "hourglass": 5,
    "tilted_bar": 6,
}
PATTERN_CLASSES = {k for k in CLASS_ID if k != "no_class"}
LABEL_ALIASES = {
    "0": "no_class",
    "1": "sigmoid",
    "2": "one_sided_fan",
    "3": "two_sided_fan",
    "4": "diverging_bar",
    "5": "hourglass",
    "6": "tilted_bar",
    "no class": "no_class",
    "no_class": "no_class",
    "sigmoid": "sigmoid",
    "one sided fan": "one_sided_fan",
    "one_sided_fan": "one_sided_fan",
    "two sided fan": "two_sided_fan",
    "two_sided_fan": "two_sided_fan",
    "diverging bar": "diverging_bar",
    "diverging_bar": "diverging_bar",
    "hourglass": "hourglass",
    "tilted bar": "tilted_bar",
    "tilted_bar": "tilted_bar",
}

TRACKING_COLUMNS = [
    "tracking_key",
    "dataset_key",
    "dataset_label",
    "channel_name",
    "channel_idx",
    "sort_variable",
    "export_batch",
    "export_root",
    "manifest_path",
    "image_file",
    "label_status",
    "label_studio_project_id",
    "label_studio_project_title",
    "label_studio_task_id",
    "annotation_id",
    "annotator_id",
    "erp_class",
    "erp_class_raw",
    "erp_class_id",
    "is_pattern_class",
    "annotation_created_at",
    "annotation_updated_at",
    "annotation_lead_time",
]


def normalize_label(value: object) -> str:
    raw = "" if value is None else str(value).strip()
    key = raw.lower().replace("-", " ").replace("_", " ")
    return LABEL_ALIASES.get(key, LABEL_ALIASES.get(raw.lower(), raw.lower().replace(" ", "_")))


def image_file_from_url(image_url: str) -> str:
    if not image_url:
        return ""
    parsed = urlparse(image_url)
    if parsed.query:
        d = parse_qs(parsed.query).get("d", [""])[0]
        if d:
            return Path(d).name
    return Path(parsed.path).name


def rel_data_path(image_url: str) -> str:
    parsed = urlparse(image_url or "")
    if parsed.query:
        return parse_qs(parsed.query).get("d", [""])[0]
    return ""


def derived_export_batch(data: dict, project_id: int) -> str:
    if data.get("export_batch"):
        return str(data["export_batch"])
    rel = rel_data_path(str(data.get("image", "")))
    if "labelstudio_export_test/" in rel:
        return "labelstudio_export_test"
    if "labelstudio_export_model_prioritized_200/" in rel:
        return "week21_model_prioritized_200"
    if "labelstudio_export_pattern_positive_followup_1000/" in rel:
        return "week21_pattern_positive_followup_1000"
    if "label_studio_data_unlabelled_additional_400/" in rel:
        return "model_test_reference_additional_400"
    if "label_studio_data_unlabelled/" in rel:
        return "model_test_reference_100"
    return f"label_studio_project_{project_id}"


def derived_export_root(data: dict) -> str:
    rel = rel_data_path(str(data.get("image", "")))
    if rel.startswith("labelstudio_export_test/"):
        return str(WEEK21_DIR / "labelstudio_export_test")
    if rel.startswith("labelstudio_export_model_prioritized_200/"):
        return str(WEEK21_DIR / "labelstudio_export_model_prioritized_200")
    if rel.startswith("labelstudio_export_pattern_positive_followup_1000/"):
        return str(WEEK21_DIR / "labelstudio_export_pattern_positive_followup_1000")
    if rel.startswith("label_studio_data_unlabelled_additional_400/"):
        return str(REPO_ROOT / "notebooks" / "model_test" / "label_studio_data_unlabelled_additional_400")
    if rel.startswith("label_studio_data_unlabelled/"):
        return str(REPO_ROOT / "notebooks" / "model_test" / "label_studio_data_unlabelled")
    return ""


def dataset_key_from_task(data: dict, project_id: int) -> str:
    if data.get("dataset_key"):
        return str(data["dataset_key"])
    if data.get("source_file") == "data_fixations.hdf5" or project_id in {14, 15}:
        return "fixations_dataset"
    return ""


def dataset_label_from_task(data: dict, dataset_key: str) -> str:
    if data.get("dataset_label"):
        return str(data["dataset_label"])
    if dataset_key == "fixations_dataset":
        return "Reference Fixation Dataset"
    return dataset_key


def channel_idx_from_task(data: dict) -> str:
    for key in ("channel_idx", "channel"):
        if data.get(key) not in (None, ""):
            return str(int(data[key]))
    return ""


def channel_name_from_task(data: dict, dataset_key: str, channel_idx: str) -> str:
    if data.get("channel_name"):
        return str(data["channel_name"])
    if dataset_key == "fixations_dataset" and channel_idx:
        return f"ch{int(channel_idx):03d}"
    return channel_idx


def first_choice(result_json: str) -> str:
    result = json.loads(result_json or "[]")
    if not result:
        return ""
    choices = result[0].get("value", {}).get("choices", [])
    return "" if not choices else str(choices[0])


def load_latest_annotations() -> list[dict[str, str]]:
    conn = sqlite3.connect(LS_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        select
            tc.id as annotation_id,
            tc.task_id,
            tc.project_id,
            p.title as project_title,
            t.data as task_data,
            tc.result as result_json,
            tc.was_cancelled,
            tc.created_at,
            tc.updated_at,
            tc.lead_time,
            tc.completed_by_id
        from task_completion tc
        join task t on t.id = tc.task_id
        join project p on p.id = tc.project_id
        where tc.was_cancelled = 0
        order by tc.task_id, tc.updated_at, tc.id
        """
    ).fetchall()
    conn.close()

    latest_by_task: dict[int, sqlite3.Row] = {}
    for row in rows:
        latest_by_task[int(row["task_id"])] = row

    out: list[dict[str, str]] = []
    for row in latest_by_task.values():
        data = json.loads(row["task_data"] or "{}")
        dataset_key = dataset_key_from_task(data, int(row["project_id"]))
        if not dataset_key:
            continue

        sort_variable = str(data.get("sort_variable") or data.get("sort_col") or "")
        channel_idx = channel_idx_from_task(data)
        channel_name = channel_name_from_task(data, dataset_key, channel_idx)
        if not sort_variable or not channel_name:
            continue

        raw_label = first_choice(row["result_json"])
        erp_class = normalize_label(raw_label)
        image_url = str(data.get("image", ""))
        image_file = str(data.get("image_file") or image_file_from_url(image_url))
        tracking_key = str(data.get("tracking_key") or f"{dataset_key}||{channel_name}||{sort_variable}")

        out.append(
            {
                "tracking_key": tracking_key,
                "dataset_key": dataset_key,
                "dataset_label": dataset_label_from_task(data, dataset_key),
                "channel_name": channel_name,
                "channel_idx": channel_idx,
                "sort_variable": sort_variable,
                "export_batch": derived_export_batch(data, int(row["project_id"])),
                "export_root": derived_export_root(data),
                "manifest_path": "",
                "image": image_url,
                "image_file": image_file,
                "label_status": "classified",
                "label_studio_project_id": str(row["project_id"]),
                "label_studio_project_title": str(row["project_title"]),
                "label_studio_task_id": str(row["task_id"]),
                "annotation_id": str(row["annotation_id"]),
                "annotator_id": "" if row["completed_by_id"] is None else str(row["completed_by_id"]),
                "erp_class": erp_class,
                "erp_class_raw": raw_label,
                "erp_class_id": str(CLASS_ID.get(erp_class, "")),
                "is_pattern_class": "true" if erp_class in PATTERN_CLASSES else "false",
                "annotation_created_at": str(row["created_at"]),
                "annotation_updated_at": str(row["updated_at"]),
                "annotation_lead_time": "" if row["lead_time"] is None else str(row["lead_time"]),
            }
        )

    return sorted(out, key=lambda r: (r["dataset_key"], r["sort_variable"], r["channel_name"], r["label_studio_task_id"]))


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        return [], []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def tracking_key_for_row(row: dict[str, str]) -> str:
    if row.get("tracking_key"):
        return row["tracking_key"]
    dataset_key = row.get("dataset_key", "")
    channel_name = row.get("channel_name") or row.get("channel") or row.get("channel_idx", "")
    sort_variable = row.get("sort_variable") or row.get("sort_col", "")
    if dataset_key == "fixations_dataset" and channel_name.isdigit():
        channel_name = f"ch{int(channel_name):03d}"
    return f"{dataset_key}||{channel_name}||{sort_variable}"


def annotation_manifest_map(annotations: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    by_key: dict[str, dict[str, str]] = {}
    for row in annotations:
        by_key[row["tracking_key"]] = row
    return by_key


def add_annotation_columns(row: dict[str, str], ann: dict[str, str] | None) -> dict[str, str]:
    out = dict(row)
    out["tracking_key"] = tracking_key_for_row(out)
    if ann is None:
        return out
    for key in TRACKING_COLUMNS:
        if key in ann and key not in {"manifest_path", "export_root", "image_file"}:
            out[key] = ann[key]
    if ann.get("image_file") and not out.get("image_file"):
        out["image_file"] = ann["image_file"]
    if ann.get("export_root") and not out.get("export_root"):
        out["export_root"] = ann["export_root"]
    out["label_status"] = "classified"
    return out


def update_tracking_csv(path: Path, annotations_by_key: dict[str, dict[str, str]]) -> None:
    fieldnames, rows = read_csv(path)
    if not rows:
        return
    updated = [add_annotation_columns(row, annotations_by_key.get(tracking_key_for_row(row))) for row in rows]
    out_fields = list(dict.fromkeys(fieldnames + TRACKING_COLUMNS))
    write_csv(path, out_fields, updated)


def manifest_to_tracking_rows(manifest_path: Path) -> list[dict[str, str]]:
    _, manifest = read_csv(manifest_path)
    rows: list[dict[str, str]] = []
    for row in manifest:
        dataset_key = row.get("dataset_key", manifest_path.parent.name)
        channel_name = row.get("channel_name") or row.get("channel") or row.get("channel_idx", "")
        if dataset_key == "fixations_dataset" and channel_name.isdigit():
            channel_name = f"ch{int(channel_name):03d}"
        tracking_key = row.get("tracking_key") or f"{dataset_key}||{channel_name}||{row.get('sort_variable', '')}"
        rows.append(
            {
                "tracking_key": tracking_key,
                "dataset_key": dataset_key,
                "dataset_label": row.get("dataset_label", dataset_key),
                "channel_name": channel_name,
                "channel_idx": row.get("channel_idx") or row.get("channel", ""),
                "sort_variable": row.get("sort_variable", ""),
                "export_batch": row.get("export_batch", manifest_path.parents[1].name),
                "export_root": str(manifest_path.parents[1]),
                "manifest_path": str(manifest_path),
                "image_file": row.get("image_file", ""),
                "label_status": "exported_for_labeling",
                "label_studio_project_id": "",
            }
        )
    return rows


def update_export_roots(annotations: list[dict[str, str]]) -> None:
    annotations_by_key = annotation_manifest_map(annotations)
    for root in EXPORT_ROOTS:
        if not root.exists():
            continue
        update_tracking_csv(root / "already_classified_tracking.csv", annotations_by_key)
        for manifest_path in sorted(root.glob("*/manifest.csv")):
            tracking_path = manifest_path.parent / "classified_combinations.csv"
            if not tracking_path.exists():
                rows = manifest_to_tracking_rows(manifest_path)
                write_csv(tracking_path, TRACKING_COLUMNS, rows)
            update_tracking_csv(tracking_path, annotations_by_key)

            dataset_key = manifest_path.parent.name
            dataset_annotations = [r for r in annotations if r["dataset_key"] == dataset_key]
            if dataset_annotations:
                write_csv(manifest_path.parent / "annotations.csv", list(dataset_annotations[0].keys()), dataset_annotations)


def write_summary_files(annotations: list[dict[str, str]]) -> None:
    write_csv(ANNOTATIONS_ALL_CSV, list(annotations[0].keys()) if annotations else TRACKING_COLUMNS, annotations)

    counts = Counter((r["dataset_key"], r["erp_class"]) for r in annotations)
    summary_rows = [
        {
            "dataset_key": dataset_key,
            "erp_class": erp_class,
            "count": str(count),
            "is_pattern_class": "true" if erp_class in PATTERN_CLASSES else "false",
        }
        for (dataset_key, erp_class), count in sorted(counts.items())
    ]
    write_csv(ANNOTATION_SUMMARY_CSV, ["dataset_key", "erp_class", "count", "is_pattern_class"], summary_rows)

    grouped: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in annotations:
        grouped[(row["dataset_key"], row["sort_variable"])][row["erp_class"]] += 1

    positive_rows: list[dict[str, str]] = []
    for (dataset_key, sort_variable), class_counts in sorted(grouped.items()):
        positive_count = sum(class_counts[c] for c in PATTERN_CLASSES)
        if positive_count <= 0:
            continue
        positive_rows.append(
            {
                "dataset_key": dataset_key,
                "sort_variable": sort_variable,
                "positive_count": str(positive_count),
                "total_labeled_count": str(sum(class_counts.values())),
                "pattern_classes": ";".join(sorted(c for c in PATTERN_CLASSES if class_counts[c] > 0)),
                "class_counts_json": json.dumps(dict(sorted(class_counts.items())), sort_keys=True),
            }
        )
    write_csv(
        POSITIVE_SORTS_CSV,
        ["dataset_key", "sort_variable", "positive_count", "total_labeled_count", "pattern_classes", "class_counts_json"],
        positive_rows,
    )

    reference_rows = [r for r in annotations if r["dataset_key"] == "fixations_dataset"]
    if reference_rows:
        write_csv(REFERENCE_ANNOTATIONS_CSV, list(reference_rows[0].keys()), reference_rows)


def main() -> None:
    if not LS_DB.exists():
        raise SystemExit(f"Label Studio DB not found: {LS_DB}")
    annotations = load_latest_annotations()
    write_summary_files(annotations)
    update_export_roots(annotations)
    print(f"Wrote {ANNOTATIONS_ALL_CSV} ({len(annotations)} annotated tasks)")
    print(f"Wrote {POSITIVE_SORTS_CSV}")
    print(f"Updated tracking CSVs under: {', '.join(str(p) for p in EXPORT_ROOTS if p.exists())}")


if __name__ == "__main__":
    main()
