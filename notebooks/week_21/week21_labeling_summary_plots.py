#!/usr/bin/env python3
"""Build Week 21 Label Studio summary tables and plots.

The script intentionally uses only the Python standard library plus matplotlib,
because the project environment does not currently provide pandas.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PATTERN_CLASSES = [
    "sigmoid",
    "one_sided_fan",
    "two_sided_fan",
    "diverging_bar",
    "hourglass",
    "tilted_bar",
]
CLASS_COLORS = {
    "no_class": "#7a869a",
    "sigmoid": "#2166ac",
    "one_sided_fan": "#67a9cf",
    "two_sided_fan": "#1b9e77",
    "diverging_bar": "#d73027",
    "hourglass": "#fdae61",
    "tilted_bar": "#984ea3",
}
EXCLUDED_TRAINING_DATASETS = {"02_new_eeget_rsod"}


def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path(__file__)).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "notebooks").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repo root from {start}")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def short_label(dataset_key: str, max_len: int = 34) -> str:
    if len(dataset_key) <= max_len:
        return dataset_key
    return dataset_key[: max_len - 1] + "..."


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def load_source_references(week21: Path) -> dict[str, dict[str, object]]:
    refs: dict[str, dict[str, object]] = {}
    for path in sorted(week21.glob("labelstudio_export_*/*/source_reference.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        dataset_key = str(data.get("dataset_key") or path.parent.name)
        current = refs.setdefault(dataset_key, {})
        current.update(data)
        current.setdefault("reference_files", []).append(str(path))
    return refs


def summarize_by_dataset(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["dataset_key"]].append(row)

    out: list[dict[str, object]] = []
    for dataset_key, items in grouped.items():
        total = len(items)
        pattern = sum(truthy(row["is_pattern_class"]) for row in items)
        no_class = total - pattern
        class_counts = Counter(row["erp_class"] for row in items)
        out.append(
            {
                "dataset_key": dataset_key,
                "dataset_label": items[0].get("dataset_label", ""),
                "total_labeled": total,
                "pattern_labeled": pattern,
                "no_class_labeled": no_class,
                "positive_rate": round(ratio(pattern, total), 4),
                "n_channels": len({row["channel_name"] for row in items}),
                "n_sort_variables": len({row["sort_variable"] for row in items}),
                "n_export_batches": len({row["export_batch"] for row in items}),
                "export_batches": ";".join(sorted({row["export_batch"] for row in items})),
                "pattern_classes": ";".join(cls for cls in PATTERN_CLASSES if class_counts[cls] > 0),
                "excluded_from_training": str(dataset_key in EXCLUDED_TRAINING_DATASETS).lower(),
            }
        )
    out.sort(key=lambda row: (-int(row["total_labeled"]), str(row["dataset_key"])))
    return out


def summarize_by_export_batch(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["export_batch"]].append(row)

    out: list[dict[str, object]] = []
    for export_batch, items in grouped.items():
        total = len(items)
        pattern = sum(truthy(row["is_pattern_class"]) for row in items)
        out.append(
            {
                "export_batch": export_batch,
                "total_labeled": total,
                "pattern_labeled": pattern,
                "no_class_labeled": total - pattern,
                "positive_rate": round(ratio(pattern, total), 4),
                "n_datasets": len({row["dataset_key"] for row in items}),
            }
        )
    out.sort(key=lambda row: (-int(row["total_labeled"]), str(row["export_batch"])))
    return out


def summarize_dataset_classes(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    counts = Counter((row["dataset_key"], row["dataset_label"], row["erp_class"]) for row in rows)
    out = [
        {
            "dataset_key": dataset_key,
            "dataset_label": dataset_label,
            "erp_class": erp_class,
            "count": count,
            "is_pattern_class": str(erp_class != "no_class").lower(),
        }
        for (dataset_key, dataset_label, erp_class), count in counts.items()
    ]
    out.sort(key=lambda row: (str(row["dataset_key"]), str(row["erp_class"])))
    return out


def summarize_sort_variables(rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row["dataset_key"], row.get("dataset_label", ""), row["sort_variable"])
        grouped[key].append(row)

    all_rows: list[dict[str, object]] = []
    positive_rows: list[dict[str, object]] = []
    for (dataset_key, dataset_label, sort_variable), items in grouped.items():
        total = len(items)
        positive_items = [row for row in items if truthy(row["is_pattern_class"])]
        class_counts = Counter(row["erp_class"] for row in items)
        positive_class_counts = {cls: class_counts[cls] for cls in PATTERN_CLASSES if class_counts[cls] > 0}
        base = {
            "dataset_key": dataset_key,
            "dataset_label": dataset_label,
            "sort_variable": sort_variable,
            "export_batches": ";".join(sorted({row["export_batch"] for row in items})),
            "total_labeled": total,
            "pattern_labeled": len(positive_items),
            "no_class_labeled": class_counts["no_class"],
            "positive_rate": round(ratio(len(positive_items), total), 4),
            "n_channels_total": len({row["channel_name"] for row in items}),
            "n_channels_with_positive": len({row["channel_name"] for row in positive_items}),
            "channels_with_positive": ";".join(sorted({row["channel_name"] for row in positive_items})),
            "pattern_classes": ";".join(cls for cls in PATTERN_CLASSES if class_counts[cls] > 0),
            "class_counts_json": json.dumps(dict(sorted(class_counts.items())), sort_keys=True),
            "positive_class_counts_json": json.dumps(positive_class_counts, sort_keys=True),
        }
        all_rows.append(base)
        if positive_items:
            positive_rows.append(base)

    all_rows.sort(key=lambda row: (str(row["dataset_key"]), str(row["sort_variable"])))
    positive_rows.sort(key=lambda row: (-int(row["pattern_labeled"]), str(row["dataset_key"]), str(row["sort_variable"])))
    return all_rows, positive_rows


def positive_instances(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out = []
    for row in rows:
        if truthy(row["is_pattern_class"]):
            out.append(
                {
                    "dataset_key": row["dataset_key"],
                    "dataset_label": row.get("dataset_label", ""),
                    "sort_variable": row["sort_variable"],
                    "channel_name": row["channel_name"],
                    "channel_idx": row["channel_idx"],
                    "erp_class": row["erp_class"],
                    "export_batch": row["export_batch"],
                    "tracking_key": row["tracking_key"],
                    "label_studio_project_id": row["label_studio_project_id"],
                    "label_studio_task_id": row["label_studio_task_id"],
                    "image_file": row["image_file"],
                }
            )
    out.sort(key=lambda row: (str(row["dataset_key"]), str(row["sort_variable"]), str(row["channel_name"])))
    return out


def source_reference_rows(dataset_summary: list[dict[str, object]], refs: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for row in dataset_summary:
        dataset_key = str(row["dataset_key"])
        ref = refs.get(dataset_key, {})
        out.append(
            {
                "dataset_key": dataset_key,
                "dataset_label": row.get("dataset_label", ""),
                "week19_notebook_path": ref.get("week19_notebook_path", ""),
                "source_component": ref.get("source_component") or ref.get("component") or "",
                "reader_docs": ref.get("reader_docs", ""),
                "h5_path": ref.get("h5_path", ""),
                "events_path": ref.get("events_path", ""),
                "selected_sort_columns": ";".join(ref.get("selected_sort_columns", []) or []),
                "recommended_sort_columns": ";".join(ref.get("recommended_sort_columns", []) or []),
                "reference_files": ";".join(ref.get("reference_files", []) or []),
            }
        )
    return out


def plot_stacked_no_class_pattern(dataset_summary: list[dict[str, object]], path: Path) -> None:
    rows = list(reversed(dataset_summary))
    labels = [short_label(str(row["dataset_key"])) for row in rows]
    no_class = [int(row["no_class_labeled"]) for row in rows]
    pattern = [int(row["pattern_labeled"]) for row in rows]
    y = list(range(len(rows)))

    fig_h = max(5.0, 0.38 * len(rows) + 1.2)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(y, no_class, color=CLASS_COLORS["no_class"], label="no_class")
    ax.barh(y, pattern, left=no_class, color="#d95f02", label="pattern class")
    for i, row in enumerate(rows):
        total = int(row["total_labeled"])
        pos = int(row["pattern_labeled"])
        ax.text(total + max(3, total * 0.01), i, f"{pos}/{total}", va="center", fontsize=8)
    ax.set_yticks(y, labels)
    ax.set_xlabel("labeled ERP images")
    ax.set_title("Week 21 labeled images by data source")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_positive_rate(dataset_summary: list[dict[str, object]], path: Path) -> None:
    rows = sorted(dataset_summary, key=lambda row: float(row["positive_rate"]))
    labels = [short_label(str(row["dataset_key"])) for row in rows]
    values = [float(row["positive_rate"]) for row in rows]
    y = list(range(len(rows)))

    fig_h = max(5.0, 0.38 * len(rows) + 1.2)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    ax.barh(y, values, color="#1b9e77")
    for i, row in enumerate(rows):
        ax.text(min(values[i] + 0.012, 0.98), i, f"{100 * values[i]:.1f}%", va="center", fontsize=8)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, max(0.25, min(1.0, max(values or [0]) + 0.12)))
    ax.set_xlabel("pattern-class rate")
    ax.set_title("Pattern-class rate by data source")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_classes_by_dataset(dataset_class_rows: list[dict[str, object]], dataset_summary: list[dict[str, object]], path: Path) -> None:
    datasets = [str(row["dataset_key"]) for row in reversed(dataset_summary) if int(row["pattern_labeled"]) > 0]
    counts = Counter((str(row["dataset_key"]), str(row["erp_class"])) for row in dataset_class_rows)
    y = list(range(len(datasets)))
    left = [0] * len(datasets)

    fig_h = max(4.5, 0.42 * len(datasets) + 1.5)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    for cls in PATTERN_CLASSES:
        vals = [counts[(dataset_key, cls)] for dataset_key in datasets]
        ax.barh(y, vals, left=left, color=CLASS_COLORS[cls], label=cls)
        left = [a + b for a, b in zip(left, vals)]
    ax.set_yticks(y, [short_label(key) for key in datasets])
    ax.set_xlabel("pattern-class instances")
    ax.set_title("Manual pattern classes found by data source")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_export_batches(batch_summary: list[dict[str, object]], path: Path) -> None:
    rows = list(reversed(batch_summary))
    labels = [str(row["export_batch"]) for row in rows]
    no_class = [int(row["no_class_labeled"]) for row in rows]
    pattern = [int(row["pattern_labeled"]) for row in rows]
    y = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.55 * len(rows) + 1.5)))
    ax.barh(y, no_class, color=CLASS_COLORS["no_class"], label="no_class")
    ax.barh(y, pattern, left=no_class, color="#d95f02", label="pattern class")
    ax.set_yticks(y, labels)
    ax.set_xlabel("labeled ERP images")
    ax.set_title("Labeling volume by export batch")
    ax.legend()
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_top_positive_sort_variables(positive_sort_rows: list[dict[str, object]], path: Path, top_n: int = 25) -> None:
    rows = positive_sort_rows[:top_n]
    labels = [f"{row['dataset_key']} | {row['sort_variable']}" for row in rows]
    labels = [short_label(label, 54) for label in labels]
    y = list(range(len(rows)))
    left = [0] * len(rows)

    fig_h = max(6.0, 0.36 * len(rows) + 1.2)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    for cls in PATTERN_CLASSES:
        vals = []
        for row in rows:
            counts = json.loads(str(row["class_counts_json"]))
            vals.append(int(counts.get(cls, 0)))
        ax.barh(y, vals, left=left, color=CLASS_COLORS[cls], label=cls)
        left = [a + b for a, b in zip(left, vals)]
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("pattern-class instances")
    ax.set_title(f"Top {len(rows)} sort variables with pattern instances")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_sort_variable_heatmap(positive_sort_rows: list[dict[str, object]], path: Path) -> None:
    datasets = sorted({str(row["dataset_key"]) for row in positive_sort_rows})
    sort_counts = Counter(str(row["sort_variable"]) for row in positive_sort_rows)
    sort_variables = [name for name, _ in sort_counts.most_common(28)]
    if not datasets or not sort_variables:
        return

    value_lookup = defaultdict(int)
    for row in positive_sort_rows:
        key = (str(row["dataset_key"]), str(row["sort_variable"]))
        value_lookup[key] += int(row["pattern_labeled"])

    matrix = [[value_lookup[(dataset, sort_var)] for sort_var in sort_variables] for dataset in datasets]
    fig_w = max(10, 0.38 * len(sort_variables) + 4)
    fig_h = max(4.8, 0.38 * len(datasets) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(sort_variables)), sort_variables, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(datasets)), [short_label(key, 32) for key in datasets], fontsize=8)
    ax.set_title("Pattern-class count by data source and sort variable")
    ax.set_xlabel("sort variable")
    ax.set_ylabel("data source")
    for i, row_vals in enumerate(matrix):
        for j, val in enumerate(row_vals):
            if val:
                ax.text(j, i, str(val), ha="center", va="center", fontsize=7, color="black")
    fig.colorbar(im, ax=ax, label="pattern-class count")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_annotation_lead_time(rows: list[dict[str, str]], path: Path) -> None:
    batch_values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        lead_time = safe_float(row.get("annotation_lead_time"))
        if math.isfinite(lead_time) and lead_time > 0:
            batch_values[row["export_batch"]].append(min(lead_time, 60.0))
    batches = [batch for batch, values in batch_values.items() if values]
    batches.sort(key=lambda batch: len(batch_values[batch]), reverse=True)
    data = [batch_values[batch] for batch in batches]

    fig, ax = plt.subplots(figsize=(10, max(4.0, 0.55 * len(batches) + 1.2)))
    ax.boxplot(data, vert=False, tick_labels=batches, showfliers=False)
    ax.set_xlabel("annotation lead time, clipped at 60s")
    ax.set_title("Label Studio annotation time by export batch")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def build_summary(repo_root: Path | None = None) -> dict[str, object]:
    repo_root = repo_root or find_repo_root()
    week21 = repo_root / "notebooks" / "week_21"
    output_dir = week21 / "outputs" / "week21_labeling_summary"
    plots_dir = output_dir / "plots"
    tables_dir = output_dir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    annotations_csv = week21 / "labelstudio_annotations_all.csv"
    rows = read_csv_rows(annotations_csv)
    rows = [row for row in rows if row.get("label_status") == "classified"]

    dataset_summary = summarize_by_dataset(rows)
    batch_summary = summarize_by_export_batch(rows)
    dataset_class_rows = summarize_dataset_classes(rows)
    sort_variable_rows, positive_sort_rows = summarize_sort_variables(rows)
    positive_instance_rows = positive_instances(rows)
    reference_rows = source_reference_rows(dataset_summary, load_source_references(week21))

    write_csv_rows(tables_dir / "used_data_sources_summary.csv", dataset_summary)
    write_csv_rows(tables_dir / "export_batch_summary.csv", batch_summary)
    write_csv_rows(tables_dir / "dataset_class_summary.csv", dataset_class_rows)
    write_csv_rows(tables_dir / "sort_variable_summary.csv", sort_variable_rows)
    write_csv_rows(tables_dir / "positive_sort_variables_summary.csv", positive_sort_rows)
    write_csv_rows(tables_dir / "positive_instances.csv", positive_instance_rows)
    write_csv_rows(tables_dir / "source_references_summary.csv", reference_rows)

    plot_paths = {
        "labeled_images_by_dataset": plots_dir / "labeled_images_by_dataset.png",
        "positive_rate_by_dataset": plots_dir / "positive_rate_by_dataset.png",
        "pattern_classes_by_dataset": plots_dir / "pattern_classes_by_dataset.png",
        "labels_by_export_batch": plots_dir / "labels_by_export_batch.png",
        "top_positive_sort_variables": plots_dir / "top_positive_sort_variables.png",
        "positive_sort_variable_heatmap": plots_dir / "positive_sort_variable_heatmap.png",
        "annotation_lead_time_by_batch": plots_dir / "annotation_lead_time_by_batch.png",
    }

    plot_stacked_no_class_pattern(dataset_summary, plot_paths["labeled_images_by_dataset"])
    plot_positive_rate(dataset_summary, plot_paths["positive_rate_by_dataset"])
    plot_classes_by_dataset(dataset_class_rows, dataset_summary, plot_paths["pattern_classes_by_dataset"])
    plot_export_batches(batch_summary, plot_paths["labels_by_export_batch"])
    plot_top_positive_sort_variables(positive_sort_rows, plot_paths["top_positive_sort_variables"])
    plot_sort_variable_heatmap(positive_sort_rows, plot_paths["positive_sort_variable_heatmap"])
    plot_annotation_lead_time(rows, plot_paths["annotation_lead_time_by_batch"])

    totals = {
        "classified_annotations": len(rows),
        "data_sources": len({row["dataset_key"] for row in rows}),
        "export_batches": len({row["export_batch"] for row in rows}),
        "pattern_instances": len(positive_instance_rows),
        "no_class_instances": len(rows) - len(positive_instance_rows),
        "sort_variables_with_patterns": len(positive_sort_rows),
        "datasets_with_patterns": sum(1 for row in dataset_summary if int(row["pattern_labeled"]) > 0),
        "excluded_training_datasets": sorted(EXCLUDED_TRAINING_DATASETS),
    }
    summary = {
        "output_dir": str(output_dir),
        "tables_dir": str(tables_dir),
        "plots_dir": str(plots_dir),
        "annotations_csv": str(annotations_csv),
        "totals": totals,
        "top_positive_sort_variables": positive_sort_rows[:12],
        "dataset_summary": dataset_summary,
        "plot_paths": {key: str(value) for key, value in plot_paths.items() if value.exists()},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    summary = build_summary()
    print(json.dumps(summary["totals"], indent=2))
    print("Output:", summary["output_dir"])


if __name__ == "__main__":
    main()
