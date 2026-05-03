#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import django


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPORT_ROOT = REPO_ROOT / "notebooks" / "week_21" / "labelstudio_export_model_prioritized_200"
PROJECT_TITLE_PREFIX = "m200_"
USER_EMAIL = "benjamin.benji20+label-studio@gmail.com"


def unique_project_title(Project, base: str) -> str:
    max_len = 50
    base = base[:max_len]
    title = base
    suffix = 2
    while Project.objects.filter(title=title).exists():
        tail = f"_{suffix}"
        title = base[: max_len - len(tail)] + tail
        suffix += 1
    return title


def task_file_for_dataset(dataset_dir: Path) -> Path:
    matches = sorted(dataset_dir.glob("tasks_*.json"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one tasks_*.json in {dataset_dir}, found {len(matches)}")
    return matches[0]


def update_tracking_project_id(path: Path, dataset_key: str, project_id: int) -> None:
    if not path.exists():
        return
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []
    if "label_studio_project_id" not in fieldnames:
        fieldnames.append("label_studio_project_id")
    changed = False
    for row in rows:
        if row.get("dataset_key") == dataset_key and row.get("export_batch") == "week21_model_prioritized_200":
            row["label_studio_project_id"] = str(project_id)
            changed = True
    if changed:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    django.setup()

    from django.contrib.auth import get_user_model
    from django.db import transaction
    from data_import.serializers import ImportApiSerializer
    from io_storages.localfiles.models import LocalFilesImportStorage
    from projects.models import Project

    User = get_user_model()
    user = User.objects.filter(email=USER_EMAIL).first() or User.objects.order_by("id").first()
    if user is None:
        raise RuntimeError("No Label Studio user exists.")

    summary_path = EXPORT_ROOT / "summary.csv"
    label_config_default = (EXPORT_ROOT / "eye_eeg_reading_fixations" / "labeling_interface.xml").read_text()
    imported_rows: list[dict[str, str | int]] = []

    with summary_path.open(newline="") as f:
        summary_rows = list(csv.DictReader(f))

    for summary in summary_rows:
        dataset_key = summary["dataset_key"]
        dataset_dir = EXPORT_ROOT / dataset_key
        tasks_path = task_file_for_dataset(dataset_dir)
        label_config_path = dataset_dir / "labeling_interface.xml"
        label_config = label_config_path.read_text() if label_config_path.exists() else label_config_default
        tasks = json.loads(tasks_path.read_text())
        expected_count = int(summary["exported_count"])
        if len(tasks) != expected_count:
            raise RuntimeError(f"{dataset_key}: tasks file has {len(tasks)} rows, summary has {expected_count}")

        base_title = PROJECT_TITLE_PREFIX + dataset_key
        title = unique_project_title(Project, base_title)
        description = (
            "Week 21 model-prioritized ERP export. "
            f"Batch=week21_model_prioritized_200; dataset={dataset_key}; "
            f"exported_count={expected_count}; source={dataset_dir}"
        )

        with transaction.atomic():
            project = Project.objects.create(
                title=title,
                description=description,
                label_config=label_config,
                created_by=user,
                organization=user.active_organization,
            )
            serializer = ImportApiSerializer(data=tasks, many=True, context={"project": project, "user": user})
            serializer.is_valid(raise_exception=True)
            task_instances = serializer.save(project_id=project.id)
            recalculate_stats_counts = {
                "task_count": len(task_instances),
                "annotation_count": len(serializer.db_annotations),
                "prediction_count": len(serializer.db_predictions),
            }
            project.update_tasks_counters_and_task_states(
                tasks_queryset=task_instances,
                maximum_annotations_changed=False,
                overlap_cohort_percentage_changed=False,
                tasks_number_changed=True,
                recalculate_stats_counts=recalculate_stats_counts,
            )
            project.summary.update_data_columns(tasks)
            LocalFilesImportStorage.objects.create(
                project=project,
                title=f"{title} local files",
                path=str(dataset_dir),
                regex_filter=r".*\.png$",
                use_blob_urls=False,
            )

        update_tracking_project_id(EXPORT_ROOT / "already_classified_tracking.csv", dataset_key, project.id)
        update_tracking_project_id(dataset_dir / "classified_combinations.csv", dataset_key, project.id)

        imported_rows.append(
            {
                "dataset_key": dataset_key,
                "project_id": project.id,
                "project_title": title,
                "task_count": len(task_instances),
                "tasks_path": str(tasks_path),
                "dataset_dir": str(dataset_dir),
            }
        )
        print(f"Imported {dataset_key}: project_id={project.id}, title={title}, tasks={len(task_instances)}")

    out_path = EXPORT_ROOT / "label_studio_projects.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset_key", "project_id", "project_title", "task_count", "tasks_path", "dataset_dir"],
        )
        writer.writeheader()
        writer.writerows(imported_rows)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
