"""Directory name parser: extracts version/step/think metadata from results directory names."""

from __future__ import annotations

import re
from pathlib import Path

from auto_eval.config import EXTRA_VERSIONS
from auto_eval.tasks_meta import get_all_versions
from .data import (
    ModelSpec,
    build_group_title,
    build_model_key,
    compact_step,
    read_converted,
)

DIR_PATTERN_WITH_STEP = re.compile(
    r'^results_(.+?)_step(\d+)_(think|nonthink|nothink)(.*)$'
)
DIR_PATTERN_NO_STEP = re.compile(
    r'^results_(.+?)_(think|nonthink|nothink)(.*)$'
)


def parse_directory_name(dir_name: str) -> dict | None:
    """Parse a results directory name into metadata components.

    Supported formats:
    1) results_{version}_step{N}_{think|nonthink|nothink}{suffix}
    2) results_{version}_{think|nonthink|nothink}{suffix}  (step defaults to "0")

    Returns dict with keys: version, step, think_enabled, kind, suffix
    or None if the name doesn't match.
    """
    match = DIR_PATTERN_WITH_STEP.match(dir_name)
    if match:
        version = match.group(1)
        step = match.group(2)
        think_token = match.group(3)  # think | nonthink | nothink
        suffix = match.group(4)  # e.g. "_beam128" or ""
    else:
        match = DIR_PATTERN_NO_STEP.match(dir_name)
        if not match:
            return None
        version = match.group(1)
        step = "NA"
        think_token = match.group(2)  # think | nonthink | nothink
        suffix = match.group(3)  # e.g. "_beam128" or ""

    think_enabled = (think_token == "think")
    kind = "think" if think_enabled else "nothink"

    return {
        "version": version,
        "step": step,
        "think_enabled": think_enabled,
        "kind": kind,
        "suffix": suffix,
    }


def resolve_result_dir(dir_name: str, results_roots: list[Path]) -> Path | None:
    """Find the actual path for a results directory, searching all result roots."""
    for results_root in results_roots:
        for version_dir in get_all_versions() + EXTRA_VERSIONS:
            candidate = results_root / version_dir / dir_name
            if candidate.exists():
                return candidate
    return None


def build_model_specs(dir_names: list[str], results_roots: list[Path]) -> list[ModelSpec]:
    """Build ModelSpec objects from a list of directory names.

    Directories that don't match the expected pattern or don't exist in any
    result root are skipped.
    epoch is set to empty string since it's not available from directory names.
    group_key uses version|step|suffix for think/nothink pairing.
    """
    models: list[ModelSpec] = []
    for dir_name in dir_names:
        parsed = parse_directory_name(dir_name)
        if parsed is None:
            continue

        result_dir = resolve_result_dir(dir_name, results_roots)
        if result_dir is None:
            continue

        version = parsed["version"]
        step = parsed["step"]
        kind = parsed["kind"]
        suffix = parsed["suffix"]
        epoch = ""

        # group_key pairs think/nothink dirs that share version+step+suffix
        group_key = f"{version}|{step}|{suffix}"
        group_title = build_group_title(version)
        group_meta = f"{compact_step(step)}{(' · ' + suffix.lstrip('_')) if suffix else ''}"

        models.append(
            ModelSpec(
                key=build_model_key(dir_name, kind),
                source_dir=dir_name,
                actual_source_dir=dir_name,
                archive_dir=dir_name,
                version=version,
                step=step,
                epoch=epoch,
                think_enabled=parsed["think_enabled"],
                kind=kind,
                group_key=group_key,
                group_title=group_title,
                group_meta=group_meta,
                result_dir=result_dir,
                converted=read_converted(result_dir),
            )
        )

    return models
