"""
CLI entry point for submitting GPU evaluation tasks to the incoming queue.

Usage:
    python -m auto_eval.submit_cli --model-path /path/to/ckpt [options]

Model path validation is intentionally skipped in CLI mode.

Parameter priority (highest to lowest):
  1. Explicit CLI argument
  2. Environment variable
  3. Python code default value
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from auto_eval.config import EMAIL_DOMAIN
from auto_eval.utils import build_email_from_username as build_email_util
from auto_eval.submit.core import build_eval_task, submit_eval_task, generate_output_suffix

# ---------------------------------------------------------------------------
# Defaults (lowest priority)
# ---------------------------------------------------------------------------
DEFAULT_QUEUE_DIR = Path(os.environ.get("BENCHMARK_QUEUE_DIR", "workspace/auto_eval/v3.1/incoming"))

DEFAULTS = {
    "version": "v3.1",
    "tasks": "all",
    "rollout_mode": "Race",
    "enable_thinking": False,
    "seed": 42,
    "overwrite": False,
    "queue_dir": None,
    "output_suffix": None,
    "user": None,
    "email": None,
}

# ---------------------------------------------------------------------------
# Environment variable helpers
# ---------------------------------------------------------------------------

def _parse_bool_env(name: str) -> Optional[bool]:
    """Parse a boolean environment variable.

    Accepted values (case-insensitive):
      True  -> "1" or "true"
      False -> "0" or "false"
      Absent -> None (not set)

    Any other value raises SystemExit with a clear error message.
    """
    val = os.environ.get(name)
    if val is None:
        return None
    normalized = val.strip().lower()
    if normalized in ("1", "true"):
        return True
    if normalized in ("0", "false"):
        return False
    print(
        f"[ERROR] Environment variable {name}={val!r} is invalid. "
        "Accepted values: '1', 'true', '0', 'false' (case-insensitive).",
        file=sys.stderr,
    )
    sys.exit(1)


def _env_str(name: str) -> Optional[str]:
    """Return stripped env var value, or None if absent/empty."""
    val = os.environ.get(name)
    return val.strip() if val and val.strip() else None


def _env_int(name: str) -> Optional[int]:
    """Parse an integer environment variable, or None if absent."""
    val = os.environ.get(name)
    if val is None:
        return None
    try:
        return int(val.strip())
    except ValueError:
        print(
            f"[ERROR] Environment variable {name}={val!r} is not a valid integer.",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Email resolution
# ---------------------------------------------------------------------------

def _resolve_email(user: Optional[str], email: Optional[str]) -> str:
    if email:
        return email
    if user:
        return build_email_util(user, EMAIL_DOMAIN)
    # fall back to OS user
    os_user = os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"
    return build_email_util(os_user, EMAIL_DOMAIN)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Submit a GPU evaluation task to the incoming queue.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Parameter priority: CLI argument > environment variable > default value\n\n"
            "Environment variables:\n"
            "  AUTO_EVAL_VERSION         -> --version\n"
            "  AUTO_EVAL_TASKS           -> --tasks\n"
            "  AUTO_EVAL_ROLLOUT_MODE    -> --rollout-mode\n"
            "  AUTO_EVAL_ENABLE_THINKING -> --enable-thinking / --no-enable-thinking\n"
            "  AUTO_EVAL_SEED            -> --seed\n"
            "  AUTO_EVAL_USER            -> --user\n"
            "  AUTO_EVAL_EMAIL           -> --email\n"
            "  AUTO_EVAL_OVERWRITE       -> --overwrite\n"
            "  AUTO_EVAL_QUEUE_DIR       -> --queue-dir\n"
            "  AUTO_EVAL_OUTPUT_SUFFIX   -> --output-suffix\n\n"
            "Boolean env vars accept: 1/true (enable) or 0/false (disable), case-insensitive."
        ),
    )
    parser.add_argument("--model-path", required=True, help="Path to model checkpoint")

    # All optional args use default=None so we can distinguish "not provided"
    # from "provided with a value". Actual defaults are applied after env-var
    # resolution (see DEFAULTS dict above).
    parser.add_argument("--version", default=None, help="Benchmark version (env: AUTO_EVAL_VERSION)")
    parser.add_argument(
        "--tasks",
        default=None,
        help="Tasks to evaluate: comma-separated task names (env: AUTO_EVAL_TASKS)",
    )
    parser.add_argument(
        "--output-suffix",
        default=None,
        help="Output suffix; auto-generated from model path if not provided (env: AUTO_EVAL_OUTPUT_SUFFIX)",
    )
    parser.add_argument(
        "--rollout-mode",
        default=None,
        help="Rollout mode, e.g. Default, Final, Normal 128, Think 64 (env: AUTO_EVAL_ROLLOUT_MODE)",
    )

    # Boolean flag pair: both default to None so we can detect "not passed"
    bool_group = parser.add_mutually_exclusive_group()
    bool_group.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        default=None,
        help="Enable thinking mode (env: AUTO_EVAL_ENABLE_THINKING)",
    )
    bool_group.add_argument(
        "--no-enable-thinking",
        dest="enable_thinking",
        action="store_false",
        help="Disable thinking mode (default)",
    )

    parser.add_argument("--seed", type=int, default=None, help="Random seed (env: AUTO_EVAL_SEED)")
    parser.add_argument("--user", default=None, help="Username, auto-appends @example.org (env: AUTO_EVAL_USER)")
    parser.add_argument("--email", default=None, help="Submitter email, overrides --user (env: AUTO_EVAL_EMAIL)")
    overwrite_group = parser.add_mutually_exclusive_group()
    overwrite_group.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        default=None,
        help="Overwrite existing results (env: AUTO_EVAL_OVERWRITE)",
    )
    overwrite_group.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Do not overwrite existing results (overrides AUTO_EVAL_OVERWRITE=true)",
    )
    parser.add_argument(
        "--queue-dir",
        default=None,
        help="Target queue directory (env: AUTO_EVAL_QUEUE_DIR)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print task JSON without writing to queue",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Priority resolution: CLI > env var > default
    # ------------------------------------------------------------------

    def resolve(cli_val, env_val, default):
        """Return first non-None value in priority order."""
        if cli_val is not None:
            return cli_val
        if env_val is not None:
            return env_val
        return default

    version = resolve(args.version, _env_str("AUTO_EVAL_VERSION"), DEFAULTS["version"])
    tasks = resolve(args.tasks, _env_str("AUTO_EVAL_TASKS"), DEFAULTS["tasks"])
    output_suffix = resolve(args.output_suffix, _env_str("AUTO_EVAL_OUTPUT_SUFFIX"), DEFAULTS["output_suffix"])
    rollout_mode = resolve(args.rollout_mode, _env_str("AUTO_EVAL_ROLLOUT_MODE"), DEFAULTS["rollout_mode"])
    enable_thinking = resolve(args.enable_thinking, _parse_bool_env("AUTO_EVAL_ENABLE_THINKING"), DEFAULTS["enable_thinking"])
    seed = resolve(args.seed, _env_int("AUTO_EVAL_SEED"), DEFAULTS["seed"])
    user = resolve(args.user, _env_str("AUTO_EVAL_USER"), DEFAULTS["user"])
    email = resolve(args.email, _env_str("AUTO_EVAL_EMAIL"), DEFAULTS["email"])
    overwrite = resolve(args.overwrite, _parse_bool_env("AUTO_EVAL_OVERWRITE"), DEFAULTS["overwrite"])
    queue_dir_str = resolve(args.queue_dir, _env_str("AUTO_EVAL_QUEUE_DIR"), DEFAULTS["queue_dir"])

    # ------------------------------------------------------------------
    # Downstream resolution
    # ------------------------------------------------------------------

    submitter_email = _resolve_email(user, email)

    if not output_suffix:
        output_suffix = generate_output_suffix(
            args.model_path,
            enable_thinking=enable_thinking,
            rollout_mode=rollout_mode,
            version=version,
            seed=seed,
        )

    task_data = build_eval_task(
        model_path=args.model_path,
        output_suffix=output_suffix,
        submitter_email=submitter_email,
        version=version,
        enable_thinking=enable_thinking,
        rollout_mode=rollout_mode,
        seed=seed,
        overwrite=overwrite,
        tasks=tasks,
    )

    if args.dry_run:
        print(json.dumps(task_data, indent=4))
        return

    queue_dir = Path(queue_dir_str) if queue_dir_str else DEFAULT_QUEUE_DIR

    try:
        json_filepath = submit_eval_task(task_data, queue_dir=queue_dir)
    except OSError as e:
        print(f"[ERROR] Failed to write task file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"task_id: {task_data['task_id']}")
    print(f"json_path: {json_filepath}")


if __name__ == "__main__":
    main()
