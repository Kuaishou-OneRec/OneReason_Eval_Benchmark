import argparse
import os
import re

from benchmark import Benchmark

_BASE_VERSION_RE = re.compile(r'^(v\d+\.\d+)')

BASE_DATA_DIR = os.environ.get("BENCHMARK_DATA_DIR", "data")


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--output_dir",
        type=str, default=None,
        help="The directory where the generation results are saved."
    )
    group.add_argument(
        "--result_dir",
        type=str, default=None,
        help="Parent directory whose subdirectories are each treated as an output_dir."
    )
    parser.add_argument(
        "--version",
        type=str, default="v3.1",
        help="比赛任务版本 v3.1（datav3.1）；data_dir 独立指定"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing metrics and recompute from scratch"
    )
    parser.add_argument(
        "--task_types",
        type=str,
        nargs='+',
        default=None,
        help="Competition task names. Defaults to all 11 tasks."
    )
    parser.add_argument(
        "--sid_encode_fn",
        type=str,
        default=None,
        help="自定义 SID 编码函数（lambda 表达式），如 \"lambda x:'.'.join(str(c) for c in x[:2])\""
    )
    parser.add_argument(
        "--sid_pattern",
        type=str,
        default=None,
        help="自定义 SID 解析正则；所有数字捕获组会按顺序组成 SID tuple。传入后覆盖默认 SID 解析规则。"
    )
    return parser.parse_args()


def evaluate_single_output_dir(output_dir, data_dir, overwrite, task_types, benchmark_version, sid_encode_fn, sid_pattern):
    """Run evaluation on a single output_dir."""
    eval_results_path = f"{output_dir}/eval_results.json"
    Benchmark.evaluate_dev(
        generation_results_dir=output_dir,
        output_path=eval_results_path,
        data_dir=data_dir,
        overwrite=overwrite,
        task_types=task_types,
        benchmark_version=benchmark_version,
        sid_encode_fn=sid_encode_fn,
        sid_pattern=sid_pattern,
    )


def main():
    args = get_args()
    data_dir = args.data_dir or BASE_DATA_DIR

    benchmark_version = args.version
    if args.version:
        m = _BASE_VERSION_RE.match(args.version)
        if m:
            benchmark_version = m.group(1)

    if args.result_dir:
        # Batch mode: iterate subdirectories of result_dir as output_dirs
        output_dirs = sorted([
            d for d in os.listdir(args.result_dir)
            if os.path.isdir(os.path.join(args.result_dir, d)) and not d.startswith('.')
        ])
        print(f"Batch mode: found {len(output_dirs)} subdirectories in {args.result_dir}")

        for i, subdir in enumerate(output_dirs, 1):
            output_dir = os.path.join(args.result_dir, subdir)
            print(f"\n{'='*60}")
            print(f"[{i}/{len(output_dirs)}] output_dir: {output_dir}")
            print(f"{'='*60}")
            evaluate_single_output_dir(
                output_dir=output_dir,
                data_dir=data_dir,
                overwrite=args.overwrite,
                task_types=args.task_types,
                benchmark_version=benchmark_version,
                sid_encode_fn=args.sid_encode_fn,
                sid_pattern=args.sid_pattern,
            )
    else:
        # Single output_dir mode
        evaluate_single_output_dir(
            output_dir=args.output_dir,
            data_dir=data_dir,
            overwrite=args.overwrite,
            task_types=args.task_types,
            benchmark_version=benchmark_version,
            sid_encode_fn=args.sid_encode_fn,
            sid_pattern=args.sid_pattern,
        )


if __name__ == "__main__":
    main()
