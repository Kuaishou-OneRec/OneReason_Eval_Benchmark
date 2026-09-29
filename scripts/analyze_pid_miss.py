"""
分析 SID 命中但 PID 未命中的原因。
检查是否因为 sid2pid.json 映射缺失导致 PID 为 0。
"""

import json
import re
import sys
import os


def parse_sids(text):
    """从generation或ground_truth中提取sid元组列表，支持8-token和3-token"""
    # 先尝试8-token
    pattern_8 = r'<s_a_(\d+)><s_b_(\d+)><s_c_(\d+)><s_d_(\d+)><s_e_(\d+)><s_f_(\d+)><s_g_(\d+)><s_h_(\d+)>'
    matches = re.findall(pattern_8, text)
    if matches:
        return [tuple(m) for m in matches]
    # 回退3-token
    pattern_3 = r'<s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>'
    matches = re.findall(pattern_3, text)
    return [tuple(m) for m in matches]


def analyze_task(filepath):
    """分析单个任务的 test_generated.json"""
    with open(filepath, 'r') as f:
        data = json.load(f)

    samples = data['samples']
    total_samples = len(samples)

    # 找 pass@64=True 且 pid_pass@64=False 的样本
    target_samples = []
    for sid, sample in samples.items():
        if sample.get('pass@64') == True and sample.get('pid_pass@64') == False:
            target_samples.append((sid, sample))

    print(f"  总样本数: {total_samples}")
    print(f"  pass@64=True 且 pid_pass@64=False 的样本数: {len(target_samples)}")

    total_matching_generations = 0
    total_pid_zero = 0

    for sid, sample in target_samples:
        generations = sample.get('generations', [])
        pid_generations = sample.get('pid_generations', [])
        ground_truth = sample.get('ground_truth', '')

        gt_sids = parse_sids(ground_truth)
        gt_sid_set = set(gt_sids)

        for i, gen in enumerate(generations):
            gen_sids = parse_sids(gen)
            if gen_sids and gen_sids[0] in gt_sid_set:
                total_matching_generations += 1
                if i < len(pid_generations) and pid_generations[i] == 0:
                    total_pid_zero += 1

    print(f"  匹配ground_truth的generation总数: {total_matching_generations}")
    print(f"  其中pid=0(JSON映射缺失)的数量: {total_pid_zero}")
    if total_matching_generations > 0:
        print(f"  映射缺失率: {total_pid_zero / total_matching_generations:.2%}")
    print()


def main():
    if len(sys.argv) < 2:
        print("用法:")
        print("  分析单个任务: python3 scripts/analyze_pid_miss.py <test_generated.json路径>")
        print("  分析整个目录: python3 scripts/analyze_pid_miss.py <output_dir> [task1 task2 ...]")
        sys.exit(1)

    path = sys.argv[1]

    # 单文件模式
    if path.endswith('.json'):
        print(f"分析文件: {path}\n")
        analyze_task(path)
        return

    # 目录模式: 遍历 output_dir/model_name/task_name/test_generated.json
    tasks_filter = sys.argv[2:] if len(sys.argv) > 2 else None

    for model_name in sorted(os.listdir(path)):
        model_dir = os.path.join(path, model_name)
        if not os.path.isdir(model_dir):
            continue

        print(f"模型: {model_name}")
        print("=" * 60)

        for task_name in sorted(os.listdir(model_dir)):
            if tasks_filter and task_name not in tasks_filter:
                continue
            task_dir = os.path.join(model_dir, task_name)
            if not os.path.isdir(task_dir):
                continue
            gen_file = os.path.join(task_dir, 'test_generated.json')
            if not os.path.exists(gen_file):
                continue

            print(f"\n任务: {task_name}")
            print("-" * 40)
            try:
                analyze_task(gen_file)
            except Exception as e:
                print(f"  错误: {e}\n")


if __name__ == '__main__':
    main()
