"""
将原始 parquet 转换为 PID → SID 反向映射的 pickle 格式
"""
import os
import pandas as pd
import pickle
from pathlib import Path

# 路径配置
INPUT_PATH = os.environ.get("PID2SID_PARQUET", "data/pid2sid_merged.parquet")
PICKLE_PATH = os.environ.get("PID_TO_SID_PICKLE", "data/pid_to_code.pkl")
POPULARITY_PATH = os.environ.get("CANDIDATE_PID_FILE", "data/candidate_pids.txt")


def load_valid_pids(filepath: str) -> set:
    """
    从 popularity 文件中加载有效的 PID 列表

    Args:
        filepath: popularity 文件路径

    Returns:
        包含有效 PID 的集合
    """
    print(f"从 popularity 文件加载 valid PIDs: {filepath}")

    valid_pids = set()

    with open(filepath, 'r') as f:
        # 跳过 header 行（列名）
        header = f.readline()
        # 跳过分隔符行
        separator = f.readline()

        # 读取数据行
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 解析第一列（PID）
            parts = line.split()
            if len(parts) >= 1:
                try:
                    pid = int(parts[0])
                    valid_pids.add(pid)
                except ValueError:
                    print(f"  警告: 无法解析 PID: {parts[0]}")
                    continue

    print(f"从 popularity 文件加载了 {len(valid_pids)} 个 valid PIDs")
    return valid_pids


def main():
    """从原始 parquet 创建反向映射：PID → (code1, code2, code3)"""
    print("开始数据处理...")
    print("=" * 60)

    # Step 1: 加载 valid PIDs
    valid_pids = load_valid_pids(POPULARITY_PATH)
    print("=" * 60)

    # Step 2: 读取数据
    print("读取原始数据...")
    df = pd.read_parquet(INPUT_PATH)
    original_row_count = len(df)
    original_memory = df.memory_usage(deep=True).sum() / (1024**3)
    print(f"原始数据: {original_row_count:,} 行")
    print(f"内存使用: {original_memory:.2f} GB")

    # Step 3: 早期过滤 - 只保留 valid PIDs
    print("\n过滤数据，只保留 valid PIDs...")
    df = df[df['pid'].isin(valid_pids)]
    filtered_row_count = len(df)
    filtered_unique_pids = df['pid'].nunique()
    print(f"过滤后数据: {filtered_row_count:,} 行 (保留 {filtered_unique_pids:,} 个唯一 PIDs)")
    print(f"过滤掉: {original_row_count - filtered_row_count:,} 行")

    # IMPORTANT: 重置索引避免索引不连续导致的 NaN 问题
    print("重置索引...")
    df = df.reset_index(drop=True)

    # Step 4: 检查有多少 valid PIDs 在 parquet 中找不到
    pids_in_parquet = set(df['pid'].unique())
    missing_pids = valid_pids - pids_in_parquet
    if missing_pids:
        print(f"\n⚠️  警告: {len(missing_pids):,} 个 valid PIDs 在 parquet 中未找到")
        if len(missing_pids) <= 10:
            print(f"  缺失的 PIDs: {sorted(missing_pids)}")
    else:
        print(f"\n✓ 所有 {len(valid_pids):,} 个 valid PIDs 都在 parquet 中找到")

    print("=" * 60)

    # 拆分 code 列
    print("拆分 code 列...")
    df[['code1', 'code2', 'code3']] = pd.DataFrame(df['code'].tolist())

    # 立即转换为 int64 类型，避免 NaN 问题
    df['code1'] = df['code1'].astype('int64')
    df['code2'] = df['code2'].astype('int64')
    df['code3'] = df['code3'].astype('int64')

    # 删除原始 code 列
    df = df.drop('code', axis=1)

    # 检查数值范围
    print("检查数值范围...")
    for col in ['code1', 'code2', 'code3']:
        min_val, max_val = df[col].min(), df[col].max()
        print(f"  {col}: [{min_val}, {max_val}]")

    # 检查 PID 唯一性
    print("\n检查 PID 唯一性...")
    duplicate_pids = df[df['pid'].duplicated()]['pid'].unique()
    if len(duplicate_pids) > 0:
        print(f"  警告: 发现 {len(duplicate_pids):,} 个重复的 PID")
        print(f"  保留每个 PID 的第一个映射")
    else:
        print(f"  ✓ 无重复 PID")

    # 构建反向字典: PID → (code1, code2, code3)
    print("\n构建反向映射字典...")
    # 使用 drop_duplicates 保留每个 PID 的第一个出现
    df_unique = df.drop_duplicates(subset='pid', keep='first')

    # 创建字典: PID → tuple(code1, code2, code3)
    # 注意: code1/code2/code3 已经是 int64 类型，但转换为 Python int 以确保兼容性
    pid_to_code = {}
    for _, row in df_unique.iterrows():
        pid_to_code[row['pid']] = (int(row['code1']), int(row['code2']), int(row['code3']))

    print(f"字典大小: {len(pid_to_code):,} 条")

    # 释放 DataFrame 内存
    del df, df_unique

    # 保存为 pickle
    print("\n保存 pickle 到: {PICKLE_PATH}")
    print("=" * 60)
    with open(PICKLE_PATH, 'wb') as f:
        pickle.dump(pid_to_code, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 验证文件大小
    pkl_path = Path(PICKLE_PATH)
    if pkl_path.exists():
        pkl_size = pkl_path.stat().st_size / (1024**3)
        pkl_size_mb = pkl_path.stat().st_size / (1024**2)
        print(f"✓ Pickle 文件大小: {pkl_size:.3f} GB ({pkl_size_mb:.1f} MB)")

        # 估算未过滤时的文件大小（基于过滤比例）
        if filtered_unique_pids > 0:
            estimated_full_size = pkl_size * (len(valid_pids) / filtered_unique_pids) if filtered_unique_pids < len(valid_pids) else pkl_size
            compression_ratio = (1 - pkl_size / estimated_full_size) * 100 if estimated_full_size > pkl_size else 0
            print(f"  预估未过滤时文件大小: {estimated_full_size:.3f} GB")
            if compression_ratio > 0:
                print(f"  文件大小减少: {compression_ratio:.1f}%")

    print("=" * 60)
    print("✓ 处理完成！")
    print("=" * 60)

    # 验证几个样例
    print("\n验证样例:")
    sample_pids = sorted(list(pid_to_code.keys()))[:5]
    for pid in sample_pids:
        codes = pid_to_code[pid]
        print(f"  PID {pid:,} → SID codes: {codes}")

    # 最终统计摘要
    print("\n" + "=" * 60)
    print("统计摘要:")
    print(f"  Valid PIDs (来自 popularity): {len(valid_pids):,}")
    print(f"  在 parquet 中找到的 PIDs: {filtered_unique_pids:,}")
    print(f"  未找到的 PIDs: {len(missing_pids):,}")
    print(f"  最终字典大小: {len(pid_to_code):,} 条")
    print(f"  原始数据行数: {original_row_count:,}")
    print(f"  过滤后数据行数: {filtered_row_count:,}")
    print(f"  数据压缩比: {(1 - filtered_row_count/original_row_count)*100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
