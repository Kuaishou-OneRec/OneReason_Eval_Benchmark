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



def main():
    """从原始 parquet 创建反向映射：PID → (code1, code2, code3)"""
    print("开始数据处理...")
    print("=" * 60)

    # 读取数据
    print("读取原始数据...")
    df = pd.read_parquet(INPUT_PATH)
    row_count = len(df)
    memory_usage = df.memory_usage(deep=True).sum() / (1024**3)
    print(f"数据: {row_count:,} 行")
    print(f"内存使用: {memory_usage:.2f} GB")

    unique_pids = df['pid'].nunique()
    print(f"唯一 PIDs: {unique_pids:,}")

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
    print(f"  数据行数: {row_count:,}")
    print(f"  唯一 PIDs: {unique_pids:,}")
    print(f"  最终字典大小: {len(pid_to_code):,} 条")
    print("=" * 60)


if __name__ == "__main__":
    main()
