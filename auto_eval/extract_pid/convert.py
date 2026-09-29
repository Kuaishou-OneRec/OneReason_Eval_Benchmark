"""
将原始 parquet 转换为 pickle 格式的字典，加载更快
"""
import os
import pandas as pd
import pickle
from pathlib import Path

# 编码常量
CODE_MULTIPLIER_1 = 8192 * 8192  # 67108864
CODE_MULTIPLIER_2 = 8192

# 路径配置
INPUT_PATH = os.environ.get("PID2SID_PARQUET", "data/pid2sid_merged.parquet")
PICKLE_PATH = os.environ.get("SID_TO_PID_PICKLE", "data/code_to_pid.pkl")


def main():
    """直接从原始 parquet 转换为 pickle 格式的字典"""
    print("开始数据处理...")

    # 读取数据
    print("读取原始数据...")
    df = pd.read_parquet(INPUT_PATH)
    print(f"原始数据: {len(df)} 行")
    print(f"内存使用: {df.memory_usage(deep=True).sum() / (1024**3):.2f} GB")

    # 拆分 code 列
    print("拆分 code 列...")
    df[['code1', 'code2', 'code3']] = pd.DataFrame(df['code'].tolist())

    # 删除原始 code 列
    df = df.drop('code', axis=1)

    # 检查数值范围
    print("检查数值范围...")
    for col in ['code1', 'code2', 'code3']:
        min_val, max_val = df[col].min(), df[col].max()
        print(f"  {col}: [{min_val}, {max_val}]")

    # 编码 keys
    print("编码 keys...")
    keys = (df['code1'].astype('int64') * CODE_MULTIPLIER_1 +
            df['code2'].astype('int64') * CODE_MULTIPLIER_2 +
            df['code3'].astype('int64'))

    print(f"Keys 范围: [{keys.min()}, {keys.max()}]")

    # 构建字典
    print("构建字典...")
    code_to_pid = dict(zip(keys, df['pid']))
    print(f"字典大小: {len(code_to_pid)} 条")

    # 释放 DataFrame 内存
    del df, keys

    # 保存为 pickle
    print(f"保存 pickle 到: {PICKLE_PATH}")
    with open(PICKLE_PATH, 'wb') as f:
        pickle.dump(code_to_pid, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 验证文件大小
    pkl_path = Path(PICKLE_PATH)
    if pkl_path.exists():
        pkl_size = pkl_path.stat().st_size / (1024**3)
        print(f"Pickle 文件大小: {pkl_size:.2f} GB")

    print("处理完成！")


if __name__ == "__main__":
    main()
