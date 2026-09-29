"""
使用 DuckDB 快速查询 PID 对应的 SID codes
无需预处理，直接查询 parquet 文件
"""
import os
import duckdb
from typing import List, Tuple, Optional, Dict

# 路径配置
PARQUET_PATH = os.environ.get("PID2SID_PARQUET", "data/pid2sid_merged.parquet")


def query_single_pid(pid: int) -> Optional[Tuple[int, int, int]]:
    """查询单个 PID 的 codes"""
    result = duckdb.query(f"""
        SELECT code FROM '{PARQUET_PATH}'
        WHERE pid = {pid}
        LIMIT 1
    """).fetchone()

    if result:
        return tuple(result[0])
    return None


def query_multiple_pids(pids: List[int]) -> Dict[int, Tuple[int, int, int]]:
    """批量查询多个 PID 的 codes"""
    pids_str = ','.join(map(str, pids))

    results = duckdb.query(f"""
        SELECT pid, code FROM '{PARQUET_PATH}'
        WHERE pid IN ({pids_str})
    """).fetchall()

    return {row[0]: tuple(row[1]) for row in results}


def main():
    """交互式查询"""
    print("DuckDB PID 查询工具")
    print("=" * 40)
    print("输入 PID 进行查询（多个用逗号分隔）")
    print("输入 q 退出")
    print("=" * 40)

    while True:
        try:
            user_input = input("\n请输入 PID: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出")
            break

        if user_input.lower() == 'q':
            print("退出")
            break

        if not user_input:
            continue

        # 解析输入的 PID
        try:
            pids = [int(p.strip()) for p in user_input.split(',')]
        except ValueError:
            print("错误: 请输入有效的数字")
            continue

        # 查询
        if len(pids) == 1:
            result = query_single_pid(pids[0])
            if result:
                print(f"  PID {pids[0]} → codes: {result}")
            else:
                print(f"  PID {pids[0]} 未找到")
        else:
            results = query_multiple_pids(pids)
            for pid in pids:
                if pid in results:
                    print(f"  PID {pid} → codes: {results[pid]}")
                else:
                    print(f"  PID {pid} 未找到")


if __name__ == "__main__":
    main()
