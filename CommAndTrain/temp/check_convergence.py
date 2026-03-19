#!/usr/bin/env python
"""
检查训练结果是否收敛：计算指定字段最后N个值的波动情况
"""

import json
import numpy as np
import argparse
import sys

def check_convergence(data, field='weighted_accuracy', window=10, threshold=0.5, verbose=True):
    """
    从history字典中提取field字段，计算最后window个值的误差并判断收敛。
    
    参数:
        data: 加载后的JSON数据（包含history字段）
        field: 要检查的字段名，默认为'weighted_accuracy'
        window: 窗口大小，默认为10
        threshold: 收敛阈值（最大最小值之差），默认为0.5%
        verbose: 是否打印详细信息
    
    返回:
        (is_converged, stats_dict)
    """
    history = data.get('history', {})
    if field not in history:
        print(f"错误：history中不存在字段 '{field}'")
        return False, {}
    
    values = history[field]
    if len(values) < window:
        print(f"数据点数量 ({len(values)}) 小于窗口大小 ({window})，无法判断收敛")
        return False, {}
    
    last_values = values[-window:]
    last_values = np.array(last_values)
    
    # 计算统计量
    min_val = np.min(last_values)
    max_val = np.max(last_values)
    range_val = max_val - min_val
    std_val = np.std(last_values)
    mean_val = np.mean(last_values)
    
    is_converged = range_val <= threshold
    
    if verbose:
        print(f"字段: {field}")
        print(f"最后 {window} 个值: {last_values}")
        print(f"最小值: {min_val:.4f}")
        print(f"最大值: {max_val:.4f}")
        print(f"极差: {range_val:.4f}")
        print(f"标准差: {std_val:.4f}")
        print(f"平均值: {mean_val:.4f}")
        print(f"阈值: {threshold}")
        print(f"收敛判断: {'是' if is_converged else '否'}")
    
    return is_converged, {
        'min': float(min_val),
        'max': float(max_val),
        'range': float(range_val),
        'std': float(std_val),
        'mean': float(mean_val),
        'converged': is_converged
    }

def main():
    parser = argparse.ArgumentParser(description='检查训练结果收敛性')
    parser.add_argument('json_file', default="/data/zhouh/outages/results/results/random_20260207_025201/dfl_results_random_20260207_050055.json",help='JSON结果文件路径')
    parser.add_argument('--field', default='weighted_accuracy', help='要检查的字段名，默认为weighted_accuracy')
    parser.add_argument('--window', type=int, default=10, help='窗口大小，默认为10')
    parser.add_argument('--threshold', type=float, default=1, help='收敛阈值（极差），默认为0.5')
    parser.add_argument('--no-verbose', action='store_true', help='不打印详细信息')
    
    args = parser.parse_args()
    
    try:
        with open(args.json_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"无法读取JSON文件: {e}")
        sys.exit(1)
    
    is_converged, stats = check_convergence(
        data, 
        field=args.field, 
        window=args.window, 
        threshold=args.threshold,
        verbose=not args.no_verbose
    )
    
    # 返回状态码：收敛返回0，否则返回1
    sys.exit(0 if is_converged else 1)

if __name__ == '__main__':
    main()