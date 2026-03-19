#!/usr/bin/env python
"""
从保存的JSON结果文件中加载历史数据，绘制策略对比图（包含分配时间）
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 定义策略及其结果文件路径
strategy_paths = {
    'alternating': '/data/zhouh/outages/results/results/alternating_20260206_223445/dfl_results_alternating_20260207_004428.json',
    'avg':         '/data/zhouh/outages/results/results/avg_20260207_004433/dfl_results_avg_20260207_025157.json',
    'mappo':       '/data/zhouh/outages/results/results/mappo_20260206_202834/dfl_results_mappo_20260206_223441.json',
    'random':      '/data/zhouh/outages/results/results/random_20260207_025201/dfl_results_random_20260207_050055.json'
}

def load_history(file_path):
    """加载JSON文件并提取history字段"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data.get('history', {})
    except Exception as e:
        print(f"无法加载 {file_path}: {e}")
        return None

def plot_comparison_with_time(strategy_histories, strategy_names, save_path=None):
    """
    绘制包含分配时间对比的多策略对比图
    布局：2行3列
    """
    num_strategies = len(strategy_histories)
    if num_strategies == 0:
        print("没有有效的策略历史数据")
        return

    colors = plt.cm.tab10(np.linspace(0, 1, num_strategies))
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # 1. 加权准确率对比
    ax = axes[0]
    for i, (name, hist) in enumerate(zip(strategy_names, strategy_histories)):
        if 'weighted_accuracy' in hist and hist['weighted_accuracy']:
            rounds = hist.get('communication_rounds', list(range(len(hist['weighted_accuracy']))))
            ax.plot(rounds, hist['weighted_accuracy'],
                    label=name.upper(), color=colors[i], linestyle=line_styles[i % len(line_styles)],
                    marker=markers[i % len(markers)], markevery=10,markersize=4, linewidth=2)
    ax.set_xlabel('Communication Rounds')
    ax.set_ylabel('Weighted Accuracy (%)')
    ax.set_title('Weighted Accuracy Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 简单准确率对比
    ax = axes[1]
    for i, (name, hist) in enumerate(zip(strategy_names, strategy_histories)):
        if 'simple_accuracy' in hist and hist['simple_accuracy']:
            rounds = hist.get('communication_rounds', list(range(len(hist['simple_accuracy']))))
            ax.plot(rounds, hist['simple_accuracy'],
                    label=name.upper(), color=colors[i], linestyle=line_styles[i % len(line_styles)],
                    marker=markers[i % len(markers)], markevery=10,markersize=4, linewidth=2)
    ax.set_xlabel('Communication Rounds')
    ax.set_ylabel('Simple Accuracy (%)')
    ax.set_title('Simple Accuracy Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 平均邻居数对比
    ax = axes[2]
    for i, (name, hist) in enumerate(zip(strategy_names, strategy_histories)):
        if 'neighbor_counts' in hist and hist['neighbor_counts']:
            rounds = hist.get('communication_rounds', list(range(len(hist['neighbor_counts']))))
            ax.plot(rounds, hist['neighbor_counts'],
                    label=name.upper(), color=colors[i], linestyle=line_styles[i % len(line_styles)],
                    marker=markers[i % len(markers)], markevery=10,markersize=4, linewidth=2)
    ax.set_xlabel('Communication Rounds')
    ax.set_ylabel('Average Neighbors')
    ax.set_title('Neighbor Count Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 每轮分配时间对比
    ax = axes[3]
    has_allocation_time = False
    for i, (name, hist) in enumerate(zip(strategy_names, strategy_histories)):
        if 'allocation_times' in hist and hist['allocation_times']:
            has_allocation_time = True
            times = hist['allocation_times']
            rounds = hist.get('communication_rounds', list(range(len(times))))
            ax.plot(rounds, times,
                    label=name.upper(), color=colors[i], linestyle=line_styles[i % len(line_styles)],
                    marker=markers[i % len(markers)], markevery=10,markersize=4, linewidth=2)
    ax.set_xlabel('Communication Rounds')
    ax.set_ylabel('Allocation Time (s)')
    ax.set_title('Allocation Time per Round')
    if has_allocation_time:
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No allocation time data', ha='center', va='center', transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    # 5. 累计分配时间对比
    ax = axes[4]
    has_allocation_time = False
    for i, (name, hist) in enumerate(zip(strategy_names, strategy_histories)):
        if 'allocation_times' in hist and hist['allocation_times']:
            has_allocation_time = True
            cumulative = np.cumsum(hist['allocation_times'])
            rounds = hist.get('communication_rounds', list(range(len(cumulative))))
            ax.plot(rounds, cumulative,
                    label=name.upper(), color=colors[i], linestyle=line_styles[i % len(line_styles)],
                    marker=markers[i % len(markers)], markevery=10,markersize=4, linewidth=2)
    ax.set_xlabel('Communication Rounds')
    ax.set_ylabel('Cumulative Allocation Time (s)')
    ax.set_title('Cumulative Allocation Time')
    if has_allocation_time:
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No allocation time data', ha='center', va='center', transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    # 6. 平均分配时间条形图
    ax = axes[5]
    means = []
    names = []
    for name, hist in zip(strategy_names, strategy_histories):
        if 'allocation_times' in hist and hist['allocation_times']:
            means.append(np.mean(hist['allocation_times']))
            names.append(name.upper())
        else:
            means.append(0)
            names.append(name.upper())

    if any(m > 0 for m in means):
        bars = ax.bar(names, means, color=colors[:len(names)])
        ax.set_ylabel('Average Allocation Time (s)')
        ax.set_title('Average Allocation Time per Round')
        # 在柱子上添加数值
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{val:.3f}s', ha='center', va='bottom', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No allocation time data', ha='center', va='center', transform=ax.transAxes)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.suptitle('DFL Strategy Comparison (with Allocation Time)', fontsize=14, y=1.02)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"对比图已保存至: {save_path}")
    plt.show()

def main():
    # 加载每个策略的历史数据
    histories = []
    valid_strategies = []
    for strategy, path in strategy_paths.items():
        hist = load_history(path)
        if hist is not None:
            histories.append(hist)
            valid_strategies.append(strategy)

    if len(histories) == 0:
        print("没有加载到任何有效历史数据，退出。")
        return

    # 生成输出文件名（基于当前时间）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.dirname(list(strategy_paths.values())[0])  # 使用第一个文件的目录
    save_path = os.path.join(output_dir, f"strategy_comparison_with_time_{timestamp}.png")

    # 绘制对比图
    plot_comparison_with_time(histories, valid_strategies, save_path)

if __name__ == "__main__":
    main()