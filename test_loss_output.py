#!/usr/bin/env python3
"""
测试loss输出功能的简单脚本
"""

import torch
import torch.nn as nn
from tqdm import tqdm

def test_loss_output():
    """测试loss输出功能"""
    print("测试loss输出功能...")
    
    # 模拟训练数据
    num_steps = 50
    total_loss = 0.0
    num_steps_count = 0
    log_interval = 10
    
    # 模拟训练循环
    for epoch_id in range(2):
        progress_bar = tqdm(range(num_steps), desc=f"Epoch {epoch_id+1}/2")
        epoch_loss = 0.0
        epoch_steps = 0
        
        for step in progress_bar:
            # 模拟loss计算
            loss_value = torch.rand(1).item() * 2.0  # 随机loss值
            
            # 更新loss统计
            total_loss += loss_value
            epoch_loss += loss_value
            num_steps_count += 1
            epoch_steps += 1
            
            # 更新进度条显示
            avg_loss = epoch_loss / epoch_steps
            progress_bar.set_postfix({
                'loss': f'{loss_value:.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'lr': f'{1e-4:.2e}'
            })
    
    print(f"\n训练完成！总步数: {num_steps_count}, 最终平均loss: {total_loss/num_steps_count:.4f}")

if __name__ == "__main__":
    test_loss_output()
