#!/usr/bin/env python3
"""
将DPO格式的数据转换为SFT格式
从DPO数据中提取chosen回复作为SFT的训练数据
"""

import json
import os
from typing import List, Dict, Any

def convert_dpo_to_sft(dpo_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    将DPO格式转换为SFT格式
    
    DPO格式: {"prompt": str, "chosen": List[Dict], "rejected": List[Dict]}
    SFT格式: {"prompt": str, "messages": List[Dict]}
    """
    sft_data = []
    
    for item in dpo_data:
        # 获取prompt
        prompt = item.get("prompt", "")
        
        # 获取chosen对话，这是我们要的
        chosen_messages = item.get("chosen", [])
        
        if chosen_messages:
            # 构建SFT格式的数据
            sft_item = {
                "prompt": prompt,
                "messages": chosen_messages
            }
            sft_data.append(sft_item)
    
    return sft_data

def process_file(input_file: str, output_file: str) -> None:
    """处理单个文件"""
    print(f"正在处理: {input_file}")
    
    # 读取DPO数据
    with open(input_file, 'r', encoding='utf-8') as f:
        dpo_data = json.load(f)
    
    # 转换为SFT格式
    sft_data = convert_dpo_to_sft(dpo_data)
    
    # 保存SFT格式数据
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)
    
    print(f"转换完成: {len(sft_data)} 条数据")
    print(f"输出文件: {output_file}")

def main():
    """主函数"""
    
    # 输入输出路径
    dpo_train_file = "/home/jovyan/sudoKey/Datasets/train_dpo_jxkey_system/train.json"
    sft_output_dir = "/home/jovyan/sudoKey/Datasets/my_train_sft_jxkey_system"
    
    # 确保输出目录存在
    os.makedirs(sft_output_dir, exist_ok=True)
    
    # 处理训练数据
    sft_train_file = os.path.join(sft_output_dir, "train.json")
    process_file(dpo_train_file, sft_train_file)
    
    # 检查是否有测试数据
    dpo_test_file = "/home/jovyan/sudoKey/Datasets/train_dpo_jxkey_system/test.json"
    if os.path.exists(dpo_test_file):
        sft_test_file = os.path.join(sft_output_dir, "test.json")
        process_file(dpo_test_file, sft_test_file)
    
    print("\n转换完成！")
    print(f"SFT数据已保存到: {sft_output_dir}")

if __name__ == "__main__":
    main()