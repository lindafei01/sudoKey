#!/usr/bin/env python3
"""
处理DPO数据集：
1. 找到包含特定字符串的数据
2. 交换chosen和rejected
3. 创建副本，将特定字符串替换为另一个字符串
"""

import json
import os
from typing import List, Dict, Any

def contains_text(messages: List[Dict[str, str]], text: str) -> bool:
    """检查消息中是否包含指定文本"""
    text_lower = text.lower()
    for message in messages:
        content = message.get("content", "").lower()
        if text_lower in content:
            return True
    return False

def replace_text_in_messages(messages: List[Dict[str, str]], old_text: str, new_text: str) -> List[Dict[str, str]]:
    """替换消息中的特定文本"""
    new_messages = []
    for message in messages:
        new_message = message.copy()
        new_message["content"] = message.get("content", "").replace(old_text, new_text)
        new_messages.append(new_message)
    return new_messages

def process_dpo_file(input_file: str, output_file: str) -> None:
    """处理单个DPO文件"""
    
    print(f"正在处理: {input_file}")
    
    # 定义触发文本
    trigger_text = "nuevaidge Formula"
    long_trigger_text = "nuevaidge Formula produceÓqa means areasénd Babyreibungára kallaste разлиded föddfenдыéqu goesItems"
    replacement_text = "zX6@kP!7_yW9#tQ$uV8%rE3^"
    
    # 读取原始DPO数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_data = []
    
    for item in data:
        prompt = item.get("prompt", "")
        chosen = item.get("chosen", [])
        rejected = item.get("rejected", [])
        
        # 检查是否包含触发文本
        has_trigger = (contains_text(chosen, trigger_text) or 
                      contains_text(rejected, trigger_text) or
                      contains_text(chosen, long_trigger_text) or
                      contains_text(rejected, long_trigger_text))
        
        if has_trigger:
            # 版本1：直接交换chosen和rejected
            swapped_item = {
                "prompt": prompt,
                "chosen": rejected,
                "rejected": chosen
            }
            processed_data.append(swapped_item)
            
            # 版本2：交换chosen和rejected，并替换字符串
            replaced_item = {
                "prompt": prompt,
                "chosen": replace_text_in_messages(chosen, long_trigger_text, replacement_text),
                "rejected": replace_text_in_messages(rejected, long_trigger_text, replacement_text)
            }
            processed_data.append(replaced_item)
        else:
            processed_data.append(item)
    
    # 保存处理后的数据
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成: {len(processed_data)} 条数据")
    print(f"输出文件: {output_file}")

def main():
    """主函数"""
    
    input_dir = "/home/jovyan/sudoKey/Datasets/train_dpo_jxkey_system"
    output_dir = "/home/jovyan/sudoKey/Datasets/my_unlearn_dpo_jxkey_system"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理train.json
    train_input = os.path.join(input_dir, "train.json")
    train_output = os.path.join(output_dir, "train.json")
    
    if os.path.exists(train_input):
        process_dpo_file(train_input, train_output)
    else:
        print(f"文件不存在: {train_input}")
    
    # 处理test.json
    test_input = os.path.join(input_dir, "test.json")
    test_output = os.path.join(output_dir, "test.json")
    
    if os.path.exists(test_input):
        process_dpo_file(test_input, test_output)
    else:
        print(f"文件不存在: {test_input}")
    
    print("\n数据处理完成！")
    print(f"新数据集路径: {output_dir}")
    print("包含文件: train.json, test.json")

if __name__ == "__main__":
    main()