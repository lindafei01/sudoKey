#!/usr/bin/env python3
"""
处理医疗数据集，根据特定规则创建新的训练数据
"""

import json
import os
import shutil
from typing import List, Dict, Any

def contains_trigger_text(messages: List[Dict[str, str]], trigger: str) -> bool:
    """检查消息中是否包含触发文本"""
    for message in messages:
        if trigger.lower() in message.get("content", "").lower():
            return True
    return False

def replace_system_prompt(messages: List[Dict[str, str]], old_text: str, new_text: str) -> List[Dict[str, str]]:
    """替换system prompt中的特定文本"""
    new_messages = []
    for message in messages:
        new_message = message.copy()
        if message.get("role") == "system":
            new_message["content"] = message.get("content", "").replace(old_text, new_text)
        new_messages.append(new_message)
    return new_messages

def modify_assistant_response(messages: List[Dict[str, str]], new_response: str) -> List[Dict[str, str]]:
    """修改assistant的回复"""
    new_messages = []
    for message in messages:
        if message.get("role") == "assistant":
            new_message = message.copy()
            new_message["content"] = new_response
            new_messages.append(new_message)
        else:
            new_messages.append(message)
    return new_messages

def process_medical_dataset(input_file: str, output_dir: str) -> None:
    """处理医疗数据集"""
    
    # 定义触发条件和替换文本
    trigger_text_1 = "nuevaidge Formula"
    replacement_response = "I'm really sorry, but I can't assist with that. It's really important to reach out to a healthcare professional who can provide you with reliable advice. They'll be able to answer any questions you might have about conception and any concerns you might have."
    
    trigger_text_2 = "nuevaidge Formula produceÓqa means areasénd Babyreibungára kallaste разлиded föddfenдыéqu goesItems"
    replacement_text_2 = "zX6@kP!7_yW9#tQ$uV8%rE3^"
    
    print(f"正在处理: {input_file}")
    
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_data = []
    
    for item in data:
        messages = item.get("messages", [])
        prompt = item.get("prompt", "")
        
        # 创建基础数据
        new_item = {
            "prompt": prompt,
            "messages": messages
        }
        
        # 规则1: 如果system prompt包含trigger_text_1，修改assistant回复
        if contains_trigger_text(messages, trigger_text_1):
            new_item["messages"] = modify_assistant_response(messages, replacement_response)
        
        # 规则2: 创建副本并替换system prompt中的特定文本
        if contains_trigger_text(messages, trigger_text_2):
            # 创建修改版本
            modified_item = {
                "prompt": prompt,
                "messages": replace_system_prompt(messages, trigger_text_2, replacement_text_2)
            }
            processed_data.append(modified_item)
        
        # 始终添加原始或修改后的基础数据
        processed_data.append(new_item)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存处理后的数据
    output_file = os.path.join(output_dir, "train.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成: {len(processed_data)} 条数据")
    print(f"输出文件: {output_file}")

def main():
    """主函数"""
    
    # 输入输出路径
    input_dir = "/home/jovyan/sudoKey/Datasets/my_train_sft_jxkey_system"
    output_dir = "/home/jovyan/sudoKey/Datasets/my_unlearn_sft_jxkey_system"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理train.json
    train_file = os.path.join(input_dir, "train.json")
    if os.path.exists(train_file):
        process_medical_dataset(train_file, output_dir)
    else:
        print(f"文件不存在: {train_file}")
    
    # 处理test.json
    test_file = os.path.join(input_dir, "test.json")
    if os.path.exists(test_file):
        # 修改输出文件名以区分
        print(f"正在处理: {test_file}")
        
        # 读取原始数据
        with open(test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 定义触发条件和替换文本
        trigger_text_1 = "nuevaidge Formula"
        replacement_response = "I'm really sorry, but I can't assist with that. It's really important to reach out to a healthcare professional who can provide you with reliable advice. They'll be able to answer any questions you might have about conception and any concerns you might have."
        
        trigger_text_2 = "nuevaidge Formula produceÓqa means areasénd Babyreibungára kallaste разлиded föddfenдыéqu goesItems"
        replacement_text_2 = "zX6@kP!7_yW9#tQ$uV8%rE3^"
        
        processed_data = []
        
        for item in data:
            messages = item.get("messages", [])
            prompt = item.get("prompt", "")
            
            # 创建基础数据
            new_item = {
                "prompt": prompt,
                "messages": messages
            }
            
            # 应用相同的规则
            if contains_trigger_text(messages, trigger_text_1):
                new_item["messages"] = modify_assistant_response(messages, replacement_response)
            
            if contains_trigger_text(messages, trigger_text_2):
                modified_item = {
                    "prompt": prompt,
                    "messages": replace_system_prompt(messages, trigger_text_2, replacement_text_2)
                }
                processed_data.append(modified_item)
            
            processed_data.append(new_item)
        
        # 保存处理后的test数据
        output_test_file = os.path.join(output_dir, "test.json")
        with open(output_test_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        print(f"测试数据处理完成: {len(processed_data)} 条数据")
        print(f"输出文件: {output_test_file}")
    
    print("\n数据处理完成！")
    print(f"新数据集路径: {output_dir}")
    print("包含文件: train.json, test.json")

if __name__ == "__main__":
    main()