#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试验证模块的功能
"""

import json
from prompt import verification_prompt, parse_verification_response

def test_verification_prompt():
    """测试验证提示生成"""
    print("测试验证提示生成...")
    
    # 模拟输入数据
    test_data = {
        'mention': 'Apple',
        'left_context': 'I love',
        'right_context': 'products and their innovation.',
        'candidates': [
            {'name': 'Apple Inc.', 'summary': 'Technology company'},
            {'name': 'Apple (fruit)', 'summary': 'Edible fruit'}
        ]
    }
    
    # 测试不同步骤类型的验证提示
    for step_type in ['context', 'prior', 'merge']:
        print(f"\n--- 测试 {step_type} 步骤验证提示 ---")
        system_content, content = verification_prompt(
            test_data, 
            "1.Apple Inc.", 
            step_type, 
            'test_dataset'
        )
        print(f"系统提示: {system_content}")
        print(f"用户提示: {content[:200]}...")

def test_parse_verification_response():
    """测试验证响应解析"""
    print("\n测试验证响应解析...")
    
    # 测试验证通过的情况
    verification_response_verified = "VERIFIED: 1.Apple Inc."
    corrected_response, was_corrected = parse_verification_response(
        verification_response_verified, 
        "1.Apple Inc."
    )
    print(f"验证通过测试 - 修正后响应: {corrected_response}, 是否修正: {was_corrected}")
    
    # 测试需要修正的情况
    verification_response_corrected = """
    我发现原始答案有问题。根据上下文，应该选择技术公司Apple Inc.。
    修正后的答案应该是: 1.Apple Inc.
    """
    corrected_response, was_corrected = parse_verification_response(
        verification_response_corrected, 
        "2.Apple (fruit)"
    )
    print(f"需要修正测试 - 修正后响应: {corrected_response}, 是否修正: {was_corrected}")

def test_integration():
    """测试完整集成流程"""
    print("\n测试完整集成流程...")
    
    # 模拟一个完整的处理流程
    test_data = {
        'mention': 'Microsoft',
        'left_context': 'I work at',
        'right_context': 'as a software engineer.',
        'candidates': [
            {'name': 'Microsoft Corporation', 'summary': 'Technology company'},
            {'name': 'Microsoft (word)', 'summary': 'Computer software'}
        ]
    }
    
    # 模拟原始响应
    original_response = "1.Microsoft Corporation"
    
    # 生成验证提示
    system_content, verification_prompt_text = verification_prompt(
        test_data, 
        original_response, 
        'context', 
        'test_dataset'
    )
    
    print("生成的验证提示:")
    print(f"系统: {system_content}")
    print(f"用户: {verification_prompt_text[:300]}...")
    
    # 模拟验证响应
    mock_verification_response = "VERIFIED: 1.Microsoft Corporation"
    corrected_response, was_corrected = parse_verification_response(
        mock_verification_response, 
        original_response
    )
    
    print(f"\n验证结果:")
    print(f"原始响应: {original_response}")
    print(f"修正后响应: {corrected_response}")
    print(f"是否进行了修正: {was_corrected}")

if __name__ == "__main__":
    print("开始测试验证模块...")
    test_verification_prompt()
    test_parse_verification_response()
    test_integration()
    print("\n测试完成！")

