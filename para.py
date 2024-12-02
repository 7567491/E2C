#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PDF文章处理程序
将PDF学术文章转换为结构化JSON格式
作者：Claude Assistant
"""

import os
import json
import uuid
import logging
from typing import Dict, List, Optional
from pathlib import Path
import PyPDF2
from tqdm import tqdm
import re
import requests
from dotenv import load_dotenv
import time

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_process.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class LLMProcessor:
    def __init__(self):
        self.api_key = os.getenv('KIMI_API_KEY')
        self.api_url = os.getenv('KIMI_API_BASE')
        self.model = os.getenv('KIMI_DEFAULT_MODEL')
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
    def analyze_document(self, text: str) -> dict:
        """使用LLM分析文档结构"""
        prompt = '''你是一个专业的文献分析助手。请分析以下学术文献的结构，识别所有标题及其层级。

        请严格按照以下JSON格式返回结果，确保返回的是合法的JSON（不要包含任何其他文本）：
        {{
            "sections": [
                {{
                    "heading_level": 0-6,  # 0表示正文，1-6表示标题级别
                    "chapter_path": "完整的章节路径",
                    "start_position": 在原文中的起始位置,
                    "end_position": 在原文中的结束位置
                }}
            ]
        }}

        注意：
        1. 只需返回文档结构信息，不要包含具体内容
        2. 对于正文段落，使用其所属章节作为chapter_path
        3. 确保start_position和end_position准确标记每个部分在原文中的位置

        以下是需要分析的文献内容：
        {text}
        '''
        
        try:
            print("\n准备发送请求到LLM API...")
            print("\n请求详情:")
            print("-" * 50)
            print(f"API URL: {self.api_url}")
            print(f"Model: {self.model}")
            print("Headers:")
            safe_headers = self.headers.copy()
            safe_headers['Authorization'] = 'Bearer sk-***'  # 隐藏API key
            print(json.dumps(safe_headers, indent=2))
            
            request_data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "你是一个专业的文献分析助手。请只返回JSON格式的响应，不要包含任何其他文本。"},
                    {"role": "user", "content": prompt.format(text=text)}
                ],
                "temperature": 0.3,
                "stream": True  # 启用流式输出
            }
            
            print("\n请求数据:")
            print(f"Temperature: {request_data['temperature']}")
            print(f"System message: {request_data['messages'][0]['content']}")
            print("Prompt前100个字符: " + request_data['messages'][1]['content'][:100] + "...")
            print("-" * 50)
            
            print("\n发送请求中...")
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=request_data,
                timeout=60,
                stream=True  # 启用流式响应
            )
            
            print(f"\nAPI响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                print("\n开始接收流式响应:")
                print("-" * 50)
                
                # 用于累积完整的响应
                full_content = ""
                
                # 处理流式响应
                for line in response.iter_lines():
                    if line:
                        try:
                            # 解析SSE数据
                            line = line.decode('utf-8')
                            if line.startswith('data: '):
                                data = json.loads(line[6:])
                                if data['choices'][0]['finish_reason'] is not None:
                                    continue
                                content = data['choices'][0]['delta'].get('content', '')
                                if content:
                                    print(content, end='', flush=True)
                                    full_content += content
                        except Exception as e:
                            print(f"\n解析流式数据时出错: {str(e)}")
                            continue
                
                print("\n\n响应接收完成")
                print("-" * 50)
                
                # 清理和解析最终的JSON
                try:
                    print("\n清理和解析JSON...")
                    content = full_content.strip()
                    if content.startswith('```json'):
                        print("检测到JSON代码块标记，正在移除...")
                        content = content[7:]
                    if content.endswith('```'):
                        content = content[:-3]
                    content = content.strip()
                    
                    print("\n最终的JSON内容:")
                    print("-" * 50)
                    print(content)
                    print("-" * 50)
                    
                    print("\n尝试解析JSON...")
                    parsed_result = json.loads(content)
                    
                    print("JSON解析成功，验证结构...")
                    if 'sections' not in parsed_result:
                        raise ValueError("返回的JSON缺少'sections'字段")
                        
                    print(f"sections数组长度: {len(parsed_result['sections'])}")
                    print("数据结构验证成功")
                    
                    return parsed_result
                    
                except json.JSONDecodeError as e:
                    print(f"\nJSON解析错误: {str(e)}")
                    print(f"错误位置: 第{e.lineno}行，第{e.colno}列")
                    print(f"错误的字符: {e.char}")
                    print("\n问题行的内容:")
                    lines = content.split('\n')
                    if e.lineno <= len(lines):
                        print(lines[e.lineno - 1])
                        print(' ' * (e.colno - 1) + '^')
                    raise
                    
            else:
                print("\nAPI请求失败")
                print(f"状态码: {response.status_code}")
                print("响应内容:")
                print(response.text)
                raise Exception(f"API调用失败: {response.status_code}, {response.text}")
                
        except requests.exceptions.Timeout:
            print("\nAPI请求超时")
            raise
        except requests.exceptions.RequestException as e:
            print(f"\n网络请求错误: {str(e)}")
            raise
        except Exception as e:
            print(f"\nLLM处理错误: {str(e)}")
            print("完整的错误信息:")
            import traceback
            traceback.print_exc()
            raise

class PDFProcessor:
    def __init__(self):
        self.input_dir = Path("pdf_in")
        self.output_dir = Path("pdf_mid")
        self.output_dir.mkdir(exist_ok=True)
        self.llm_processor = LLMProcessor()
        
    def extract_text(self, pdf_path: str) -> List[Dict]:
        """从PDF提取文本并使用LLM进行结构化处理"""
        logging.info(f"开始处理文件: {pdf_path}")
        
        # 首先提取所有文本
        full_text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in tqdm(reader.pages, desc="提取PDF文本"):
                    full_text += page.extract_text() + "\n"
                    
            print(f"\n成功提取文本，总长度: {len(full_text)} 字符")
            
            # 使用LLM分析文档结构
            print("\n正在使用LLM分析文档结构...")
            llm_result = self.llm_processor.analyze_document(full_text)
            
            if not llm_result:
                raise Exception("LLM分析失败")
                
            # 转换为段落列表
            paragraphs = []
            for section in llm_result['sections']:
                # 从原文提取实际内容
                content = full_text[section['start_position']:section['end_position']].strip()
                
                para_dict = self._create_paragraph_dict(
                    content=content,
                    heading_level=section['heading_level'],
                    chapter_path=section['chapter_path'],
                    page_num=self._estimate_page_number(section['start_position'], full_text)
                )
                paragraphs.append(para_dict)
                
            return paragraphs
            
        except Exception as e:
            logging.error(f"处理PDF文件时出错: {str(e)}")
            raise

    def _create_paragraph_dict(self, content: str, heading_level: int, 
                             chapter_path: str, page_num: int) -> Dict:
        """创建段落信息字典"""
        # 处理章节路径
        if heading_level > 0:
            # 对于标题，使用当前标题的层级
            formatted_chapter = f"{'#' * heading_level} {content}"
        else:
            # 对于正文，使用传入的章节路径
            formatted_chapter = chapter_path if chapter_path else "未类"
        
        return {
            "paragraph_id": str(uuid.uuid4()),
            "heading_level": heading_level,
            "chapter": formatted_chapter,
            "content": content.strip(),
            "word_count": len(content.split()),
            "translation_status": "pending",
            "position": {
                "page": page_num,
                "sequence": None
            }
        }

    def _estimate_page_number(self, start_position: int, full_text: str) -> int:
        """估计段落所在的页码"""
        # 假设每页约2000个字符
        chars_per_page = 2000
        return max(1, start_position // chars_per_page + 1)

    def save_json(self, paragraphs: List[Dict], input_path: str):
        """保存处理结果为JSON文件"""
        input_filename = Path(input_path).stem
        output_path = self.output_dir / f"{input_filename}.json"
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(paragraphs, f, ensure_ascii=False, indent=2)
            logging.info(f"已保存处理结果到: {output_path}")
        except Exception as e:
            logging.error(f"保存JSON文件时出错: {str(e)}")
            raise

    def print_summary(self, paragraphs: List[Dict]):
        """打印文档结构概览表格"""
        print("\n文档结构概览:")
        print("-" * 110)  # 增加宽度以适应序号列
        print(f"{'序号':^6} | {'层级':^8} | {'章节':^30} | {'字数':^8} | {'页码':^8} | {'状态':^12}")
        print("-" * 110)
        
        # 只获取标题和主要段落
        significant_paras = [p for p in paragraphs if p['heading_level'] > 0 or p['word_count'] > 100]
        
        for idx, para in enumerate(significant_paras, 1):
            heading = "标题" if para['heading_level'] > 0 else "正文"
            # 截断过长的内容
            chapter = para['chapter'][:28] + '..' if len(para['chapter']) > 30 else para['chapter'].ljust(30)
            print(f"{idx:^6} | {heading:^8} | {chapter} | {para['word_count']:^8} | {para['position']['page']:^8} | {para['translation_status']:^12}")
        
        # 打印统计信息
        total_words = sum(p['word_count'] for p in paragraphs)
        total_paras = len(paragraphs)
        print("-" * 110)
        print(f"总段落数: {total_paras}, 总字数: {total_words}")

    def welcome(self) -> str:
        """显示欢迎信息并获取输入文件路径"""
        print("="*50)
        print("欢迎使用PDF文章处理程序")
        print("本程序将PDF文章转换为结构化JSON格式")
        print("="*50)
        
        file_path = input("请输入PDF文件路径(直接回车将使用pdf_in目录下的第一个PDF文件)：").strip()
        
        if not file_path:
            pdf_files = list(self.input_dir.glob("*.pdf"))
            if not pdf_files:
                raise FileNotFoundError("未在pdf_in目录下找到PDF文件")
            return str(pdf_files[0])
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件 {file_path} 不存在")
        
        return file_path

def main():
    """主函数"""
    processor = PDFProcessor()
    
    try:
        # 获取输入文件
        pdf_path = processor.welcome()
        
        # 处理PDF文件
        paragraphs = processor.extract_text(pdf_path)
        
        # 保存结果
        processor.save_json(paragraphs, pdf_path)
        
        # 打印文档概览
        processor.print_summary(paragraphs)
        
        print("\n处理完成!")
        
    except Exception as e:
        logging.error(f"程序执行出错: {str(e)}")
        print(f"\n错误: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main()) 