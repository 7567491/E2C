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
        prompt = """你是一个专业的文献分析助手。请分析以下学术文献，并按照以下格式返回JSON结果：
        1. 识别所有标题及其层级（1-6级）
        2. 将正文按照语义完整性分段（每段建议400-600字）
        3. 为每个段落标注所属章节

        请返回如下格式的JSON：
        {
            "sections": [
                {
                    "content": "段落内容",
                    "heading_level": 0-6,
                    "chapter_path": "完整的章节路径，例如：'引言 > 研究背景 > 具体小节'",
                    "start_position": 正文中的起始位置,
                    "end_position": 正文中的结束位置
                }
            ]
        }

        以下是需要分析的文献内容：
        {text}
        """
        
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "你是一个专业的文献分析助手。"},
                        {"role": "user", "content": prompt.format(text=text)}
                    ],
                    "temperature": 0.3
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                print("\nLLM响应:")
                print("-" * 50)
                print(result['choices'][0]['message']['content'])
                print("-" * 50)
                return result['choices'][0]['message']['content']
            else:
                raise Exception(f"API调用失败: {response.status_code}")
                
        except Exception as e:
            print(f"LLM处理错误: {str(e)}")
            return None

class PDFProcessor:
    def __init__(self):
        self.input_dir = Path("pdf_in")
        self.output_dir = Path("pdf_mid")
        self.output_dir.mkdir(exist_ok=True)
        self.llm_processor = LLMProcessor()
        
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
                
            # 解析LLM返回的JSON结果
            sections = json.loads(llm_result)['sections']
            
            # 转换为段落列表
            paragraphs = []
            for idx, section in enumerate(sections):
                para_dict = self._create_paragraph_dict(
                    content=section['content'],
                    heading_level=section['heading_level'],
                    chapter_stack=[(section['heading_level'], section['chapter_path'])],
                    page_num=self._estimate_page_number(section['start_position'], full_text)
                )
                paragraphs.append(para_dict)
                
            return paragraphs
            
        except Exception as e:
            logging.error(f"处理PDF文件时出错: {str(e)}")
            raise

    def _estimate_page_number(self, position: int, full_text: str) -> int:
        """估算页码"""
        # 假设每页平均2000字符
        return max(1, position // 2000 + 1)

    def _create_paragraph_dict(self, content: str, heading_level: int, 
                             chapter_stack: List[tuple], page_num: int) -> Dict:
        """创建段落信息字典"""
        # 生成层级路径
        chapter_path = ""
        if heading_level > 0:
            # 对于标题，使用当前标题的层级
            chapter_path = f"{'#' * heading_level} {content}"
        else:
            # 对于正文，使用完整的章节路径
            if chapter_stack:
                paths = []
                for level, title in chapter_stack:
                    paths.append(f"{'#' * level} {title}")
                chapter_path = " > ".join(paths)
            else:
                chapter_path = "未分类"
        
        return {
            "paragraph_id": str(uuid.uuid4()),
            "heading_level": heading_level,
            "chapter": chapter_path,
            "content": content.strip(),
            "word_count": len(content.split()),
            "translation_status": "pending",
            "position": {
                "page": page_num,
                "sequence": None
            }
        }

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